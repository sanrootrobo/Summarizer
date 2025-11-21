# Final Enhanced Script: Summary_On_Sterioids2.py
import os
import requests
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote, unquote
from concurrent.futures import ThreadPoolExecutor
import re
import argparse
from time import sleep
import json
from datetime import datetime
import html2text
import random

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "gemini-2.5-pro"
SEARCH_MODEL_NAME = "gemini-2.5-flash"
API_KEY_FILE = "geminaikey"
GOOGLE_API_KEY_FILE = "googlesearchapi"
GOOGLE_CX_FILE = "googlecx"
REQUEST_TIMEOUT = 15
MAX_CONTENT_CHARS = 500000
MAX_SEARCH_RESULTS = 8
MAX_RESEARCH_PAGES = 20

# --- ANTI-BOT CONFIGURATION ---
# A list of realistic User-Agents to rotate for each session
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

# Optional Proxy Configuration (uncomment and fill to use)
# PROXIES = {
#    "http": "http://user:pass@host:port",
#    "https": "http://user:pass@host:port",
# }
PROXIES = None # Set to your proxy dictionary to enable


class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str = "geminaikey") -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"Gemini API key file not found at '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10:
                logger.error("Invalid Gemini API key found in file.")
                return None
            logger.info("Gemini API key loaded successfully.")
            return api_key
        except Exception as e:
            logger.error(f"Error reading Gemini API key: {e}")
            return None
    
    @staticmethod
    def get_google_search_credentials() -> Tuple[Optional[str], Optional[str]]:
        try:
            api_key_path = Path(GOOGLE_API_KEY_FILE)
            if not api_key_path.exists():
                logger.warning(f"Google Search API key file not found at '{GOOGLE_API_KEY_FILE}'.")
                return None, None
            
            with open(api_key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            
            cx_path = Path(GOOGLE_CX_FILE)
            if not cx_path.exists():
                logger.warning(f"Google CX file not found at '{GOOGLE_CX_FILE}'.")
                return None, None
                
            with open(cx_path, 'r', encoding='utf-8') as f:
                cx_id = f.read().strip()
            
            if not api_key or len(api_key) < 10 or not cx_id:
                logger.warning("Invalid Google Search credentials. Skipping research enhancement.")
                return None, None
                
            logger.info("Google Search API credentials loaded successfully.")
            return api_key, cx_id
            
        except Exception as e:
            logger.warning(f"Error reading Google Search credentials: {e}.")
            return None, None

class ContentAnalyzer:
    @staticmethod
    def categorize_content(url: str, content: str) -> Dict[str, any]:
        categories = {
            'type': 'general',
            'priority': 'medium',
            'contains_code': False,
            'contains_examples': False,
            'is_api_reference': False,
            'is_tutorial': False,
            'is_concept': False,
            'is_research': False,
            'word_count': len(content.split()),
        }
        url_lower = url.lower()
        content_lower = content.lower()
        
        research_indicators = ['stackoverflow.com', 'github.com', 'medium.com', 'dev.to', '/tutorial', '/guide', '/example', '/blog']
        if any(indicator in url_lower for indicator in research_indicators):
            categories['is_research'] = True
            categories['priority'] = 'high'
        
        if any(p in url_lower for p in ['api', 'reference', 'docs/api']):
            categories['type'] = 'api_reference'
            categories['is_api_reference'] = True
            categories['priority'] = 'high'
        elif any(p in url_lower for p in ['tutorial', 'guide', 'getting-started']):
            categories['type'] = 'tutorial'
            categories['is_tutorial'] = True
            categories['priority'] = 'high'
        elif any(p in url_lower for p in ['concept', 'overview', 'introduction']):
            categories['type'] = 'concept'
            categories['is_concept'] = True
            categories['priority'] = 'high'
        elif any(p in url_lower for p in ['example', 'sample', 'demo']):
            categories['type'] = 'example'
            categories['contains_examples'] = True
            categories['priority'] = 'medium'
        
        if any(p in content_lower for p in ['```', 'code', 'function', 'class', 'import']):
            categories['contains_code'] = True
        
        if any(p in content_lower for p in ['example', 'for instance', 'here\'s how']):
            categories['contains_examples'] = True
            
        return categories

class ResearchQueryGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=SEARCH_MODEL_NAME, google_api_key=api_key, temperature=0.7)
        self.output_parser = StrOutputParser()
    
    def generate_research_queries(self, content_insights: Dict[str, str]) -> List[str]:
        prompt = ChatPromptTemplate.from_template("""
You are a research analyst generating search queries to find complementary resources for the given documentation.
Based on this analysis, generate 10-12 specific, context-aware search queries.

**CONTENT ANALYSIS:**
- Website: {main_url}
- Main Technology: {main_technology}
- Key Topics: {main_topics}
- Gaps/Weaknesses: {content_gaps}

**YOUR MISSION:**
Generate a JSON array of search queries to find tutorials, real-world examples, and solutions to common problems.

**CRITICAL REQUIREMENTS:**
- Focus on practical, actionable resources (Stack Overflow, GitHub, tutorials).
- Use specific feature names, API endpoints, or concepts from the docs.
- Target missing content types (e.g., if no tutorials, search for tutorials).

**AVOID:**
- Generic queries that would just return the original documentation.
- Overly broad queries.

**OUTPUT FORMAT:**
Return ONLY a JSON array of search queries. Example:
["React hooks tutorial", "React context API best practices"]

Generate the queries:
""")
        chain = prompt | self.llm | self.output_parser
        
        try:
            logger.info("🧠 Generating context-aware research queries...")
            result = chain.invoke(content_insights)
            queries = json.loads(result.strip())
            if isinstance(queries, list) and queries:
                logger.info(f"✅ Generated {len(queries)} research queries.")
                return queries[:12]
            return self._generate_fallback_queries(content_insights)
        except Exception as e:
            logger.warning(f"Failed to generate queries with AI: {e}. Using fallback.")
            return self._generate_fallback_queries(content_insights)
    
    def extract_content_insights(self, scraped_content: Dict[str, str], metadata: Dict[str, Dict], main_url: str) -> Dict[str, str]:
        all_text = " ".join(scraped_content.values())
        main_tech = self._extract_main_technology(main_url, all_text)
        
        topics = set()
        content_types = {}
        for meta in metadata.values():
            analysis = meta.get('analysis', {})
            content_type = analysis.get('type', 'general')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            if 'main_topics' in analysis:
                topics.update(analysis['main_topics'])

        gaps = [f"needs more {t} content" for t in ['tutorial', 'example'] if content_types.get(t, 0) < 2]

        return {
            'main_url': main_url,
            'main_technology': main_tech,
            'main_topics': ', '.join(list(topics)[:10]),
            'content_gaps': ', '.join(gaps) or 'comprehensive coverage',
        }

    def _extract_main_technology(self, main_url: str, content: str) -> str:
        domain = urlparse(main_url).netloc.replace('www.', '').lower()
        domain_parts = domain.split('.')
        if len(domain_parts) >= 2:
            potential_tech = domain_parts[0]
            if potential_tech not in ['docs', 'api', 'developer', 'help']:
                return potential_tech
        return 'unknown'

    def _generate_fallback_queries(self, content_insights: Dict[str, str]) -> List[str]:
        main_tech = content_insights.get('main_technology', 'tech')
        return [
            f"{main_tech} tutorial",
            f"{main_tech} real world example",
            f"{main_tech} common issues",
            f"how to use {main_tech} for beginners"
        ]

class GoogleSearchResearcher:
    def __init__(self, api_key: str = None, cx_id: str = None):
        self.api_key = api_key
        self.cx_id = cx_id
        self.google_base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1'
        })
        self.google_requests_made = 0
        self.max_google_requests_per_session = 90
        self.last_google_request_time = 0

    def search_and_extract_urls(self, queries: List[str], exclude_domain: str = None) -> List[str]:
        all_urls = set()
        google_available = bool(self.api_key and self.cx_id)
        
        for i, query in enumerate(queries):
            logger.info(f"🔍 Searching [{i+1}/{len(queries)}]: {query}")
            urls_found = False
            
            if (google_available and self.google_requests_made < self.max_google_requests_per_session):
                try:
                    google_urls = self._search_google(query, exclude_domain)
                    if google_urls:
                        all_urls.update(google_urls)
                        urls_found = True
                        logger.info(f"✅ Google found {len(google_urls)} URLs.")
                except Exception as e:
                    logger.warning(f"❌ Google search failed: {e}")
                    if "429" in str(e):
                        google_available = False
            
            if not urls_found:
                try:
                    logger.info(f"🦆 Using DuckDuckGo fallback for: {query}")
                    ddg_urls = self._search_duckduckgo_improved(query, exclude_domain)
                    if ddg_urls:
                        all_urls.update(ddg_urls)
                        logger.info(f"✅ DuckDuckGo found {len(ddg_urls)} URLs.")
                except Exception as e:
                    logger.warning(f"❌ DuckDuckGo search also failed: {e}")
            
            sleep(1.0 if google_available else 0.8)
        
        return list(all_urls)[:MAX_RESEARCH_PAGES]

    def _search_google(self, query: str, exclude_domain: str = None) -> List[str]:
        current_time = datetime.now().timestamp()
        if current_time - self.last_google_request_time < 1.0:
            sleep(1.0 - (current_time - self.last_google_request_time))
        
        params = {'key': self.api_key, 'cx': self.cx_id, 'q': query, 'num': min(MAX_SEARCH_RESULTS, 10)}
        response = self.session.get(self.google_base_url, params=params, timeout=REQUEST_TIMEOUT, proxies=PROXIES)
        self.last_google_request_time = datetime.now().timestamp()
        self.google_requests_made += 1
        response.raise_for_status()
        
        results = response.json()
        urls = [item['link'] for item in results.get('items', []) if self._is_useful_url(item.get('link', ''), exclude_domain)]
        return urls

    def _search_duckduckgo_improved(self, query: str, exclude_domain: str = None) -> List[str]:
        sleep_time = random.uniform(1.5, 3.5)
        logger.info(f"🦆 Waiting for {sleep_time:.2f} seconds before DDG search...")
        sleep(sleep_time)

        search_url = "https://html.duckduckgo.com/html/"
        enhanced_query = f"{query} (site:stackoverflow.com OR site:github.com OR site:medium.com OR site:dev.to)"
        params = {'q': enhanced_query}
        headers = {'Referer': 'https://duckduckgo.com/'}
        
        response = self.session.get(search_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT, proxies=PROXIES)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []
        
        link_selectors = ['a.result__a', '.result__body a[href^="http"]', 'a[href*="uddg="]']
        
        for selector in link_selectors:
            for link in soup.select(selector):
                href = link.get('href', '')
                if 'uddg=' in href:
                    try:
                        real_url = unquote(href.split('uddg=')[1].split('&')[0])
                        if real_url and self._is_useful_url(real_url, exclude_domain):
                            urls.append(real_url)
                    except Exception:
                        continue
                elif href.startswith('http') and self._is_useful_url(href, exclude_domain):
                    urls.append(href)
            if urls:
                break
        
        return list(dict.fromkeys(urls))

    def _is_useful_url(self, url: str, exclude_domain: str = None) -> bool:
        if not url: return False
        try:
            parsed = urlparse(url)
            if exclude_domain and exclude_domain.lower() in parsed.netloc.lower():
                return False
            
            excluded_ext = ['.pdf', '.zip', '.mp4', '.jpg']
            if any(parsed.path.lower().endswith(ext) for ext in excluded_ext):
                return False
            
            excluded_domains = ['youtube.com', 'twitter.com', 'facebook.com', 'linkedin.com', 'amazon.com', 'ebay.com', 'reddit.com', 'quora.com']
            if any(ex_dom in parsed.netloc.lower() for ex_dom in excluded_domains):
                return False
            
            return True
        except Exception:
            return False

class WebsiteScraper:
    def __init__(self, base_url: str, max_pages: int):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        self.scraped_content = {}
        self.content_metadata = {}
        self.visited_urls = set()
        self.text_maker = html2text.HTML2Text()
        self.text_maker.body_width = 0
        self.text_maker.ignore_links = True
        self.text_maker.ignore_images = True

    def _get_page_content(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT, proxies=PROXIES)
            response.raise_for_status()
            return response.text if 'text/html' in response.headers.get('Content-Type', '') else None
        except requests.RequestException as e:
            logger.warning(f"Could not fetch {url}: {e}")
            return None

    def _parse_content_and_links(self, html: str, page_url: str) -> Tuple[str, List[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title')
        page_title = title.get_text().strip() if title else "Untitled Page"

        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        main_content = soup.find('main') or soup.find('article') or soup.body
        markdown_content = self.text_maker.handle(str(main_content)) if main_content else ""
        
        links = []
        if urlparse(page_url).netloc == self.domain:
            for a in soup.find_all('a', href=True):
                full_url = urljoin(page_url, a['href']).split('#')[0]
                if urlparse(full_url).netloc == self.domain:
                    links.append(full_url)
        
        return markdown_content, list(set(links)), page_title

    def crawl(self, additional_urls: List[str] = None) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        urls_to_visit = {self.base_url}
        if additional_urls:
            urls_to_visit.update(additional_urls)

        while urls_to_visit and (self.max_pages == 0 or len(self.visited_urls) < self.max_pages):
            url = urls_to_visit.pop()
            if url in self.visited_urls:
                continue
            
            self.visited_urls.add(url)
            is_research = urlparse(url).netloc != self.domain
            prefix = "🔬" if is_research else "📄"
            logger.info(f"[{len(self.visited_urls)}] {prefix} Scraping: {url}")
            
            html = self._get_page_content(url)
            if not html:
                continue
            
            text, new_links, title = self._parse_content_and_links(html, url)
            
            if text and len(text.strip()) > 100:
                self.scraped_content[url] = text
                self.content_metadata[url] = {
                    'title': title,
                    'analysis': ContentAnalyzer.categorize_content(url, text),
                    'is_research': is_research
                }
            
            if not is_research:
                for link in new_links:
                    if link not in self.visited_urls:
                        urls_to_visit.add(link)
            
            sleep(random.uniform(0.3, 0.8))
        
        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages.")
        return self.scraped_content, self.content_metadata

class EnhancedNoteGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0.3)
        self.output_parser = StrOutputParser()

    def _create_comprehensive_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
You are an expert technical writer. Transform the following raw documentation and research materials into a comprehensive, multi-layered study guide.

**Content to Process:**
- Website: {website_url}
- Documentation + Research: {content}

**Required Output Structure:**
# 📚 Comprehensive Study Guide: [Main Topic]

## 🎯 Executive Summary
(One-sentence description, target audience, key benefits, prerequisites)

## 🧠 Core Concepts Explained
(ELI5 section, architecture, real-world analogies, common misconceptions)

## 📖 Detailed Reference Guide
(API/Feature Reference, configuration, advanced patterns)

## 💡 Practical Examples & Tutorials
(Quick start, common use cases, integration examples from research)

## 📝 Quick Reference Materials
(Ultimate cheat sheet, command reference, troubleshooting guide from community discussions)

**Guidelines:**
- Seamlessly blend official docs with community wisdom.
- Use examples from research to show multiple approaches.
- Address gaps in docs with community solutions.

Generate the study notes:
""")

    def _chunk_content(self, data: Dict[str, str]) -> List[str]:
        chunks = []
        current_chunk = ""
        for url, content in data.items():
            content_with_header = f"\n\n--- SOURCE: {url} ---\n\n{content}"
            if len(current_chunk) + len(content_with_header) > MAX_CONTENT_CHARS:
                chunks.append(current_chunk)
                current_chunk = content_with_header
            else:
                current_chunk += content_with_header
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def generate_comprehensive_notes(self, data: Dict[str, str], metadata: Dict[str, Dict], url: str) -> List[str]:
        if not data: return ["No content was scraped."]
        
        chunks = self._chunk_content(data)
        prompt = self._create_comprehensive_prompt()
        chain = prompt | self.llm | self.output_parser
        
        notes = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Generating notes for chunk {i+1}/{len(chunks)}...")
            try:
                note_section = chain.invoke({"content": chunk, "website_url": url})
                notes.append(note_section)
            except Exception as e:
                logger.error(f"Error generating notes for chunk {i+1}: {e}")
                notes.append(f"Error generating notes for section {i+1}: {e}")
        
        return notes

def save_comprehensive_notes(notes: List[str], main_topic: str, url: str) -> None:
    if not notes or not any(notes):
        print("No notes were generated.")
        return
    
    safe_topic = re.sub(r'[^a-zA-Z0-9_-]', '_', main_topic).strip('_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"study_guide_{safe_topic}_{timestamp}.md"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# 📚 Comprehensive Study Guide for: {main_topic}\n\n")
            f.write(f"**Source:** {url}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"{'='*100}\n\n")
            for section in notes:
                f.write(section)
                f.write(f"\n\n{'='*100}\n\n")
        
        print(f"\n✅ Success! Enhanced notes saved to: {filename}")
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced documentation scraper with AI research.")
    parser.add_argument("url", type=str, help="Starting URL of the documentation.")
    parser.add_argument("--limit", type=int, default=30, help="Max pages to scrape from main site (0 for no limit).")
    parser.add_argument("--research-limit", type=int, default=MAX_RESEARCH_PAGES, help="Max research pages to scrape.")
    parser.add_argument("--no-research", action="store_true", help="Disable research enhancement.")
    args = parser.parse_args()

    gemini_api_key = APIKeyManager.get_gemini_api_key()
    if not gemini_api_key:
        print(f"❌ FATAL: Gemini API key not found in '{API_KEY_FILE}'.")
        return

    google_api_key, google_cx = None, None
    if not args.no_research:
        google_api_key, google_cx = APIKeyManager.get_google_search_credentials()
        if not google_api_key or not google_cx:
            print("⚠️ Warning: Google credentials not found. Research will use DuckDuckGo only.")
    
    print(f"\n🚀 Starting Enhanced Scraper for: {args.url}")
    print(f"📄 Main Site Limit: {args.limit if args.limit > 0 else 'Unlimited'}")
    if not args.no_research:
        print(f"🔬 Research Limit: {args.research_limit} pages ({'Google & DDG' if google_api_key else 'DDG only'})")
    else:
        print("🔬 Research: DISABLED")
    print(f"{'='*80}\n")

    scraper = WebsiteScraper(args.url, max_pages=args.limit)
    scraped_data, metadata = scraper.crawl()

    if not scraped_data:
        print("\n❌ No content scraped from main site. Exiting.")
        return

    query_generator = ResearchQueryGenerator(gemini_api_key)
    content_insights = query_generator.extract_content_insights(scraped_data, metadata, args.url)
    main_topic = content_insights.get('main_technology', urlparse(args.url).netloc)
    print(f"✅ Main topic identified as: {main_topic}")

    if not args.no_research:
        print("\n🔬 Starting Research Phase...")
        research_queries = query_generator.generate_research_queries(content_insights)
        
        if research_queries:
            researcher = GoogleSearchResearcher(google_api_key, google_cx)
            research_urls = researcher.search_and_extract_urls(research_queries, urlparse(args.url).netloc)
            
            if research_urls:
                research_urls = research_urls[:args.research_limit]
                print(f"\n🔍 Crawling {len(research_urls)} research URLs...")
                research_scraper = WebsiteScraper(args.url, max_pages=len(research_urls))
                research_data, research_meta = research_scraper.crawl(additional_urls=research_urls)
                scraped_data.update(research_data)
                metadata.update(research_meta)
            else:
                print("⚠️ No useful research URLs found.")
        else:
            print("⚠️ Could not generate research queries.")

    print(f"\n🧠 Generating comprehensive study materials for '{main_topic}'...")
    generator = EnhancedNoteGenerator(gemini_api_key)
    notes_sections = generator.generate_comprehensive_notes(scraped_data, metadata, args.url)
    
    save_comprehensive_notes(notes_sections, main_topic, args.url)
    
    print("\n🎉 Process completed successfully!")

if __name__ == "__main__":
    main()
