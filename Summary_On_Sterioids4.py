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

# Playwright is an optional import, only needed if the user specifies it
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

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
REQUEST_TIMEOUT = 20
MAX_CONTENT_CHARS = 500000
MAX_SEARCH_RESULTS = 8
MAX_RESEARCH_PAGES = 20

# --- ANTI-BOT CONFIGURATION ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
]
PROXIES = None # Example: {"http": "http://user:pass@host:port", "https": "http://user:pass@host:port"}


class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str = API_KEY_FILE) -> Optional[str]:
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
            if not api_key_path.exists(): return None, None
            with open(api_key_path, 'r', encoding='utf-8') as f: api_key = f.read().strip()
            cx_path = Path(GOOGLE_CX_FILE)
            if not cx_path.exists(): return None, None
            with open(cx_path, 'r', encoding='utf-8') as f: cx_id = f.read().strip()
            if not api_key or len(api_key) < 10 or not cx_id: return None, None
            logger.info("Google Search API credentials loaded successfully.")
            return api_key, cx_id
        except Exception as e:
            logger.warning(f"Error reading Google Search credentials: {e}.")
            return None, None

class ContentAnalyzer:
    @staticmethod
    def categorize_content(url: str, content: str) -> Dict[str, any]:
        categories = {'type': 'general', 'priority': 'medium', 'is_research': False}
        url_lower = url.lower()
        if any(indicator in url_lower for indicator in ['stackoverflow.com', 'github.com', 'medium.com', 'dev.to']):
            categories['is_research'] = True
            categories['priority'] = 'high'
        if any(p in url_lower for p in ['api', 'reference']): categories.update({'type': 'api_reference', 'priority': 'high'})
        elif any(p in url_lower for p in ['tutorial', 'guide']): categories.update({'type': 'tutorial', 'priority': 'high'})
        elif any(p in url_lower for p in ['concept', 'overview']): categories.update({'type': 'concept', 'priority': 'high'})
        return categories

class ResearchQueryGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=SEARCH_MODEL_NAME, google_api_key=api_key, temperature=0.7)
        self.output_parser = StrOutputParser()
    
    def generate_research_queries(self, content_insights: Dict[str, str]) -> List[str]:
        prompt = ChatPromptTemplate.from_template("""
You are a research analyst generating search queries to find complementary resources for the given documentation.
Based on this analysis, generate 8-10 specific, context-aware search queries.

**CONTENT ANALYSIS:**
- Website: {main_url}
- Main Technology: {main_technology}

**YOUR MISSION:**
Generate a JSON array of search queries to find tutorials, real-world examples, and solutions to common problems for '{main_technology}'.

**OUTPUT FORMAT:**
Return ONLY a JSON array of search queries. Example: ["React hooks tutorial", "React context API best practices"]
""")
        chain = prompt | self.llm | self.output_parser
        try:
            logger.info("🧠 Generating context-aware research queries...")
            result = chain.invoke(content_insights)
            queries = json.loads(result.strip())
            if isinstance(queries, list) and queries:
                logger.info(f"✅ Generated {len(queries)} research queries.")
                return queries[:10]
            return self._generate_fallback_queries(content_insights)
        except Exception:
            return self._generate_fallback_queries(content_insights)
    
    def extract_main_topic(self, scraped_content: Dict[str, str], url: str) -> str:
        # A simplified heuristic to find the main topic
        domain = urlparse(url).netloc.replace('www.', '').lower().split('.')[0]
        page_titles = [meta.get('title', '').lower() for meta in scraped_content.values()]
        common_words = {}
        for title in page_titles:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', title)
            for word in words:
                if word not in ['docs', 'documentation', 'guide', 'api', 'home']:
                    common_words[word] = common_words.get(word, 0) + 1
        
        if common_words:
            main_topic = max(common_words, key=common_words.get)
            return f"{domain} {main_topic}".title()

        return domain.title()

    def _generate_fallback_queries(self, insights: Dict[str, str]) -> List[str]:
        tech = insights.get('main_technology', 'tech')
        return [f"{tech} tutorial", f"{tech} common issues", f"{tech} best practices"]

class GoogleSearchResearcher:
    """Performs searches using Google's official API or scrapes DuckDuckGo."""
    def __init__(self, api_key: str = None, cx_id: str = None):
        self.api_key = api_key
        self.cx_id = cx_id
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    
    def search_and_extract_urls(self, queries: List[str], exclude_domain: str = None) -> List[str]:
        all_urls = set()
        google_available = bool(self.api_key and self.cx_id)
        for i, query in enumerate(queries):
            urls_found = False
            if google_available:
                try:
                    google_urls = self._search_google(query, exclude_domain)
                    if google_urls:
                        all_urls.update(google_urls)
                        urls_found = True
                        logger.info(f"✅ Google API found {len(google_urls)} URLs for query {i+1}.")
                except Exception as e:
                    logger.warning(f"❌ Google API search failed: {e}")
            if not urls_found:
                try:
                    logger.info(f"🦆 Using DuckDuckGo scraping fallback for query {i+1}...")
                    ddg_urls = self._search_duckduckgo(query, exclude_domain)
                    if ddg_urls: all_urls.update(ddg_urls); logger.info(f"✅ DuckDuckGo found {len(ddg_urls)} URLs.")
                except Exception as e:
                    logger.error(f"❌ DuckDuckGo scraping also failed: {e}")
            sleep(random.uniform(1.0, 2.5))
        return list(all_urls)[:MAX_RESEARCH_PAGES]

    def _search_google(self, query: str, exclude_domain: str) -> List[str]:
        params = {'key': self.api_key, 'cx': self.cx_id, 'q': query, 'num': MAX_SEARCH_RESULTS}
        response = self.session.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=REQUEST_TIMEOUT, proxies=PROXIES)
        response.raise_for_status()
        results = response.json()
        return [item['link'] for item in results.get('items', []) if self._is_useful_url(item.get('link'), exclude_domain)]

    def _search_duckduckgo(self, query: str, exclude_domain: str) -> List[str]:
        sleep(random.uniform(2.0, 4.0))
        response = self.session.get("https://html.duckduckgo.com/html/", params={'q': query}, timeout=REQUEST_TIMEOUT, proxies=PROXIES)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = {unquote(a['href'].split('uddg=')[1]) for a in soup.select('a.result__a') if 'uddg=' in a['href']}
        return [url for url in urls if self._is_useful_url(url, exclude_domain)]
    
    def _is_useful_url(self, url: str, domain: str) -> bool:
        return url and domain not in url and not any(url.endswith(e) for e in ['.pdf', '.zip'])

class PlaywrightResearcher:
    """Performs searches using a real browser via Playwright."""
    def __init__(self):
        self.search_engine_url = "https://duckduckgo.com/"
        logger.info("🤖 Initialized Playwright-based researcher.")

    def search_and_extract_urls(self, queries: List[str], exclude_domain: str = None) -> List[str]:
        all_urls = set()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=random.choice(USER_AGENTS))
            page = context.new_page()
            try:
                for i, query in enumerate(queries):
                    logger.info(f"🤖 Playwright searching [{i+1}/{len(queries)}]: {query}")
                    try:
                        page.goto(self.search_engine_url, timeout=20000)
                        page.locator('input[name="q"]').fill(query)
                        page.locator('input[name="q"]').press("Enter")
                        page.wait_for_selector("div#links", timeout=15000)
                        found = {link.get_attribute('href') for link in page.locator('h2 > a').all()}
                        useful_urls = {url for url in found if self._is_useful_url(url, exclude_domain)}
                        all_urls.update(useful_urls)
                        logger.info(f"✅ Playwright found {len(useful_urls)} useful URLs.")
                        sleep(random.uniform(2.0, 4.0))
                    except PlaywrightTimeoutError: logger.warning(f"❌ Playwright timed out on query: {query}")
                    except Exception as e: logger.error(f"❌ Playwright search error: {e}")
            finally:
                browser.close()
        return list(all_urls)[:MAX_RESEARCH_PAGES]

    def _is_useful_url(self, url: str, domain: str) -> bool:
         return url and domain not in url and not any(url.endswith(e) for e in ['.pdf', '.zip'])

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

    def _parse_content_and_links(self, html: str, page_url: str) -> Tuple[str, List[str], str]:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title')
        page_title = title.get_text().strip() if title else "Untitled Page"
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']): element.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.body
        markdown_content = self.text_maker.handle(str(main_content)) if main_content else ""
        links = []
        if urlparse(page_url).netloc == self.domain:
            links = {urljoin(page_url, a['href']).split('#')[0] for a in soup.find_all('a', href=True) if urlparse(urljoin(page_url, a['href'])).netloc == self.domain}
        return markdown_content, list(links), page_title

    def crawl(self, additional_urls: List[str] = None) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        urls_to_visit = {self.base_url}
        if additional_urls: urls_to_visit.update(additional_urls)
        
        while urls_to_visit and (self.max_pages == 0 or len(self.visited_urls) < self.max_pages):
            url = urls_to_visit.pop()
            if url in self.visited_urls: continue
            self.visited_urls.add(url)
            
            is_research = urlparse(url).netloc != self.domain
            logger.info(f"[{len(self.visited_urls)}] {'🔬' if is_research else '📄'} Scraping: {url}")
            
            html = self._get_page_content(url)
            if not html: continue
            
            text, new_links, title = self._parse_content_and_links(html, url)
            if text and len(text.strip()) > 100:
                self.scraped_content[url] = text
                self.content_metadata[url] = {'title': title, 'analysis': ContentAnalyzer.categorize_content(url, text), 'is_research': is_research}
            
            if not is_research: urls_to_visit.update(new_links)
            sleep(random.uniform(0.3, 0.8))
            
        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages.")
        return self.scraped_content, self.content_metadata

class EnhancedNoteGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0.3)
        self.output_parser = StrOutputParser()

    def generate_comprehensive_notes(self, data: Dict[str, str], url: str) -> List[str]:
        if not data: return ["No content was scraped."]
        prompt = ChatPromptTemplate.from_template("""
You are an expert technical writer. Transform the following raw documentation and research into a comprehensive study guide.

**Content:**
{content}

**Instructions:**
Create a detailed, well-structured guide with the following sections:
1.  **Executive Summary:** What this is, for who, and key benefits.
2.  **Core Concepts:** Explain the main ideas simply (ELI5).
3.  **Practical Examples:** Show code for common use cases, using examples from the research content provided.
4.  **Quick Reference / Cheat Sheet:** A condensed list of commands and tips.
""")
        chain = prompt | self.llm | self.output_parser
        
        full_content = "\n\n".join([f"--- SOURCE: {u} ---\n\n{c}" for u, c in data.items()])
        chunks = [full_content[i:i + MAX_CONTENT_CHARS] for i in range(0, len(full_content), MAX_CONTENT_CHARS)]
        
        notes = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Generating notes for chunk {i+1}/{len(chunks)}...")
            try:
                notes.append(chain.invoke({"content": chunk, "website_url": url}))
            except Exception as e:
                logger.error(f"Error generating notes for chunk {i+1}: {e}")
        return notes

def save_comprehensive_notes(notes: List[str], main_topic: str, url: str) -> None:
    if not notes or not any(notes):
        print("No notes generated.")
        return
    safe_topic = re.sub(r'[^a-zA-Z0-9_-]', '_', main_topic).strip('_')
    filename = f"study_guide_{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# 📚 Study Guide: {main_topic}\n\n**Source:** {url}\n\n")
            f.write("\n\n---\n\n".join(notes))
        print(f"\n✅ Success! Notes saved to: {filename}")
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced documentation scraper with AI research.")
    parser.add_argument("url", type=str, help="Starting URL of the documentation.")
    parser.add_argument("--limit", type=int, default=30, help="Max pages to scrape from main site (0 for no limit).")
    parser.add_argument("--research-limit", type=int, default=MAX_RESEARCH_PAGES, help="Max research pages to scrape.")
    parser.add_argument("--no-research", action="store_true", help="Disable research enhancement.")
    parser.add_argument("--use-playwright", action="store_true", help="Use Playwright for research (robust but slower).")
    args = parser.parse_args()

    if args.use_playwright and not PLAYWRIGHT_AVAILABLE:
        logger.error("❌ Playwright not found. Install with 'pip install playwright' and 'playwright install'.")
        return

    gemini_api_key = APIKeyManager.get_gemini_api_key()
    if not gemini_api_key: return

    google_api_key, google_cx = APIKeyManager.get_google_search_credentials()
    
    research_method = "Google API / DDG Scrape"
    if args.use_playwright: research_method = "Playwright"
    elif not google_api_key: research_method = "DDG Scrape Only"

    print(f"\n🚀 Starting Scraper for: {args.url}")
    print(f"📄 Main Site Limit: {args.limit if args.limit > 0 else 'Unlimited'}")
    if not args.no_research: print(f"🔬 Research Method: {research_method} ({args.research_limit} page limit)")
    else: print("🔬 Research: DISABLED")
    print(f"{'='*80}\n")

    scraper = WebsiteScraper(args.url, max_pages=args.limit)
    scraped_data, metadata = scraper.crawl()
    if not scraped_data:
        print("\n❌ No content scraped. Exiting."); return

    query_generator = ResearchQueryGenerator(gemini_api_key)
    main_topic = query_generator.extract_main_topic(metadata, args.url)
    print(f"✅ Main topic identified: {main_topic}")

    if not args.no_research:
        print("\n🔬 Starting Research Phase...")
        content_insights = {'main_url': args.url, 'main_technology': main_topic}
        research_queries = query_generator.generate_research_queries(content_insights)
        
        if research_queries:
            researcher = PlaywrightResearcher() if args.use_playwright else GoogleSearchResearcher(google_api_key, google_cx)
            research_urls = researcher.search_and_extract_urls(research_queries, urlparse(args.url).netloc)
            
            if research_urls:
                print(f"\n🔍 Crawling {len(research_urls)} research URLs...")
                research_scraper = WebsiteScraper(args.url, max_pages=len(research_urls))
                research_data, research_meta = research_scraper.crawl(additional_urls=research_urls)
                scraped_data.update(research_data)
        else:
            print("⚠️ Could not generate research queries.")

    print(f"\n🧠 Generating study materials for '{main_topic}'...")
    generator = EnhancedNoteGenerator(gemini_api_key)
    notes = generator.generate_comprehensive_notes(scraped_data, args.url)
    
    save_comprehensive_notes(notes, main_topic, args.url)
    print("\n🎉 Process completed successfully!")

if __name__ == "__main__":
    main()
