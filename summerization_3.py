import os
import requests
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
import re
import argparse
from time import sleep
import json
from datetime import datetime
import html2text
import yaml
import sys

# Third-party libraries
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Global Configuration (will be updated from config file) ---
RATE_LIMIT_DELAY = 0.5
MAX_CONTENT_CHARS = 400000

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- File Loading Functions ---
def load_config(filepath: str = "config.yml") -> Dict:
    """Loads the YAML configuration file and handles errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from '{filepath}'.")
        return config
    except FileNotFoundError:
        logger.error(f"FATAL: Configuration file not found at '{filepath}'.")
        print(f"\n❌ Error: The configuration file '{filepath}' was not found. Please create it and try again.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"FATAL: Error parsing YAML in '{filepath}': {e}")
        print(f"\n❌ Error: There was a problem parsing '{filepath}'. Please check its format.")
        sys.exit(1)

def load_prompt(filepath: str = "prompt.md") -> str:
    """Loads the prompt template from an external text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        logger.info(f"Prompt loaded successfully from '{filepath}'.")
        return prompt_content
    except FileNotFoundError:
        logger.error(f"FATAL: Prompt file not found at '{filepath}'.")
        print(f"\n❌ Error: The prompt file '{filepath}' was not found. Please create it and try again.")
        sys.exit(1)

class APIKeyManager:
    """Handles loading the API key from a file."""
    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"API key file not found at '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10:
                logger.error("Invalid API key found in file.")
                return None
            logger.info("API key loaded successfully.")
            return api_key
        except Exception as e:
            logger.error(f"Error reading API key: {e}")
            return None

class ContentAnalyzer:
    """Categorizes scraped content based on URL and text analysis."""
    @staticmethod
    def categorize_content(url: str, content: str) -> Dict[str, any]:
        categories = {'type': 'general', 'complexity_level': 'intermediate'}
        url_lower = url.lower()
        if any(p in url_lower for p in ['api', 'reference']): categories['type'] = 'api_reference'
        elif any(p in url_lower for p in ['tutorial', 'guide']): categories['type'] = 'tutorial'
        elif any(p in url_lower for p in ['concept', 'overview']): categories['type'] = 'concept'
        return categories

class WebsiteScraper:
    """Crawls a website, extracts content, and follows internal links."""
    def __init__(self, base_url: str, max_pages: int, user_agent: str, request_timeout: int):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.request_timeout = request_timeout
        self.scraped_content = {}
        self.content_metadata = {}
        self.visited_urls = set()
        self.failed_urls = set()
        
        self.text_maker = html2text.HTML2Text()
        self.text_maker.body_width = 0
        self.text_maker.ignore_links = True
        self.text_maker.ignore_images = True

    def _get_page_content(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            self.failed_urls.add(url)
        return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for tag in [soup.find('title'), soup.find('h1'), soup.find('meta', {'property': 'og:title'})]:
            if tag:
                title = tag.get_text(strip=True) or tag.get('content', '').strip()
                if title: return re.sub(r'\s+', ' ', title)
        return "Untitled Page"

    def _parse_content_and_links(self, html: str, page_url: str) -> Tuple[str, List[str], str]:
        soup = BeautifulSoup(html, 'html.parser')
        title = self._extract_title(soup)
        
        for element in soup.select('script, style, nav, footer, header, aside, form, .ad'):
            element.decompose()
        
        main_content_area = soup.select_one('main, article, [role="main"], .main-content, .content') or soup.body
        markdown_content = self.text_maker.handle(str(main_content_area)) if main_content_area else ""
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content).strip()
        
        links = set()
        for a_tag in soup.find_all('a', href=True):
            try:
                full_url = urljoin(page_url, a_tag['href'])
                parsed_url = urlparse(full_url)
                if (parsed_url.netloc == self.domain and not any(ext in parsed_url.path for ext in ['.pdf', '.zip'])):
                    links.add(full_url.split('#')[0])
            except Exception:
                continue
        return markdown_content, list(links), title

    def crawl(self) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        urls_to_visit = {self.base_url}
        while urls_to_visit:
            if 0 < self.max_pages <= len(self.visited_urls):
                logger.info(f"Reached scraping limit of {self.max_pages} pages.")
                break

            url = urls_to_visit.pop()
            if url in self.visited_urls or url in self.failed_urls:
                continue
            
            self.visited_urls.add(url)
            logger.info(f"[{len(self.visited_urls)}/{self.max_pages or '∞'}] Scraping: {url}")
            
            html = self._get_page_content(url)
            if not html: continue
            
            text, new_links, title = self._parse_content_and_links(html, url)
            if text and len(text.strip()) > 150:
                self.scraped_content[url] = text
                self.content_metadata[url] = {
                    'title': title, 'analysis': ContentAnalyzer.categorize_content(url, text),
                    'scraped_at': datetime.now().isoformat(),
                    'estimated_read_time': max(1, len(text.split()) // 200)
                }
            
            for link in new_links:
                if link not in self.visited_urls and link not in self.failed_urls:
                    urls_to_visit.add(link)
            sleep(RATE_LIMIT_DELAY)
        
        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} valid pages.")
        return self.scraped_content, self.content_metadata

class EnhancedNoteGenerator:
    """Generates a study guide from scraped content using an external prompt template."""
    def __init__(self, api_key: str, llm_config: Dict, prompt_template_string: str):
        model_name = llm_config.get("model_name", "gemini-1.5-pro")
        llm_params = llm_config.get("parameters", {})
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **llm_params)
        self.output_parser = StrOutputParser()
        # The prompt is now created from the externally loaded string
        self.prompt = ChatPromptTemplate.from_template(prompt_template_string)

    def _chunk_content(self, scraped_data: Dict[str, str], metadata: Dict[str, Dict]) -> List[Dict]:
        """Chunks content to fit within the model's context window."""
        chunks, current_chunk = [], {'content': "", 'urls': []}
        for url, content in scraped_data.items():
            header = f"\n\n--- Source: {metadata.get(url, {}).get('title', 'Untitled')} ({url}) ---\n\n"
            if len(current_chunk['content']) + len(header) + len(content) > MAX_CONTENT_CHARS:
                if current_chunk['content']: chunks.append(current_chunk)
                current_chunk = {'content': header + content, 'urls': [url]}
            else:
                current_chunk['content'] += header + content
                current_chunk['urls'].append(url)
        if current_chunk['content']: chunks.append(current_chunk)
        return chunks

    def generate_comprehensive_notes(self, scraped_data: Dict[str, str], metadata: Dict[str, Dict], website_url: str) -> List[str]:
        if not scraped_data: return ["No content was scraped to generate notes."]
        
        chunks = self._chunk_content(scraped_data, metadata)
        chain = self.prompt | self.llm | self.output_parser
        
        sections = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Generating notes for section {i+1}/{len(chunks)} ({len(chunk['urls'])} pages)...")
            try:
                notes = chain.invoke({"content": chunk['content'], "website_url": website_url})
                if len(chunks) > 1:
                    notes = f"\n\n{'='*80}\n# Study Guide Section {i+1} of {len(chunks)}\n{'='*80}\n\n" + notes
                sections.append(notes)
            except Exception as e:
                logger.error(f"Error generating notes for section {i+1}: {e}")
                sections.append(f"\n\n# ❌ Error in Section {i+1}: {e}\n")
        return sections

def save_comprehensive_notes(notes: List[str], url: str, metadata: Dict[str, Dict]) -> None:
    if not notes or not any(s.strip() for s in notes):
        print("❌ No notes were generated to save.")
        return
    
    domain = urlparse(url).netloc
    safe_domain = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
    filename = f"study_guide_{safe_domain}_{datetime.now():%Y%m%d_%H%M%S}.md"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            total_read_time = sum(m.get('estimated_read_time', 0) for m in metadata.values())
            f.write(f"# 🎓 Comprehensive Study Guide for {domain}\n")
            f.write(f"* **Source URL:** {url}\n")
            f.write(f"* **Pages Processed:** {len(metadata)}\n")
            f.write(f"* **Estimated Study Time:** ~{total_read_time} minutes\n")
            f.write(f"* **Generated on:** {datetime.now():%Y-%m-%d %H:%M:%S}*\n\n")
            f.write(f"{'='*80}\n")
            f.write("\n".join(notes))
        
        print(f"\n🎉 **SUCCESS! Your study guide is ready!**")
        print(f"📖 **File saved:** {filename}")
        print(f"🚀 **Next Step:** Open {filename} in a markdown viewer to start learning!")
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")

def main():
    global RATE_LIMIT_DELAY, MAX_CONTENT_CHARS

    # Load all external configurations first
    config = load_config()
    prompt_template = load_prompt("prompt.md") # Load the external prompt

    api_cfg = config.get('api', {})
    llm_cfg = config.get('llm', {})
    scraper_cfg = config.get('scraper', {})

    api_key_file = api_cfg.get('key_file', 'geminaikey')
    model_name = llm_cfg.get('model_name', 'gemini-1.5-pro')
    user_agent = scraper_cfg.get('user_agent', 'Mozilla/5.0 ...')
    default_delay = scraper_cfg.get('rate_limit_delay', 0.5)
    timeout = scraper_cfg.get('request_timeout', 15)
    MAX_CONTENT_CHARS = scraper_cfg.get('max_content_chars', 400000)

    parser = argparse.ArgumentParser(description="🚀 Enhanced Documentation Scraper")
    parser.add_argument("url", type=str, help="🌐 Starting URL of the documentation")
    parser.add_argument("--limit", type=int, default=25, help="📄 Max pages to scrape (0=unlimited)")
    parser.add_argument("--delay", type=float, default=default_delay, help=f"⏱️ Request delay (default: {default_delay}s)")
    args = parser.parse_args()
    
    RATE_LIMIT_DELAY = args.delay

    api_key = APIKeyManager.get_api_key(api_key_file)
    if not api_key:
        print(f"\n❌ SETUP REQUIRED: API key not found in '{api_key_file}'.")
        print("Please create the file and add your key from https://makersuite.google.com/app/apikey")
        return

    print(f"\n{'='*60}\n🚀 Starting Documentation Scraper\n{'='*60}")
    print(f"🎯 Target: {args.url} | 🤖 Model: {model_name}")
    print(f"📄 Page Limit: {args.limit or 'Unlimited'} | ⏱️ Delay: {args.delay}s")
    print(f"{'='*60}\n")

    print("🕷️ Phase 1: Scraping website...")
    scraper = WebsiteScraper(args.url, args.limit, user_agent, timeout)
    scraped_data, metadata = scraper.crawl()

    if not scraped_data:
        print("\n❌ No content was found. The site may block scrapers or be JavaScript-heavy.")
        return

    print(f"\n🧠 Phase 2: Generating study guide with AI...")
    # Pass the loaded prompt template to the generator
    generator = EnhancedNoteGenerator(api_key, llm_cfg, prompt_template)
    notes = generator.generate_comprehensive_notes(scraped_data, metadata, args.url)

    save_comprehensive_notes(notes, args.url, metadata)
    
    print(f"\n🎊 **MISSION ACCOMPLISHED!** Happy learning! 🎉")

if __name__ == "__main__":
    main()
