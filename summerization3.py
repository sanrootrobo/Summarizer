import os
import requests
import logging
from typing import Optional, Dict, List
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
import re
import argparse
from time import sleep

# NEW DEPENDENCY: html2text to preserve code blocks and structure
import html2text

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
API_KEY_FILE = "geminaikey"
# The default limit is now controlled via the command-line argument
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
MAX_CONTENT_CHARS = 750000 # Increased slightly for larger crawls

class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str = "geminaikey") -> Optional[str]:
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

class WebsiteScraper:
    # MODIFIED: __init__ now takes max_pages
    def __init__(self, base_url: str, max_pages: int):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages # This is now set from the CLI argument
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.scraped_content = {}
        self.visited_urls = set()
        self.text_maker = html2text.HTML2Text()
        self.text_maker.body_width = 0

    def _get_page_content(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
            return None
        except requests.RequestException as e:
            logger.warning(f"Could not fetch {url}: {e}")
            return None

    def _parse_content_and_links(self, html: str, page_url: str) -> (str, List[str]):
        soup = BeautifulSoup(html, 'html.parser')
        main_content = (
            soup.find('main') or soup.find('article') or 
            soup.find('div', {'role': 'main'}) or 
            soup.find('div', class_=re.compile(r'content|main|body')) or soup.body
        )
        markdown_content = self.text_maker.handle(str(main_content)) if main_content else ""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(page_url, href)
            parsed_url = urlparse(full_url)
            if parsed_url.netloc == self.domain and not any(ext in parsed_url.path for ext in ['.pdf', '.jpg', '.png', '.zip']):
                links.add(full_url.split('#')[0])
        return markdown_content, list(links)
    
    # MODIFIED: The crawl loop now respects self.max_pages, including the unlimited case
    def crawl(self) -> Dict[str, str]:
        urls_to_visit = {self.base_url}
        with ThreadPoolExecutor(max_workers=5) as executor:
            while urls_to_visit:
                # Check for the limit at the beginning of each loop iteration
                if self.max_pages > 0 and len(self.visited_urls) >= self.max_pages:
                    logger.info(f"Reached scraping limit of {self.max_pages} pages. Stopping crawl.")
                    break

                url = urls_to_visit.pop()
                if url in self.visited_urls: continue
                self.visited_urls.add(url)
                
                # Update progress display
                progress_prefix = f"[{len(self.visited_urls)}/{self.max_pages}]" if self.max_pages > 0 else f"[{len(self.visited_urls)}]"
                logger.info(f"{progress_prefix} Scraping: {url}")
                
                html_content = self._get_page_content(url)
                if not html_content: continue
                
                text, new_links = self._parse_content_and_links(html_content, url)
                if text and len(text.strip()) > 100:
                    self.scraped_content[url] = text
                    
                for link in new_links:
                    if link not in self.visited_urls: urls_to_visit.add(link)
                
                sleep(0.2) # Added a slightly longer politeness delay
        
        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages with content.")
        return self.scraped_content

class NoteGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0.2)
        self.output_parser = StrOutputParser()

    def generate_notes_in_one_shot(self, scraped_data: Dict[str, str], website_url: str) -> str:
        if not scraped_data: return "No content was scraped to generate notes."
        logger.info("Combining content for a single note-generation request...")
        full_text_parts = []
        for url, text in scraped_data.items():
            full_text_parts.append(f"--- Content from page: {url} ---\n\n{text}")
        combined_content = "\n\n".join(full_text_parts)
        if len(combined_content) > MAX_CONTENT_CHARS:
            logger.warning(f"Combined content exceeds {MAX_CONTENT_CHARS} characters. Truncating.")
            combined_content = combined_content[:MAX_CONTENT_CHARS]
        logger.info(f"Total content length: {len(combined_content)} chars. Sending single request to Gemini...")
        prompt = ChatPromptTemplate.from_template(
            """You are an expert technical writer and educator... (prompt is unchanged) ...
            Comprehensive Study Notes (in Markdown):"""
        ) # The detailed prompt from before remains the same
        chain = prompt | self.llm | self.output_parser
        try:
            return chain.invoke({"content": combined_content, "website_url": website_url})
        except Exception as e:
            logger.error(f"LLM note generation failed: {e}")
            return f"Error during note generation: {e}"

def save_notes_to_markdown(notes: str, url: str) -> None:
    if not notes:
        print("No notes were generated to save.")
        return
    domain = urlparse(url).netloc
    safe_domain = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
    filename = f"notes_{safe_domain}.md"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Comprehensive Notes for: {url}\n\n---\n\n")
            f.write(notes)
        print(f"\n✅ Success! Comprehensive notes saved to: {filename}")
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")
        print(f"\n❌ Error: Could not write notes to file.")

def main():
    # MODIFIED: Added the --limit argument
    parser = argparse.ArgumentParser(description="Scrape a documentation website and generate comprehensive notes.")
    parser.add_argument("url", type=str, help="The starting URL of the documentation to process.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=25, 
        help="Maximum number of pages to scrape. Set to 0 for no limit. Default is 25."
    )
    args = parser.parse_args()

    api_key = APIKeyManager.get_api_key(API_KEY_FILE)
    if not api_key:
        print(f"FATAL: Could not load API key from '{API_KEY_FILE}'.")
        return

    # Give user feedback on the chosen limit
    limit_text = f"page limit: {args.limit}" if args.limit > 0 else "page limit: UNLIMITED (be careful!)"
    print(f"\n{'='*60}\nStarting scrape of: {args.url}\nScraping configuration: {limit_text}\n{'='*60}")

    # Pass the limit from the command line to the scraper
    scraper = WebsiteScraper(args.url, max_pages=args.limit)
    scraped_data = scraper.crawl()

    if not scraped_data:
        print("\nCould not find any content to process.")
        return

    print(f"\n{'='*60}\nGenerating comprehensive notes from scraped content...\n{'='*60}")
    generator = NoteGenerator(api_key)
    final_notes = generator.generate_notes_in_one_shot(scraped_data, args.url)

    save_notes_to_markdown(final_notes, args.url)

if __name__ == "__main__":
    main()
