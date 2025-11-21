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
# Using Gemini 1.5 Pro is ideal here due to its large context window,
# which is perfect for processing the entire website's text at once.
MODEL_NAME = "gemini-2.5-pro"
API_KEY_FILE = "geminaikey"
MAX_PAGES_TO_SCRAPE = 30
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
MAX_CONTENT_CHARS = 300000 # Safety limit for total characters to send to the LLM

# --- Reusing Authentication ---

class APIKeyManager:
    """Enhanced API key management."""
    
    @staticmethod
    def get_api_key(filepath: str = "geminaikey") -> Optional[str]:
        """Retrieve and validate API key from file."""
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

# --- Scraping Component (Unchanged) ---

class WebsiteScraper:
    """Crawls and scrapes text content from a website."""

    def __init__(self, base_url: str, max_pages: int = MAX_PAGES_TO_SCRAPE):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.scraped_content = {}
        self.visited_urls = set()

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
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            element.decompose()
        body = soup.find('body')
        text_content = body.get_text(separator=' ', strip=True) if body else ""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(page_url, href)
            parsed_url = urlparse(full_url)
            if parsed_url.netloc == self.domain and not parsed_url.path.split('/')[-1].__contains__('.'):
                links.add(full_url.split('#')[0])
        return text_content, list(links)
    
    def crawl(self) -> Dict[str, str]:
        urls_to_visit = {self.base_url}
        with ThreadPoolExecutor(max_workers=5) as executor:
            while urls_to_visit and len(self.visited_urls) < self.max_pages:
                url = urls_to_visit.pop()
                if url in self.visited_urls:
                    continue
                self.visited_urls.add(url)
                logger.info(f"[{len(self.visited_urls)}/{self.max_pages}] Scraping: {url}")
                html_content = self._get_page_content(url)
                if not html_content:
                    continue
                text, new_links = self._parse_content_and_links(html_content, url)
                if text and len(text.split()) > 50:
                    self.scraped_content[url] = text
                for link in new_links:
                    if link not in self.visited_urls:
                        urls_to_visit.add(link)
                sleep(0.1)
        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages.")
        return self.scraped_content


# --- MODIFIED Summarization Component ---

class ContentSummarizer:
    """Uses a single Gemini API call to summarize the entire website's content."""

    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key)
        self.output_parser = StrOutputParser()

    def summarize_website_in_one_shot(self, scraped_data: Dict[str, str]) -> str:
        """
        Combines all scraped text and generates a single summary.
        This method makes only ONE call to the LLM.
        """
        if not scraped_data:
            return "No content was scraped from the website to summarize."

        logger.info("Combining content from all pages for a single summary request...")

        # Combine all text into one large string, with separators
        full_text_parts = []
        for url, text in scraped_data.items():
            full_text_parts.append(f"--- Content from page: {url} ---\n\n{text}")
        
        combined_content = "\n\n".join(full_text_parts)

        # Safety check for content length
        if len(combined_content) > MAX_CONTENT_CHARS:
            logger.warning(
                f"Combined content exceeds {MAX_CONTENT_CHARS} characters. "
                f"Truncating to fit within safety limits."
            )
            combined_content = combined_content[:MAX_CONTENT_CHARS]
        
        logger.info(f"Total content length for summarization: {len(combined_content)} characters.")
        logger.info("Sending single request to Gemini for final summary...")

        # Define the single-shot prompt
        prompt = ChatPromptTemplate.from_template(
            """You are a highly skilled analyst. Your task is to synthesize the combined text from an entire website into a concise, well-structured executive summary in Markdown format.

The following text is a large string containing the raw content from multiple pages of a single website. Each page's content is separated by a "--- Content from page: <URL> ---" divider.

Analyze all of it and produce a single, cohesive summary that covers the following sections:
1.  **Overall Purpose:** What is the main goal of this website? (e.g., e-commerce, technical documentation, corporate blog)
2.  **Key Offerings/Topics:** What are the primary products, services, or themes discussed across the site?
3.  **Target Audience:** Who is this website designed for?
4.  **Key Value Proposition:** What makes this website unique or valuable to its audience?

---
BEGIN WEBSITE CONTENT
---

{content}

---
END WEBSITE CONTENT
---

Final Executive Summary (in Markdown):"""
        )

        chain = prompt | self.llm | self.output_parser
        try:
            return chain.invoke({"content": combined_content})
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return f"Error during summarization: {e}"


def save_summary_to_markdown(summary: str, url: str) -> None:
    """Saves the summary content to a Markdown file."""
    if not summary:
        print("No summary was generated to save.")
        return
    domain = urlparse(url).netloc
    safe_domain = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
    filename = f"website_summary_{safe_domain}.md"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Website Summary for: {url}\n\n")
            f.write("---\n\n")
            f.write(summary)
        print(f"\n✅ Success! Summary saved to: {filename}")
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")
        print(f"\n❌ Error: Could not write summary to file.")


def main():
    parser = argparse.ArgumentParser(description="Scrape a website, generate a single summary, and save it to a Markdown file.")
    parser.add_argument("url", type=str, help="The starting URL of the website to scrape.")
    args = parser.parse_args()

    api_key = APIKeyManager.get_api_key(API_KEY_FILE)
    if not api_key:
        print(f"FATAL: Could not load API key from '{API_KEY_FILE}'.")
        return

    print(f"\n{'='*50}\nStarting scrape of: {args.url}\n{'='*50}")
    scraper = WebsiteScraper(args.url)
    try:
        scraped_data = scraper.crawl()
    except Exception as e:
        logger.error(f"An unexpected error occurred during scraping: {e}")
        return

    if not scraped_data:
        print("\nCould not find any content to summarize.")
        return

    print(f"\n{'='*50}\nGenerating single-shot summary for the entire site...\n{'='*50}")
    summarizer = ContentSummarizer(api_key)
    final_summary = summarizer.summarize_website_in_one_shot(scraped_data)

    save_summary_to_markdown(final_summary, args.url)


if __name__ == "__main__":
    main()
