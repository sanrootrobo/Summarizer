import os
import requests
import logging
from typing import Optional, Dict, Any, List, Set
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MODEL_NAME = "gemini-2.5-pro" # Using 1.5 Pro as it's highly capable and widely available
API_KEY_FILE = "geminaikey"
MAX_PAGES_TO_SCRAPE = 20  # Safety limit to avoid crawling huge sites
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- Reusing Authentication from your provided code ---

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

# --- New Components for Scraping and Summarization ---

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
        """Fetches HTML content of a single page."""
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
        """Extracts clean text and internal links from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove non-content tags
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            element.decompose()
        
        # Get clean text from the main body
        body = soup.find('body')
        text_content = body.get_text(separator=' ', strip=True) if body else ""
        
        # Find all internal links
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(page_url, href)
            parsed_url = urlparse(full_url)
            
            # Check if it's an internal link on the same domain and not a file
            if parsed_url.netloc == self.domain and not parsed_url.path.split('/')[-1].__contains__('.'):
                links.add(full_url.split('#')[0]) # Remove fragment identifiers
                
        return text_content, list(links)
    
    def crawl(self) -> Dict[str, str]:
        """Crawls the website starting from the base URL."""
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
                
                # Store content if it's substantial
                if text and len(text.split()) > 50:
                    self.scraped_content[url] = text[:15000] # Limit content per page
                
                # Add new, unvisited links to the queue
                for link in new_links:
                    if link not in self.visited_urls:
                        urls_to_visit.add(link)
                
                sleep(0.1) # Be polite

        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages with content.")
        return self.scraped_content


class ContentSummarizer:
    """Uses Gemini to summarize text content."""

    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key)
        self.output_parser = StrOutputParser()

    def _get_summary(self, prompt_template: ChatPromptTemplate, content: str, **kwargs) -> str:
        """Helper to run a summarization chain."""
        chain = prompt_template | self.llm | self.output_parser
        try:
            return chain.invoke({"content": content, **kwargs})
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return f"Error during summarization: {e}"

    def summarize_website_content(self, scraped_data: Dict[str, str]) -> str:
        """Generates a two-stage summary of the entire website."""
        if not scraped_data:
            return "No content was scraped from the website."

        # Stage 1: Summarize each page individually
        logger.info("Starting Stage 1: Summarizing individual pages...")
        page_summaries = {}
        page_summary_prompt = ChatPromptTemplate.from_template(
            """As an expert analyst, create a concise, bullet-point summary of the key information from the following webpage content. Focus on its main purpose and topics.
            
            Webpage Content:
            "{content}"
            
            Concise Summary:"""
        )

        for url, text in scraped_data.items():
            logger.info(f"Summarizing content from {url}")
            page_summaries[url] = self._get_summary(page_summary_prompt, text)
            sleep(1) # Rate limiting

        # Stage 2: Create a final executive summary from the page summaries
        logger.info("Starting Stage 2: Creating final executive summary...")
        all_summaries = "\n\n---\n\n".join(
            f"From page {url}:\n{summary}" for url, summary in page_summaries.items()
        )
        
        final_summary_prompt = ChatPromptTemplate.from_template(
            """You are a strategic business analyst. Below are summaries from various pages of a single website. 
            Synthesize them into a single, high-level executive summary.

            Your final output should be a set of short and precise notes covering:
            1.  **Overall Purpose:** What is the main goal of this website? (e.g., e-commerce, blog, corporate site)
            2.  **Key Offerings/Topics:** What are the primary products, services, or themes discussed?
            3.  **Target Audience:** Who is this website for?
            4.  **Key Value Proposition:** What makes this website unique or valuable to its audience?

            Individual Page Summaries:
            {content}

            ---
            Final Executive Summary:"""
        )

        final_summary = self._get_summary(final_summary_prompt, all_summaries)
        return final_summary


def main():
    """Main function to run the website summarizer."""
    parser = argparse.ArgumentParser(description="Scrape a website and generate a summary using Gemini.")
    parser.add_argument("url", type=str, help="The starting URL of the website to scrape.")
    args = parser.parse_args()

    # 1. Get API Key
    api_key = APIKeyManager.get_api_key(API_KEY_FILE)
    if not api_key:
        print(f"FATAL: Could not load API key from '{API_KEY_FILE}'. Please ensure the file exists and is valid.")
        return

    # 2. Scrape the website
    print(f"\n{'='*50}\nStarting scrape of: {args.url}\n{'='*50}")
    scraper = WebsiteScraper(args.url)
    try:
        scraped_data = scraper.crawl()
    except Exception as e:
        logger.error(f"An unexpected error occurred during scraping: {e}")
        print(f"Error: Could not complete the scraping process. See logs for details.")
        return

    if not scraped_data:
        print("\nCould not find any content to summarize. The website might be using heavy JavaScript or access was blocked.")
        return

    # 3. Summarize the content
    print(f"\n{'='*50}\nContent scraped. Now generating summary...\n{'='*50}")
    summarizer = ContentSummarizer(api_key)
    final_summary = summarizer.summarize_website_content(scraped_data)

    # 4. Display the result
    print(f"\n{'='*20} FINAL WEBSITE SUMMARY {'='*20}\n")
    print(final_summary)
    print(f"\n{'='*61}\n")


if __name__ == "__main__":
    main()
