import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import queue
import logging
import re
from urllib.parse import urlparse
import json
from datetime import datetime

# --- Import all your existing classes ---
# (I've included them here for a single-file solution)
import requests
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import html2text
import yaml
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Existing Classes (with minor adjustments for GUI integration) ---

def load_config(filepath: str = "config.yml") -> dict:
    # ... (same as before)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {} # Return empty dict if not found
    except yaml.YAMLError:
        return {}


def load_prompt(filepath: str = "prompt.md") -> str:
    # ... (same as before)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Prompt file not found. Please create prompt.md"


class APIKeyManager:
    # ... (same as before)
    @staticmethod
    def get_api_key(filepath: str) -> str | None:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logging.error(f"API key file not found at '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10:
                logging.error("Invalid API key found in file.")
                return None
            logging.info("API key loaded successfully.")
            return api_key
        except Exception as e:
            logging.error(f"Error reading API key: {e}")
            return None


class ContentAnalyzer:
    # ... (same as before)
    @staticmethod
    def categorize_content(url: str, content: str) -> dict:
        categories = {
            'type': 'general', 'complexity_level': 'intermediate'
        }
        url_lower = url.lower()
        if any(p in url_lower for p in ['api', 'reference']):
            categories['type'] = 'api_reference'
        elif any(p in url_lower for p in ['tutorial', 'guide']):
            categories['type'] = 'tutorial'
        elif any(p in url_lower for p in ['concept', 'overview']):
            categories['type'] = 'concept'
        return categories


class WebsiteScraper:
    # ... (same as before)
    def __init__(self, base_url: str, max_pages: int, user_agent: str, request_timeout: int, rate_limit_delay: float):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.request_timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay
        self.scraped_content = {}
        self.content_metadata = {}
        self.visited_urls = set()
        self.failed_urls = set()

        self.text_maker = html2text.HTML2Text()
        self.text_maker.body_width = 0
        self.text_maker.ignore_links = True
        self.text_maker.ignore_images = True

    def _get_page_content(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed for {url}: {e}")
            self.failed_urls.add(url)
        return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for tag in [soup.find('title'), soup.find('h1'), soup.find('meta', {'property': 'og:title'})]:
            if tag:
                title = tag.get_text(strip=True) or tag.get('content', '').strip()
                if title: return re.sub(r'\s+', ' ', title)
        return "Untitled Page"

    def _parse_content_and_links(self, html: str, page_url: str) -> tuple[str, list[str], str]:
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

    def crawl(self) -> tuple[dict[str, str], dict[str, dict]]:
        urls_to_visit = {self.base_url}
        with ThreadPoolExecutor(max_workers=5) as executor:
            while urls_to_visit:
                if 0 < self.max_pages <= len(self.visited_urls):
                    logging.info(f"Reached scraping limit of {self.max_pages} pages.")
                    break

                url = urls_to_visit.pop()
                if url in self.visited_urls or url in self.failed_urls:
                    continue

                self.visited_urls.add(url)
                logging.info(f"[{len(self.visited_urls)}/{self.max_pages or '∞'}] Scraping: {url}")

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
                sleep(self.rate_limit_delay)

        logging.info(f"Crawling complete. Scraped {len(self.scraped_content)} valid pages.")
        return self.scraped_content, self.content_metadata


class EnhancedNoteGenerator:
    # ... (same as before)
    def __init__(self, api_key: str, llm_config: dict, prompt_template_string: str):
        model_name = llm_config.get("model_name", "gemini-1.5-pro")
        llm_params = llm_config.get("parameters", {})
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **llm_params)
        self.output_parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_template(prompt_template_string)

    def _chunk_content(self, scraped_data: dict[str, str], metadata: dict[str, dict], max_chars: int) -> list[dict]:
        chunks, current_chunk = [], {'content': "", 'urls': []}
        for url, content in scraped_data.items():
            header = f"\n\n--- Source: {metadata.get(url, {}).get('title', 'Untitled')} ({url}) ---\n\n"
            if len(current_chunk['content']) + len(header) + len(content) > max_chars:
                if current_chunk['content']: chunks.append(current_chunk)
                current_chunk = {'content': header + content, 'urls': [url]}
            else:
                current_chunk['content'] += header + content
                current_chunk['urls'].append(url)
        if current_chunk['content']: chunks.append(current_chunk)
        return chunks

    def generate_comprehensive_notes(self, scraped_data: dict[str, str], metadata: dict[str, dict], website_url: str, max_chars: int) -> list[str]:
        if not scraped_data:
            return ["No content was scraped to generate notes."]

        chunks = self._chunk_content(scraped_data, metadata, max_chars)
        chain = self.prompt | self.llm | self.output_parser

        sections = []
        for i, chunk in enumerate(chunks):
            logging.info(f"Generating notes for section {i + 1}/{len(chunks)} ({len(chunk['urls'])} pages)...")
            try:
                notes = chain.invoke({"content": chunk['content'], "website_url": website_url})
                if len(chunks) > 1:
                    notes = f"\n\n{'=' * 80}\n# Study Guide Section {i + 1} of {len(chunks)}\n{'=' * 80}\n\n" + notes
                sections.append(notes)
            except Exception as e:
                logging.error(f"Error generating notes for section {i + 1}: {e}")
                sections.append(f"\n\n# ❌ Error in Section {i + 1}: {e}\n")
        return sections


# --- GUI Specific Classes ---

class QueueHandler(logging.Handler):
    """Class to send logging records to a queue."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class ScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Documentation Scraper GUI")
        self.root.geometry("800x600")

        # Load configurations
        self.config = load_config()
        self.prompt_template = load_prompt()

        # Final generated notes content
        self.final_notes_content = ""

        # --- GUI Layout ---
        self.create_widgets()

    def create_widgets(self):
        # Frame for inputs
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.pack(fill=tk.X)

        # URL Input
        ttk.Label(input_frame, text="URL:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.url_var = tk.StringVar(value="https://docs.python.org/3/library/tkinter.html")
        ttk.Entry(input_frame, textvariable=self.url_var, width=60).grid(row=0, column=1, sticky=tk.EW)

        # Page Limit Input
        ttk.Label(input_frame, text="Page Limit:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.limit_var = tk.IntVar(value=10)
        ttk.Spinbox(input_frame, from_=0, to=1000, textvariable=self.limit_var, width=10).grid(row=1, column=1, sticky=tk.W)

        # Start Button
        self.start_button = ttk.Button(input_frame, text="Start Scraping", command=self.start_scraping_thread)
        self.start_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Save Button (initially hidden)
        self.save_button = ttk.Button(input_frame, text="Save Study Guide As...", command=self.save_notes, state=tk.DISABLED)
        self.save_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Log display area
        log_frame = ttk.LabelFrame(self.root, text="Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Setup Logging ---
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(self.queue_handler)
        
        self.root.after(100, self.poll_log_queue)

    def poll_log_queue(self):
        """Check the queue for new log messages and add them to the text widget."""
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, record + '\n')
                self.log_text.configure(state='disabled')
                self.log_text.yview(tk.END)
        self.root.after(100, self.poll_log_queue)

    def start_scraping_thread(self):
        """Starts the scraping process in a new thread to avoid freezing the GUI."""
        self.start_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.final_notes_content = "" # Reset content
        
        threading.Thread(target=self.run_scraper_and_generator, daemon=True).start()

    def run_scraper_and_generator(self):
        """The main logic that runs in the background thread."""
        try:
            # --- 1. Get GUI parameters ---
            url = self.url_var.get()
            limit = self.limit_var.get()

            # --- 2. Load settings from config ---
            api_cfg = self.config.get('api', {})
            llm_cfg = self.config.get('llm', {})
            scraper_cfg = self.config.get('scraper', {})

            api_key_file = api_cfg.get('key_file', 'geminaikey')
            user_agent = scraper_cfg.get('user_agent', 'Mozilla/5.0 ...')
            rate_limit_delay = scraper_cfg.get('rate_limit_delay', 0.5)
            timeout = scraper_cfg.get('request_timeout', 15)
            max_chars = scraper_cfg.get('max_content_chars', 400000)
            
            # --- 3. Run the process ---
            logging.info("--- Starting Process ---")
            api_key = APIKeyManager.get_api_key(api_key_file)
            if not api_key:
                logging.error("API Key not found or invalid. Please check your config.")
                self.process_finished()
                return

            logging.info("Phase 1: Scraping website...")
            scraper = WebsiteScraper(url, limit, user_agent, timeout, rate_limit_delay)
            scraped_data, metadata = scraper.crawl()

            if not scraped_data:
                logging.info("No content was found. Halting process.")
                self.process_finished()
                return

            logging.info("Phase 2: Generating study guide with AI...")
            generator = EnhancedNoteGenerator(api_key, llm_cfg, self.prompt_template)
            notes_sections = generator.generate_comprehensive_notes(scraped_data, metadata, url, max_chars)
            
            # --- 4. Prepare content for saving ---
            self.final_notes_content = "\n".join(notes_sections)
            logging.info("--- Study Guide Generation Complete! ---")
            logging.info("You can now save the file.")
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        finally:
            self.process_finished()

    def process_finished(self):
        """Called when the background thread is done."""
        self.start_button.config(state=tk.NORMAL)

    def save_notes(self):
        """Opens a 'Save As' dialog and saves the generated notes."""
        if not self.final_notes_content:
            logging.warning("No content to save.")
            return

        domain = urlparse(self.url_var.get()).netloc
        safe_domain = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
        initial_filename = f"study_guide_{safe_domain}_{datetime.now():%Y%m%d_%H%M%S}.md"

        filepath = filedialog.asksaveasfilename(
            initialfile=initial_filename,
            defaultextension=".md",
            filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )

        if not filepath:
            logging.info("Save operation cancelled.")
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.final_notes_content)
            logging.info(f"Successfully saved study guide to: {filepath}")
        except IOError as e:
            logging.error(f"Failed to save file: {e}")


if __name__ == '__main__':
    app_root = tk.Tk()
    app = ScraperApp(app_root)
    app_root.mainloop()
