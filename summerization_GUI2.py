import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
import re
from urllib.parse import urlparse
from datetime import datetime

# --- Standard Library Imports for Backend ---
import requests
from pathlib import Path
from urllib.parse import urljoin
from time import sleep
import html2text
import yaml

# --- Third-Party Imports for Backend ---
# Make sure to install these: pip install beautifulsoup4 langchain-google-genai google-generativeai
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Backend Classes (self-contained in this script) ---
# NOTE: These classes are the same as before, with minor logging adjustments.

class APIKeyManager:
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
            logging.info(f"API key loaded successfully from {filepath}.")
            return api_key
        except Exception as e:
            logging.error(f"Error reading API key: {e}")
            return None

class WebsiteScraper:
    def __init__(self, base_url: str, max_pages: int, user_agent: str, request_timeout: int, rate_limit_delay: float):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.request_timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay
        self.scraped_content = {}
        self.visited_urls = set()

    def _get_page_content(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed for {url}: {e}")
        return None

    def _parse_content_and_links(self, html: str, page_url: str) -> tuple[str, list[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup.select('script, style, nav, footer, header, aside, form, .ad'):
            element.decompose()
        main_content_area = soup.select_one('main, article, [role="main"], .main-content, .content') or soup.body
        text_maker = html2text.HTML2Text()
        text_maker.body_width = 0
        text_maker.ignore_links = True
        markdown_content = text_maker.handle(str(main_content_area)) if main_content_area else ""
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content).strip()
        links = {urljoin(page_url, a['href']).split('#')[0] for a in soup.find_all('a', href=True) if urlparse(urljoin(page_url, a['href'])).netloc == self.domain}
        return markdown_content, list(links)

    def crawl(self) -> dict[str, str]:
        urls_to_visit = {self.base_url}
        while urls_to_visit:
            if 0 < self.max_pages <= len(self.visited_urls):
                logging.info(f"Reached scraping limit of {self.max_pages} pages.")
                break
            url = urls_to_visit.pop()
            if url in self.visited_urls: continue
            self.visited_urls.add(url)
            logging.info(f"[{len(self.visited_urls)}/{self.max_pages or '∞'}] Scraping: {url}")
            html = self._get_page_content(url)
            if html:
                text, new_links = self._parse_content_and_links(html, url)
                if text and len(text.strip()) > 150:
                    self.scraped_content[url] = text
                urls_to_visit.update(new_links)
            sleep(self.rate_limit_delay)
        logging.info(f"Crawling complete. Scraped {len(self.scraped_content)} valid pages.")
        return self.scraped_content

class EnhancedNoteGenerator:
    def __init__(self, api_key: str, llm_config: dict, prompt_template_string: str):
        self.llm = ChatGoogleGenerativeAI(model=llm_config['model_name'], google_api_key=api_key, **llm_config['parameters'])
        self.output_parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_template(prompt_template_string)

    def generate_comprehensive_notes(self, scraped_data: dict[str, str], website_url: str) -> str:
        if not scraped_data: return "No content was scraped to generate notes."
        logging.info(f"Generating notes from {len(scraped_data)} pages...")
        full_content = "\n\n---NEW PAGE---\n".join(scraped_data.values())
        chain = self.prompt | self.llm | self.output_parser
        try:
            notes = chain.invoke({"content": full_content, "website_url": website_url})
            return notes
        except Exception as e:
            logging.error(f"Error during note generation: {e}")
            return f"# Generation Error\n\nAn error occurred while communicating with the AI model:\n\n`{e}`"

# --- GUI Specific Classes ---

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class ScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Documentation Scraper GUI")
        self.root.geometry("850x700")

        # --- Variables for GUI Widgets ---
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar()
        self.delay_var = tk.DoubleVar()
        self.api_key_file_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()

        self.final_notes_content = ""
        self.create_widgets()
        self.load_initial_settings()

    def create_widgets(self):
        # --- Main Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Settings Pane (Left)
        settings_pane = ttk.Frame(main_frame)
        settings_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Log Pane (Right)
        log_pane = ttk.Frame(main_frame)
        log_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Settings Notebook ---
        notebook = ttk.Notebook(settings_pane)
        notebook.pack(fill="both", expand=True)

        # Scraper Settings Tab
        scraper_tab = ttk.Frame(notebook, padding="10")
        ttk.Label(scraper_tab, text="Target URL:").pack(fill=tk.X, pady=(0, 2))
        ttk.Entry(scraper_tab, textvariable=self.url_var).pack(fill=tk.X)
        ttk.Label(scraper_tab, text="Page Limit (0=unlimited):").pack(fill=tk.X, pady=(10, 2))
        ttk.Spinbox(scraper_tab, from_=0, to=1000, textvariable=self.limit_var).pack(fill=tk.X)
        ttk.Label(scraper_tab, text="Request Delay (seconds):").pack(fill=tk.X, pady=(10, 2))
        ttk.Scale(scraper_tab, from_=0.1, to=5.0, orient=tk.HORIZONTAL, variable=self.delay_var).pack(fill=tk.X)

        # AI/Model Settings Tab
        ai_tab = ttk.Frame(notebook, padding="10")
        ttk.Label(ai_tab, text="API Key File:").pack(fill=tk.X, pady=(0, 2))
        api_frame = ttk.Frame(ai_tab)
        api_frame.pack(fill=tk.X)
        ttk.Entry(api_frame, textvariable=self.api_key_file_var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(api_frame, text="...", width=3, command=self.browse_api_key).pack(side=tk.RIGHT)
        ttk.Label(ai_tab, text="Model Name:").pack(fill=tk.X, pady=(10, 2))
        ttk.Entry(ai_tab, textvariable=self.model_name_var).pack(fill=tk.X)
        ttk.Label(ai_tab, text="Temperature:").pack(fill=tk.X, pady=(10, 2))
        ttk.Scale(ai_tab, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var).pack(fill=tk.X)
        ttk.Label(ai_tab, text="Max Output Tokens:").pack(fill=tk.X, pady=(10, 2))
        ttk.Spinbox(ai_tab, from_=1024, to=16384, increment=128, textvariable=self.max_tokens_var).pack(fill=tk.X)
        
        # Prompt Editor Tab
        prompt_tab = ttk.Frame(notebook, padding="10")
        ttk.Button(prompt_tab, text="Load Prompt from File...", command=self.load_prompt_from_file).pack(fill=tk.X, pady=(0,5))
        self.prompt_text = scrolledtext.ScrolledText(prompt_tab, wrap=tk.WORD, height=10)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        notebook.add(scraper_tab, text="Scraper")
        notebook.add(ai_tab, text="AI Model")
        notebook.add(prompt_tab, text="Prompt")
        
        # --- Control Buttons ---
        self.start_button = ttk.Button(settings_pane, text="Start Generation", command=self.start_task_thread)
        self.start_button.pack(fill=tk.X, pady=(10, 0))
        self.save_button = ttk.Button(settings_pane, text="Save Study Guide As...", command=self.save_notes, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)
        
        # --- Log Viewer ---
        log_label = ttk.LabelFrame(log_pane, text="Logs")
        log_label.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget = scrolledtext.ScrolledText(log_label, state='disabled', wrap=tk.WORD)
        self.log_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        logging.getLogger().addHandler(self.queue_handler)
        logging.getLogger().setLevel(logging.INFO)
        self.root.after(100, self.poll_log_queue)

    def load_initial_settings(self):
        config = self.load_config_file()
        prompt_template = self.load_prompt_file()
        
        scraper_cfg = config.get('scraper', {})
        api_cfg = config.get('api', {})
        llm_cfg = config.get('llm', {})
        llm_params = llm_cfg.get('parameters', {})

        self.url_var.set("https://docs.python.org/3/")
        self.limit_var.set(scraper_cfg.get('page_limit', 10))
        self.delay_var.set(scraper_cfg.get('rate_limit_delay', 0.5))
        self.api_key_file_var.set(api_cfg.get('key_file', 'geminaikey'))
        self.model_name_var.set(llm_cfg.get('model_name', 'gemini-1.5-pro'))
        self.temperature_var.set(llm_params.get('temperature', 0.5))
        self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
        self.prompt_text.insert(tk.END, prompt_template)

    def load_config_file(self, filepath="config.yml"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError): return {}

    def load_prompt_file(self, filepath="prompt.md"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return f.read()
        except FileNotFoundError: return "Default prompt: Could not find prompt.md"

    def browse_api_key(self):
        filepath = filedialog.askopenfilename(title="Select API Key File")
        if filepath: self.api_key_file_var.set(filepath)

    def load_prompt_from_file(self):
        filepath = filedialog.askopenfilename(title="Select Prompt File", filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt")])
        if filepath:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, self.load_prompt_file(filepath))

    def poll_log_queue(self):
        while True:
            try: record = self.log_queue.get(block=False)
            except queue.Empty: break
            else:
                self.log_text_widget.configure(state='normal')
                self.log_text_widget.insert(tk.END, record + '\n')
                self.log_text_widget.configure(state='disabled')
                self.log_text_widget.yview(tk.END)
        self.root.after(100, self.poll_log_queue)

    def start_task_thread(self):
        self.start_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.final_notes_content = ""
        threading.Thread(target=self.run_full_process, daemon=True).start()

    def run_full_process(self):
        try:
            # --- Get all parameters from the GUI at runtime ---
            url = self.url_var.get()
            limit = self.limit_var.get()
            delay = self.delay_var.get()
            api_key_file = self.api_key_file_var.get()
            model_name = self.model_name_var.get()
            temperature = self.temperature_var.get()
            max_tokens = self.max_tokens_var.get()
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            
            if not url or not api_key_file or not model_name or not prompt:
                logging.error("Missing critical inputs (URL, API Key File, Model, or Prompt).")
                messagebox.showerror("Input Error", "Please ensure URL, API Key file, Model Name, and Prompt are all set.")
                return

            logging.info("--- Starting Process ---")
            api_key = APIKeyManager.get_api_key(api_key_file)
            if not api_key:
                logging.error("API Key is invalid. Halting.")
                return

            # Use defaults from a generic config if not in GUI (e.g. user_agent)
            scraper_cfg = self.load_config_file().get('scraper', {})
            user_agent = scraper_cfg.get('user_agent', 'Mozilla/5.0')
            timeout = scraper_cfg.get('request_timeout', 15)
            
            scraper = WebsiteScraper(url, limit, user_agent, timeout, delay)
            scraped_data = scraper.crawl()

            if not scraped_data:
                logging.warning("No content was found. Halting process.")
                return

            llm_config = {'model_name': model_name, 'parameters': {'temperature': temperature, 'max_output_tokens': max_tokens}}
            generator = EnhancedNoteGenerator(api_key, llm_config, prompt)
            self.final_notes_content = generator.generate_comprehensive_notes(scraped_data, url)
            
            logging.info("--- Study Guide Generation Complete! ---")
            logging.info("Ready to save the file.")
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            logging.error(f"An unexpected error occurred in the main process: {e}", exc_info=True)
            messagebox.showerror("Runtime Error", f"An error occurred:\n\n{e}")
        finally:
            self.start_button.config(state=tk.NORMAL)

    def save_notes(self):
        if not self.final_notes_content:
            messagebox.showwarning("No Content", "There is no generated content to save.")
            return

        initial_filename = f"study_guide_{urlparse(self.url_var.get()).netloc}.md"
        filepath = filedialog.asksaveasfilename(
            initialfile=initial_filename,
            defaultextension=".md",
            filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt")]
        )
        if not filepath:
            logging.info("Save operation cancelled.")
            return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.final_notes_content)
            logging.info(f"Successfully saved study guide to: {filepath}")
            messagebox.showinfo("Success", f"File saved successfully to:\n{filepath}")
        except IOError as e:
            logging.error(f"Failed to save file: {e}")
            messagebox.showerror("Save Error", f"Could not save the file:\n\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = ScraperApp(root)
    root.mainloop()
