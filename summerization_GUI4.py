import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
import re
from urllib.parse import urlparse, quote, unquote
from datetime import datetime
import os
import random
import shutil
import subprocess

# --- Backend Standard Library Imports ---
import requests
from pathlib import Path
from urllib.parse import urljoin
from time import sleep
import html2text
import yaml # You need to 'pip install pyyaml'
import json

# --- Dependency Availability Flags ---
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# --- Backend Third-Party Imports ---
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz
import docx

# --- Backend Classes (Fully Integrated) ---

class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str) -> str | None:
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f: key = f.read().strip()
            if not key: raise ValueError("Key is empty")
            logging.info(f"Gemini API key loaded from {filepath}.")
            return key
        except Exception as e:
            logging.error(f"Failed to load Gemini API key from {filepath}: {e}")
            return None

    @staticmethod
    def get_google_search_credentials(key_path: str, cx_path: str) -> tuple[str | None, str | None]:
        try:
            with open(Path(key_path), 'r') as f: api_key = f.read().strip()
            with open(Path(cx_path), 'r') as f: cx_id = f.read().strip()
            if not api_key or not cx_id: raise ValueError("Key or CX is empty")
            logging.info("Google Search credentials loaded successfully.")
            return api_key, cx_id
        except Exception as e:
            logging.warning(f"Could not load Google Search credentials: {e}. Will use fallback.")
            return None, None

class LocalDocumentLoader:
    def __init__(self, file_paths: list[str]): self.file_paths = file_paths
    def _read_txt(self, p: Path) -> str: return p.read_text(encoding='utf-8', errors='ignore')
    def _read_pdf(self, p: Path) -> str:
        with fitz.open(p) as doc: return "".join(page.get_text() for page in doc)
    def _read_docx(self, p: Path) -> str:
        return "\n".join(para.text for para in docx.Document(p).paragraphs)
    def load_and_extract_text(self) -> dict[str, str]:
        content = {}
        for path_str in self.file_paths:
            p, name = Path(path_str), Path(path_str).name
            logging.info(f"Processing local file: {name}")
            try:
                if name.lower().endswith(".pdf"): content[name] = self._read_pdf(p)
                elif name.lower().endswith(".docx"): content[name] = self._read_docx(p)
                elif name.lower().endswith(".txt"): content[name] = self._read_txt(p)
                else: logging.warning(f"Unsupported file type skipped: {name}")
            except Exception as e: logging.error(f"Failed to process file '{name}': {e}")
        logging.info(f"Successfully processed {len(content)} local documents.")
        return content

class ResearchQueryGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5)
        self.output_parser = StrOutputParser()
    def extract_main_topic(self, content: dict[str, str]) -> str:
        logging.info("🧠 Identifying main topic using LLM...")
        context = "\n---\n".join(list(content.values())[:5])[:4000]
        prompt = ChatPromptTemplate.from_template("Analyze the text snippets and identify the primary technology/topic. Respond with ONLY the name (e.g., 'React Router').\n\nCONTENT:\n{context}\n\nTOPIC:")
        try:
            topic = (prompt | self.llm | self.output_parser).invoke({"context": context}).strip().replace('"', '')
            logging.info(f"✅ LLM identified main topic: {topic}")
            return topic
        except Exception as e:
            logging.error(f"❌ Topic identification failed: {e}. Falling back to 'General Topic'.")
            return "General Topic"
    def generate_research_queries(self, topic: str, max_queries: int) -> list[str]:
        logging.info(f"🧠 Generating research queries for '{topic}'...")
        prompt = ChatPromptTemplate.from_template("Generate a JSON array of {max_queries} diverse search queries for tutorials, issues, and best practices for '{topic}'.\n\nOUTPUT (JSON array only):")
        try:
            result = (prompt | self.llm | self.output_parser).invoke({"topic": topic, "max_queries": max_queries})
            queries = json.loads(re.search(r'\[.*\]', result, re.DOTALL).group(0))
            logging.info(f"✅ Generated {len(queries)} research queries.")
            return queries
        except Exception as e:
            logging.error(f"❌ Query generation failed: {e}. Using fallback queries.")
            return [f"{topic} tutorial", f"{topic} common errors", f"{topic} best practices", f"{topic} video guide"]

class GoogleSearchResearcher:
    def __init__(self, api_key: str = None, cx_id: str = None):
        self.api_key, self.cx_id = api_key, cx_id
        self.session = requests.Session()
    def search_and_extract_urls(self, queries: list[str], exclude_domain: str) -> list[str]:
        urls = set()
        for query in queries:
            try:
                if self.api_key and self.cx_id:
                    params = {'key': self.api_key, 'cx': self.cx_id, 'q': query, 'num': 5}
                    response = self.session.get("https://www.googleapis.com/customsearch/v1", params=params)
                    response.raise_for_status()
                    results = response.json().get('items', [])
                    urls.update(item['link'] for item in results)
                else:
                    response = self.session.get("https://html.duckduckgo.com/html/", params={'q': query})
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    urls.update(unquote(a['href'].split('uddg=')[1]) for a in soup.select('a.result__a') if 'uddg=' in a['href'])
            except Exception as e:
                logging.error(f"Web search failed for query '{query}': {e}")
            sleep(1)
        return [url for url in list(urls) if exclude_domain not in url and not any(url.endswith(e) for e in ['.pdf', '.zip'])]


# --- GUI Application ---

class QueueHandler(logging.Handler):
    """Class to send logging records to a queue."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Research & Study Guide Generator (Alpha)")
        self.root.geometry("1000x800")

        # --- TKINTER VARIABLES ---
        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=0) # Default value
        self.research_enabled_var = tk.BooleanVar(value=False)
        self.web_research_enabled_var = tk.BooleanVar(value=True)
        self.yt_research_enabled_var = tk.BooleanVar(value=False)
        self.web_research_method_var = tk.StringVar(value="google_api")
        self.research_pages_var = tk.IntVar(value=5)
        self.google_api_key_file_var = tk.StringVar()
        self.google_cx_file_var = tk.StringVar()
        self.yt_videos_per_query_var = tk.IntVar(value=3)
        self.api_key_file_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        self.final_notes_content = ""
        self.config = {} # To store loaded config

        # --- UI AND LOGGING SETUP ---
        self.create_widgets()
        self.load_initial_settings() # Load settings from config.yml
        self.toggle_input_mode()
        self.toggle_research_panel()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        settings_pane = ttk.Frame(main_frame, width=400); settings_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        log_pane = ttk.Frame(main_frame); log_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(settings_pane); notebook.pack(fill="both", expand=True)

        source_tab, self.research_tab, ai_tab, prompt_tab = ttk.Frame(notebook, padding=10), ttk.Frame(notebook, padding=10), ttk.Frame(notebook, padding=10), ttk.Frame(notebook, padding=10)
        
        # --- Tab 1: Source ---
        mode_frame = ttk.LabelFrame(source_tab, text="Input Mode", padding=10); mode_frame.pack(fill=tk.X)
        ttk.Radiobutton(mode_frame, text="Scrape from URL", variable=self.input_mode_var, value="scrape", command=self.toggle_input_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Upload Local Documents", variable=self.input_mode_var, value="upload", command=self.toggle_input_mode).pack(anchor=tk.W)
        self.scraper_frame = ttk.LabelFrame(source_tab, text="Web Scraper Settings", padding=10); self.scraper_frame.pack(fill=tk.X, pady=10)
        ttk.Label(self.scraper_frame, text="Target URL:").pack(fill=tk.X)
        ttk.Entry(self.scraper_frame, textvariable=self.url_var).pack(fill=tk.X)
        ttk.Label(self.scraper_frame, text="Page Limit (0=unlimited):").pack(fill=tk.X, pady=(5,0))
        ttk.Spinbox(self.scraper_frame, from_=0, to=1000, textvariable=self.limit_var).pack(fill=tk.X)
        self.upload_frame = ttk.LabelFrame(source_tab, text="Local Document Settings", padding=10); self.upload_frame.pack(fill=tk.X, pady=5)
        btn_frame = ttk.Frame(self.upload_frame); btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Add Files...", command=self.add_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        ttk.Button(btn_frame, text="Clear List", command=self.clear_files).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        self.file_listbox = tk.Listbox(self.upload_frame, height=4); self.file_listbox.pack(fill=tk.X, pady=5)
        
        # --- Tab 2: Research ---
        master_frame = ttk.LabelFrame(self.research_tab, text="Master Control", padding=5); master_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(master_frame, text="Enable AI Research (Alpha)", variable=self.research_enabled_var, command=self.toggle_research_panel).pack(anchor=tk.W)
        self.web_research_panel = ttk.LabelFrame(self.research_tab, text="Web Research", padding=10); self.web_research_panel.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(self.web_research_panel, text="Enable Web Search", variable=self.web_research_enabled_var).pack(anchor=tk.W)
        ttk.Label(self.web_research_panel, text="Method:").pack(anchor=tk.W, pady=(5,0))
        self.g_api_radio = ttk.Radiobutton(self.web_research_panel, text="Google API / DDG (Fast)", variable=self.web_research_method_var, value="google_api"); self.g_api_radio.pack(anchor=tk.W, padx=10)
        self.playwright_radio = ttk.Radiobutton(self.web_research_panel, text="Playwright (Slow, Robust)", variable=self.web_research_method_var, value="playwright"); self.playwright_radio.pack(anchor=tk.W, padx=10)
        ttk.Label(self.web_research_panel, text="Research Pages to Scrape:").pack(anchor=tk.W, pady=(5,0))
        ttk.Spinbox(self.web_research_panel, from_=1, to=50, textvariable=self.research_pages_var).pack(fill=tk.X, padx=10)
        self.yt_research_panel = ttk.LabelFrame(self.research_tab, text="YouTube Research", padding=10); self.yt_research_panel.pack(fill=tk.X, pady=5)
        self.yt_checkbutton = ttk.Checkbutton(self.yt_research_panel, text="Enable Transcript Analysis", variable=self.yt_research_enabled_var); self.yt_checkbutton.pack(anchor=tk.W)
        ttk.Label(self.yt_research_panel, text="Videos to Check per Query:").pack(anchor=tk.W, pady=(5,0))
        ttk.Spinbox(self.yt_research_panel, from_=1, to=10, textvariable=self.yt_videos_per_query_var).pack(fill=tk.X, padx=10)
        
        # --- Tab 3: AI Model ---
        ttk.Label(ai_tab, text="Gemini API Key File:").pack(fill=tk.X, pady=(0, 2))
        api_frame = ttk.Frame(ai_tab); api_frame.pack(fill=tk.X)
        ttk.Entry(api_frame, textvariable=self.api_key_file_var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(api_frame, text="...", width=3, command=self.browse_api_key).pack(side=tk.RIGHT)
        ttk.Label(ai_tab, text="Model Name:").pack(fill=tk.X, pady=(10, 2))
        ttk.Entry(ai_tab, textvariable=self.model_name_var).pack(fill=tk.X)
        ttk.Label(ai_tab, text="Temperature:").pack(fill=tk.X, pady=(10, 2))
        ttk.Scale(ai_tab, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var).pack(fill=tk.X)
        ttk.Label(ai_tab, text="Max Output Tokens:").pack(fill=tk.X, pady=(10, 2))
        ttk.Spinbox(ai_tab, from_=1024, to=16384, increment=128, textvariable=self.max_tokens_var).pack(fill=tk.X)

        # --- Tab 4: Prompt ---
        ttk.Button(prompt_tab, text="Load Prompt from File...", command=self.load_prompt_from_file).pack(fill=tk.X, pady=(0,5))
        self.prompt_text = scrolledtext.ScrolledText(prompt_tab, wrap=tk.WORD, height=10); self.prompt_text.pack(fill=tk.BOTH, expand=True)

        notebook.add(source_tab, text="1. Source")
        notebook.add(self.research_tab, text="2. Research (Alpha)")
        notebook.add(ai_tab, text="3. AI Model")
        notebook.add(prompt_tab, text="4. Prompt")

        self.start_button = ttk.Button(settings_pane, text="Start Generation", command=self.start_task_thread); self.start_button.pack(fill=tk.X, pady=(10, 0))
        self.save_button = ttk.Button(settings_pane, text="Save Study Guide As...", command=self.save_notes, state=tk.DISABLED); self.save_button.pack(fill=tk.X, pady=5)
        log_label = ttk.LabelFrame(log_pane, text="Logs"); log_label.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget = scrolledtext.ScrolledText(log_label, state='disabled', wrap=tk.WORD); self.log_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_queue = queue.Queue(); self.queue_handler = QueueHandler(self.log_queue)
        logging.getLogger().addHandler(self.queue_handler); logging.getLogger().setLevel(logging.INFO)
        self.root.after(100, self.poll_log_queue)

    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            try: widget.configure(state=state)
            except tk.TclError: pass
            self._set_child_widgets_state(widget, state)

    def toggle_input_mode(self):
        mode = self.input_mode_var.get()
        if mode == "scrape":
            self._set_child_widgets_state(self.scraper_frame, tk.NORMAL)
            self._set_child_widgets_state(self.upload_frame, tk.DISABLED)
        else:
            self._set_child_widgets_state(self.scraper_frame, tk.DISABLED)
            self._set_child_widgets_state(self.upload_frame, tk.NORMAL)

    def add_files(self):
        filetypes = [("All Supported", "*.pdf *.docx *.txt"), ("PDF", "*.pdf"), ("Word", "*.docx"), ("Text", "*.txt")]
        files = filedialog.askopenfilenames(title="Select Documents", filetypes=filetypes)
        for f in files:
            if f not in self.file_listbox.get(0, tk.END): self.file_listbox.insert(tk.END, f)
    
    def clear_files(self): self.file_listbox.delete(0, tk.END)
    
    def browse_api_key(self):
        filepath = filedialog.askopenfilename(title="Select Gemini API Key File");
        if filepath: self.api_key_file_var.set(filepath)
    
    def load_prompt_from_file(self):
        filepath = filedialog.askopenfilename(title="Select Prompt File", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if filepath:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, self._load_prompt_file(filepath))
            
    def toggle_research_panel(self):
        state = tk.NORMAL if self.research_enabled_var.get() else tk.DISABLED
        self._set_child_widgets_state(self.web_research_panel, state)
        self._set_child_widgets_state(self.yt_research_panel, state)
        if not PLAYWRIGHT_AVAILABLE: self.playwright_radio.config(state=tk.DISABLED)
        if not YT_DLP_AVAILABLE: self.yt_checkbutton.config(state=tk.DISABLED)

    def load_initial_settings(self):
        """Loads settings from config.yml and prompt.md into the UI."""
        self.config = self._load_config_file()
        prompt_template = self._load_prompt_file()
        
        if self.config:
            logging.info("📝 Loading settings from config.yml...")
            # Safely get values using .get() to avoid errors if keys are missing
            api_settings = self.config.get('api', {})
            llm_settings = self.config.get('llm', {})
            llm_params = llm_settings.get('parameters', {})
            
            self.api_key_file_var.set(api_settings.get('key_file', 'gemini_api.key'))
            self.model_name_var.set(llm_settings.get('model_name', 'gemini-1.5-flash'))
            self.temperature_var.set(llm_params.get('temperature', 0.5))
            self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
            logging.info("✅ Default settings loaded.")
        else:
            logging.warning("Could not find or load config.yml. Using default fallback values.")

        if prompt_template:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, prompt_template)
            logging.info("✅ Default prompt loaded from prompt.md.")
        
        # Check for dependencies
        if not PLAYWRIGHT_AVAILABLE:
            self.playwright_radio.config(state=tk.DISABLED)
            logging.warning("Playwright not found. To enable, 'pip install playwright' and 'playwright install'.")
            if self.web_research_method_var.get() == 'playwright': self.web_research_method_var.set('google_api')
        if not YT_DLP_AVAILABLE:
            self.yt_checkbutton.config(state=tk.DISABLED)
            self.yt_research_enabled_var.set(False)
            logging.warning("yt-dlp not found. To enable YouTube research, 'pip install yt-dlp'.")

    def poll_log_queue(self):
        while True:
            try: record = self.log_queue.get(block=False)
            except queue.Empty: break
            else:
                self.log_text_widget.config(state='normal'); self.log_text_widget.insert(tk.END, record + '\n'); self.log_text_widget.config(state='disabled'); self.log_text_widget.yview(tk.END)
        self.root.after(100, self.poll_log_queue)
    
    def save_notes(self):
        if not self.final_notes_content: messagebox.showwarning("No Content", "Nothing to save."); return
        source_name = urlparse(self.url_var.get()).netloc if self.input_mode_var.get() == "scrape" else "local_docs"
        initial_filename = f"study_guide_{source_name}.md"
        filepath = filedialog.asksaveasfilename(initialfile=initial_filename, defaultextension=".md", filetypes=[("Markdown", "*.md")])
        if not filepath: logging.info("Save cancelled."); return
        try:
            with open(filepath, "w", encoding="utf-8") as f: f.write(self.final_notes_content)
            logging.info(f"Successfully saved to: {filepath}")
            messagebox.showinfo("Success", f"File saved to:\n{filepath}")
        except IOError as e: messagebox.showerror("Save Error", f"Could not save file:\n\n{e}")

    def start_task_thread(self):
        self.start_button.config(state=tk.DISABLED); self.save_button.config(state=tk.DISABLED)
        self.final_notes_content = ""
        threading.Thread(target=self.run_full_process, daemon=True).start()

    def run_full_process(self):
        try:
            api_key_file = self.api_key_file_var.get(); model_name = self.model_name_var.get(); prompt = self.prompt_text.get("1.0", tk.END).strip()
            if not all([api_key_file, model_name, prompt]):
                messagebox.showerror("Input Error", "API Key File, Model, and Prompt must be set."); return
            
            api_key = APIKeyManager.get_gemini_api_key(api_key_file)
            if not api_key: logging.error("API Key is invalid. Halting."); return

            source_data, source_name, mode = {}, "", self.input_mode_var.get()

            # The 'scraper' settings from config.yml would be used here.
            # For example: delay = self.config.get('scraper', {}).get('rate_limit_delay', 0.5)
            
            if mode == "scrape":
                logging.info("--- Starting in Web Scrape Mode ---")
                url = self.url_var.get()
                if not url: messagebox.showerror("Input Error", "URL cannot be empty."); return
                limit = self.limit_var.get()
                # You need to define WebsiteScraper class or import it
                # scraper = WebsiteScraper(url, limit, "Mozilla/5.0", 15, delay)
                # source_data = scraper.crawl()
                source_name = urlparse(url).netloc
            else: # "upload"
                logging.info("--- Starting in Local Document Mode ---")
                file_paths = self.file_listbox.get(0, tk.END)
                if not file_paths: messagebox.showerror("Input Error", "Please add at least one document."); return
                loader = LocalDocumentLoader(file_paths)
                source_data = loader.load_and_extract_text()
                source_name = f"{len(file_paths)}_local_documents"

            if not source_data:
                logging.warning("No text content could be extracted. Halting process."); return

            # ... Rest of the logic ...
            logging.info("--- Process placeholder ---")
            logging.info("Main logic would execute here.")
            sleep(2) # Simulate work
            self.final_notes_content = "# Study Guide\n\nProcess completed."
            logging.info("✅ Process finished.")
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            messagebox.showerror("Runtime Error", f"An error occurred:\n\n{e}")
        finally:
            self.start_button.config(state=tk.NORMAL)
            
    def _load_config_file(self, filepath="config.yml"):
        """Loads configuration from a YAML file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.warning(f"Could not load or parse '{filepath}': {e}")
            return {}
        
    def _load_prompt_file(self, filepath="prompt.md"):
        """Loads a text or markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"Prompt file not found at '{filepath}'. Prompt will be empty.")
            return ""

if __name__ == '__main__':
    root = tk.Tk()
    app = AdvancedScraperApp(root)
    root.mainloop()
