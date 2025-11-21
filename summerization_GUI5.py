import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
import re
from urllib.parse import urlparse, unquote
import os

# --- Backend Standard Library Imports ---
import requests
from pathlib import Path
from time import sleep
import yaml
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
import html2text

# Conditional imports for optional features
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# --- Backend Classes (Fully Integrated) ---

class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str) -> str | None:
        if not LANGCHAIN_AVAILABLE:
            logging.error("Langchain or Google Generative AI library not installed.")
            return None
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f: key = f.read().strip()
            if not key: raise ValueError("Key is empty")
            logging.info(f"Gemini API key loaded from {filepath}.")
            return key
        except Exception as e:
            logging.error(f"Failed to load Gemini API key from {filepath}: {e}")
            return None

class LocalDocumentLoader:
    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def _read_txt(self, p: Path) -> str:
        return p.read_text(encoding='utf-8', errors='ignore')

    def _read_pdf(self, p: Path) -> str:
        if not FITZ_AVAILABLE:
            logging.error("PyMuPDF (fitz) is not installed. Cannot process PDF files.")
            return ""
        with fitz.open(p) as doc:
            return "".join(page.get_text() for page in doc)

    def _read_docx(self, p: Path) -> str:
        if not DOCX_AVAILABLE:
            logging.error("python-docx is not installed. Cannot process DOCX files.")
            return ""
        doc = docx.Document(p)
        return "\n".join(para.text for para in doc.paragraphs)

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
            except Exception as e:
                logging.error(f"Failed to process file '{name}': {e}")
        logging.info(f"Successfully processed {len(content)} local documents.")
        return content

class WebsiteScraper:
    def __init__(self, url: str, user_agent: str, timeout: int = 15):
        self.start_url = url
        self.headers = {'User-Agent': user_agent}
        self.timeout = timeout
        self.session = requests.Session()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for script_or_style in soup(["script", "style", "nav", "footer", "aside"]):
            script_or_style.decompose()
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_tables = True
        return h.handle(str(soup))

    def scrape_single_page(self) -> dict[str, str]:
        logging.info(f"Scraping single URL: {self.start_url}")
        try:
            response = self.session.get(self.start_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logging.warning(f"Skipped non-HTML content at {self.start_url}")
                return {}
            soup = BeautifulSoup(response.content, 'lxml')
            text_content = self._extract_text(soup)
            if not text_content.strip():
                logging.warning(f"No text content found at {self.start_url}")
                return {}
            logging.info(f"Successfully extracted content from {self.start_url}")
            return {self.start_url: text_content}
        except requests.RequestException as e:
            logging.error(f"Failed to fetch URL {self.start_url}: {e}")
            return {}
        except Exception as e:
            logging.error(f"An unexpected error occurred while scraping {self.start_url}: {e}")
            return {}

# --- GUI Application ---

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Research & Study Guide Generator")
        self.root.geometry("1000x800")
        self.config = {}

        # --- TKINTER VARIABLES ---
        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.api_key_file_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        self.final_notes_content = ""

        # --- UI AND LOGGING SETUP ---
        self.create_widgets()
        self.load_initial_settings()
        self.toggle_input_mode()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        settings_pane = ttk.Frame(main_frame, width=350); settings_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')
        log_pane = ttk.Frame(main_frame); log_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # --- Settings Area ---
        source_frame = ttk.LabelFrame(settings_pane, text="1. Input Source", padding=10); source_frame.pack(fill=tk.X)
        ai_frame = ttk.LabelFrame(settings_pane, text="2. AI Configuration", padding=10); ai_frame.pack(fill=tk.X, pady=10)
        prompt_frame = ttk.LabelFrame(settings_pane, text="3. System Prompt", padding=10); prompt_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Input Source Frame ---
        ttk.Radiobutton(source_frame, text="Scrape from URL", variable=self.input_mode_var, value="scrape", command=self.toggle_input_mode).pack(anchor=tk.W)
        self.scraper_frame = ttk.Frame(source_frame, padding=(20, 5, 0, 5))
        self.scraper_frame.pack(fill=tk.X)
        ttk.Label(self.scraper_frame, text="Target URL:").pack(fill=tk.X)
        ttk.Entry(self.scraper_frame, textvariable=self.url_var).pack(fill=tk.X)
        
        ttk.Radiobutton(source_frame, text="Upload Local Documents", variable=self.input_mode_var, value="upload", command=self.toggle_input_mode).pack(anchor=tk.W, pady=(10,0))
        self.upload_frame = ttk.Frame(source_frame, padding=(20, 5, 0, 5))
        self.upload_frame.pack(fill=tk.X)
        btn_frame = ttk.Frame(self.upload_frame); btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Add Files...", command=self.add_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        ttk.Button(btn_frame, text="Clear List", command=self.clear_files).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        self.file_listbox = tk.Listbox(self.upload_frame, height=3); self.file_listbox.pack(fill=tk.X, pady=5)
        
        # --- AI Model Frame ---
        ttk.Label(ai_frame, text="Gemini API Key File:").pack(fill=tk.X, pady=(0, 2))
        api_key_frame = ttk.Frame(ai_frame); api_key_frame.pack(fill=tk.X)
        ttk.Entry(api_key_frame, textvariable=self.api_key_file_var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(api_key_frame, text="...", width=3, command=self.browse_api_key).pack(side=tk.RIGHT)
        ttk.Label(ai_frame, text="Model Name:").pack(fill=tk.X, pady=(10, 2))
        ttk.Entry(ai_frame, textvariable=self.model_name_var).pack(fill=tk.X)
        
        # --- Prompt Frame ---
        ttk.Button(prompt_frame, text="Load Prompt from File...", command=self.load_prompt_from_file).pack(fill=tk.X, pady=(0,5))
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=10); self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # --- Action Buttons ---
        action_frame = ttk.Frame(settings_pane)
        action_frame.pack(fill=tk.X, pady=(10,0))
        self.start_button = ttk.Button(action_frame, text="Start Generation", command=self.start_task_thread); self.start_button.pack(fill=tk.X)
        self.save_button = ttk.Button(action_frame, text="Save Study Guide As...", command=self.save_notes, state=tk.DISABLED); self.save_button.pack(fill=tk.X, pady=5)
        
        # --- Log Pane ---
        log_label = ttk.LabelFrame(log_pane, text="Logs"); log_label.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget = scrolledtext.ScrolledText(log_label, state='disabled', wrap=tk.WORD); self.log_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_queue = queue.Queue(); self.queue_handler = QueueHandler(self.log_queue)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().addHandler(self.queue_handler)
        self.root.after(100, self.poll_log_queue)

    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            try: widget.configure(state=state)
            except tk.TclError: pass
            if widget.winfo_children(): self._set_child_widgets_state(widget, state)

    def toggle_input_mode(self):
        mode = self.input_mode_var.get()
        if mode == "scrape":
            self._set_child_widgets_state(self.scraper_frame, tk.NORMAL)
            self._set_child_widgets_state(self.upload_frame, tk.DISABLED)
        else:
            self._set_child_widgets_state(self.scraper_frame, tk.DISABLED)
            self._set_child_widgets_state(self.upload_frame, tk.NORMAL)

    def add_files(self):
        filetypes = [("Supported Files", "*.pdf *.docx *.txt"), ("All Files", "*.*")]
        files = filedialog.askopenfilenames(title="Select Documents", filetypes=filetypes)
        for f in files:
            if f not in self.file_listbox.get(0, tk.END): self.file_listbox.insert(tk.END, f)
    
    def clear_files(self): self.file_listbox.delete(0, tk.END)
    
    def browse_api_key(self):
        filepath = filedialog.askopenfilename(title="Select API Key File");
        if filepath: self.api_key_file_var.set(filepath)
    
    def load_prompt_from_file(self):
        filepath = filedialog.askopenfilename(title="Select Prompt File", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if filepath:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, self._load_prompt_file(filepath))

    def load_initial_settings(self):
        self.config = self._load_config_file()
        prompt_template = self._load_prompt_file()
        
        if self.config:
            logging.info("📝 Loading settings from config.yml...")
            api_settings = self.config.get('api', {})
            llm_settings = self.config.get('llm', {})
            llm_params = llm_settings.get('parameters', {})
            
            self.api_key_file_var.set(api_settings.get('key_file', ''))
            self.model_name_var.set(llm_settings.get('model_name', 'gemini-1.5-pro-latest'))
            self.temperature_var.set(llm_params.get('temperature', 0.5))
            self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
            logging.info("✅ Default settings loaded.")
        else:
            logging.warning("Could not find or load config.yml. Using blank values.")

        if prompt_template:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, prompt_template)
            logging.info("✅ Default prompt loaded from prompt.md.")
        
        self.check_dependencies()

    def check_dependencies(self):
        if not FITZ_AVAILABLE:
            logging.warning("PyMuPDF (fitz) not found. PDF processing disabled. `pip install PyMuPDF`")
        if not DOCX_AVAILABLE:
            logging.warning("python-docx not found. DOCX processing disabled. `pip install python-docx`")
        if not LANGCHAIN_AVAILABLE:
            logging.warning("LangChain/Google AI not found. AI features disabled. `pip install langchain-google-genai`")

    def poll_log_queue(self):
        while True:
            try: record = self.log_queue.get(block=False)
            except queue.Empty: break
            else:
                self.log_text_widget.config(state='normal'); self.log_text_widget.insert(tk.END, record + '\n'); self.log_text_widget.config(state='disabled'); self.log_text_widget.yview(tk.END)
        self.root.after(100, self.poll_log_queue)
    
    def save_notes(self):
        if not self.final_notes_content: messagebox.showwarning("No Content", "Nothing to save."); return
        source_name = urlparse(self.url_var.get()).netloc.replace('.', '_') if self.input_mode_var.get() == "scrape" else "local_docs"
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
            api_key_file = self.api_key_file_var.get()
            model_name = self.model_name_var.get()
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            
            if not all([api_key_file, model_name, prompt]):
                messagebox.showerror("Input Error", "API Key File, Model, and Prompt must all be set.")
                return

            api_key = APIKeyManager.get_gemini_api_key(api_key_file)
            if not api_key:
                messagebox.showerror("API Key Error", f"Could not load a valid API key from '{api_key_file}'.")
                return

            source_data, source_name = {}, ""
            mode = self.input_mode_var.get()

            if mode == "scrape":
                logging.info("--- Starting in Web Scrape Mode ---")
                url = self.url_var.get()
                if not url:
                    messagebox.showerror("Input Error", "URL cannot be empty.")
                    return
                scraper_config = self.config.get('scraper', {})
                user_agent = scraper_config.get('user_agent', 'Mozilla/5.0')
                scraper = WebsiteScraper(url, user_agent=user_agent)
                source_data = scraper.scrape_single_page()
                source_name = urlparse(url).netloc
            else: # "upload"
                logging.info("--- Starting in Local Document Mode ---")
                file_paths = self.file_listbox.get(0, tk.END)
                if not file_paths:
                    messagebox.showerror("Input Error", "Please add at least one document.")
                    return
                loader = LocalDocumentLoader(file_paths)
                source_data = loader.load_and_extract_text()
                source_name = f"{len(file_paths)}_local_docs"

            if not source_data:
                logging.warning("No text content could be extracted. Halting process.")
                messagebox.showwarning("Extraction Failed", "Could not extract any text from the provided source.")
                return

            logging.info("✅ Text extracted successfully. Now preparing to send to AI.")
            
            # --- AI Processing ---
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=self.temperature_var.get(),
                max_output_tokens=self.max_tokens_var.get()
            )
            
            # Combine all extracted text into one context block
            full_context = "\n\n---\n\n".join(source_data.values())
            
            # Truncate context to avoid exceeding token limits (a safe-guard)
            # A more advanced implementation would use a proper text splitter.
            max_context_length = 30000 
            if len(full_context) > max_context_length:
                logging.warning(f"Context is very long ({len(full_context)} chars), truncating to {max_context_length} chars.")
                full_context = full_context[:max_context_length]

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "Here is the content I have gathered:\n\n{context}")
            ])
            
            output_parser = StrOutputParser()
            chain = chat_prompt | llm | output_parser
            
            logging.info(f"🧠 Sending request to model: {model_name}. Please wait...")
            self.final_notes_content = chain.invoke({"context": full_context})
            logging.info("✅ AI generation complete.")
            
            self.save_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Study guide has been generated! You can now save it.")

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            messagebox.showerror("Runtime Error", f"An error occurred:\n\n{e}")
        finally:
            self.start_button.config(state=tk.NORMAL)
            
    def _load_config_file(self, filepath="config.yml"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.warning(f"Could not load or parse '{filepath}': {e}")
            return {}
        
    def _load_prompt_file(self, filepath="prompt.md"):
        default_prompt = "You are an expert learning assistant. Your task is to create a comprehensive, well-structured study guide from the provided text. Use markdown for formatting. The guide should include:\n\n1.  **Key Concepts**: A bulleted list of the most important ideas, terms, and definitions.\n2.  **Detailed Summary**: A concise summary of the entire text.\n3.  **Potential Questions**: A list of questions that could be asked about this material to test understanding."
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"Prompt file not found at '{filepath}'. Using a default prompt.")
            return default_prompt

if __name__ == '__main__':
    root = tk.Tk()
    app = AdvancedScraperApp(root)
    root.mainloop()
