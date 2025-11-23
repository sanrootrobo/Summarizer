import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
import re
from urllib.parse import urlparse
from datetime import datetime
import os

# --- Backend Standard Library Imports ---
import requests
from pathlib import Path
from urllib.parse import urljoin
from time import sleep
import html2text
import yaml

# --- Backend Third-Party Imports ---
# Make sure to install these: pip install beautifulsoup4 langchain-google-genai google-generativeai PyMuPDF python-docx markdown2
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for DOCX files
import markdown2 # For converting markdown to HTML

# --- Backend Classes (self-contained in this script) ---

class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str) -> str | None:
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10: raise ValueError("Invalid API key")
            logging.info(f"API key loaded successfully from {filepath}.")
            return api_key
        except Exception as e:
            logging.error(f"Error reading API key file '{filepath}': {e}")
            return None

class LocalDocumentLoader:
    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def _read_txt(self, filepath: Path) -> str:
        return filepath.read_text(encoding='utf-8', errors='ignore')

    def _read_pdf(self, filepath: Path) -> str:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _read_docx(self, filepath: Path) -> str:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])

    def load_and_extract_text(self) -> dict[str, str]:
        extracted_content = {}
        for file_path_str in self.file_paths:
            filepath = Path(file_path_str)
            filename = filepath.name
            logging.info(f"Processing local file: {filename}")
            try:
                if filename.lower().endswith(".pdf"):
                    extracted_content[filename] = self._read_pdf(filepath)
                elif filename.lower().endswith(".docx"):
                    extracted_content[filename] = self._read_docx(filepath)
                elif filename.lower().endswith(".txt"):
                    extracted_content[filename] = self._read_txt(filepath)
                else:
                    logging.warning(f"Unsupported file type skipped: {filename}")
            except Exception as e:
                logging.error(f"Failed to process file '{filename}': {e}")
        logging.info(f"Successfully processed {len(extracted_content)} local documents.")
        return extracted_content

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
            if 'text/html' in response.headers.get('Content-Type', ''): return response.text
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed for {url}: {e}")
        return None

    def _parse_content_and_links(self, html: str, page_url: str) -> tuple[str, list[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup.select('script, style, nav, footer, header, aside, form, .ad'): element.decompose()
        main_content_area = soup.select_one('main, article, [role="main"], .main-content, .content') or soup.body
        text_maker = html2text.HTML2Text(); text_maker.body_width = 0; text_maker.ignore_links = True
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
                if text and len(text.strip()) > 150: self.scraped_content[url] = text
                urls_to_visit.update(new_links)
            sleep(self.rate_limit_delay)
        logging.info(f"Crawling complete. Scraped {len(self.scraped_content)} valid pages.")
        return self.scraped_content

class EnhancedNoteGenerator:
    def __init__(self, api_key: str, llm_config: dict, prompt_template_string: str):
        self.llm = ChatGoogleGenerativeAI(model=llm_config['model_name'], google_api_key=api_key, **llm_config['parameters'])
        self.output_parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_template(prompt_template_string)

    def generate_comprehensive_notes(self, source_data: dict[str, str], source_name: str) -> str:
        if not source_data: return "No content was provided to generate notes."
        logging.info(f"Generating notes from {len(source_data)} sources...")
        full_content = ""
        for name, text in source_data.items():
            full_content += f"\n\n--- SOURCE: {name} ---\n{text}"
        chain = self.prompt | self.llm | self.output_parser
        try:
            notes = chain.invoke({"content": full_content, "website_url": source_name})
            return notes
        except Exception as e:
            logging.error(f"Error during note generation: {e}")
            return f"# Generation Error\n\nAn error occurred while communicating with the AI model:\n\n`{e}`"

# --- GUI Specific Classes ---

class QueueHandler(logging.Handler):
    def __init__(self, log_queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record): self.log_queue.put(self.format(record))

class ScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate Study Guide Generator")
        self.root.geometry("900x750")

        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar()
        self.api_key_file_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        self.final_notes_content = ""

        self.create_widgets()
        self.load_initial_settings()
        self.toggle_input_mode()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        settings_pane = ttk.Frame(main_frame); settings_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        log_pane = ttk.Frame(main_frame); log_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(settings_pane)
        notebook.pack(fill="both", expand=True)

        source_tab = ttk.Frame(notebook, padding="10")
        mode_frame = ttk.LabelFrame(source_tab, text="Select Input Mode", padding="10")
        mode_frame.pack(fill=tk.X)
        ttk.Radiobutton(mode_frame, text="Scrape from URL", variable=self.input_mode_var, value="scrape", command=self.toggle_input_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Upload Local Documents", variable=self.input_mode_var, value="upload", command=self.toggle_input_mode).pack(anchor=tk.W)

        self.scraper_frame = ttk.LabelFrame(source_tab, text="Web Scraper Settings", padding="10")
        self.scraper_frame.pack(fill=tk.X, pady=10)
        ttk.Label(self.scraper_frame, text="Target URL:").pack(fill=tk.X)
        self.url_entry = ttk.Entry(self.scraper_frame, textvariable=self.url_var)
        self.url_entry.pack(fill=tk.X)
        ttk.Label(self.scraper_frame, text="Page Limit (0=unlimited):").pack(fill=tk.X, pady=(5,0))
        self.limit_spinbox = ttk.Spinbox(self.scraper_frame, from_=0, to=1000, textvariable=self.limit_var)
        self.limit_spinbox.pack(fill=tk.X)

        self.upload_frame = ttk.LabelFrame(source_tab, text="Local Document Settings", padding="10")
        self.upload_frame.pack(fill=tk.X, pady=5)
        upload_buttons_frame = ttk.Frame(self.upload_frame)
        upload_buttons_frame.pack(fill=tk.X)
        self.add_files_button = ttk.Button(upload_buttons_frame, text="Add Files...", command=self.add_files)
        self.add_files_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        self.clear_files_button = ttk.Button(upload_buttons_frame, text="Clear List", command=self.clear_files)
        self.clear_files_button.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        self.file_listbox = tk.Listbox(self.upload_frame, height=4)
        self.file_listbox.pack(fill=tk.X, pady=5)

        ai_tab = ttk.Frame(notebook, padding="10")
        prompt_tab = ttk.Frame(notebook, padding="10")
        
        ttk.Label(ai_tab, text="API Key File:").pack(fill=tk.X, pady=(0, 2))
        api_frame = ttk.Frame(ai_tab); api_frame.pack(fill=tk.X)
        ttk.Entry(api_frame, textvariable=self.api_key_file_var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(api_frame, text="...", width=3, command=self.browse_api_key).pack(side=tk.RIGHT)
        ttk.Label(ai_tab, text="Model Name:").pack(fill=tk.X, pady=(10, 2))
        ttk.Entry(ai_tab, textvariable=self.model_name_var).pack(fill=tk.X)
        ttk.Label(ai_tab, text="Temperature:").pack(fill=tk.X, pady=(10, 2))
        ttk.Scale(ai_tab, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var).pack(fill=tk.X)
        ttk.Label(ai_tab, text="Max Output Tokens:").pack(fill=tk.X, pady=(10, 2))
        ttk.Spinbox(ai_tab, from_=1024, to=16384, increment=128, textvariable=self.max_tokens_var).pack(fill=tk.X)
        
        ttk.Button(prompt_tab, text="Load Prompt from File...", command=self.load_prompt_from_file).pack(fill=tk.X, pady=(0,5))
        self.prompt_text = scrolledtext.ScrolledText(prompt_tab, wrap=tk.WORD, height=10)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        notebook.add(source_tab, text="1. Data Source")
        notebook.add(ai_tab, text="2. AI Settings")
        notebook.add(prompt_tab, text="3. Prompt Editor")

        self.start_button = ttk.Button(settings_pane, text="Start Generation", command=self.start_task_thread); self.start_button.pack(fill=tk.X, pady=(10, 0))
        
        action_buttons_frame = ttk.Frame(settings_pane)
        action_buttons_frame.pack(fill=tk.X, pady=5)
        self.preview_button = ttk.Button(action_buttons_frame, text="Preview Guide", command=self.show_markdown_preview, state=tk.DISABLED)
        self.preview_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        self.save_button = ttk.Button(action_buttons_frame, text="Save Guide As...", command=self.save_notes, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, expand=True, fill=tk.X)

        log_label = ttk.LabelFrame(log_pane, text="Logs"); log_label.pack(fill=tk.BOTH, expand=True)
        self.log_text_widget = scrolledtext.ScrolledText(log_label, state='disabled', wrap=tk.WORD); self.log_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_queue = queue.Queue(); self.queue_handler = QueueHandler(self.log_queue)
        logging.getLogger().addHandler(self.queue_handler); logging.getLogger().setLevel(logging.INFO)
        self.root.after(100, self.poll_log_queue)

    def show_markdown_preview(self):
        if not self.final_notes_content:
            messagebox.showinfo("No Content", "Generate a study guide first to see a preview.")
            return

        preview_window = tk.Toplevel(self.root)
        preview_window.title("Study Guide Preview")
        preview_window.geometry("700x800")

        text_area = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD)
        text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # --- Basic Markdown to Tkinter Text Widget Styling ---
        text_area.tag_configure("h1", font=("Arial", 24, "bold"), spacing3=10)
        text_area.tag_configure("h2", font=("Arial", 20, "bold"), spacing3=8)
        text_area.tag_configure("h3", font=("Arial", 16, "bold"), spacing3=6)
        text_area.tag_configure("bold", font=("Arial", 10, "bold"))
        text_area.tag_configure("italic", font=("Arial", 10, "italic"))
        text_area.tag_configure("code", font=("Courier New", 10), background="#f0f0f0", wrap="none")
        
        # Convert Markdown to HTML
        html_content = markdown2.markdown(self.final_notes_content, extras=["fenced-code-blocks", "cuddled-lists", "tables"])
        
        # A simple HTML parser to apply tags
        soup = BeautifulSoup(html_content, "html.parser")
        
        for tag in soup.find_all(True):
            if tag.name in ["h1", "h2", "h3"]:
                text_area.insert(tk.END, tag.get_text() + "\n\n", tag.name)
            elif tag.name in ["p", "div"]:
                text_area.insert(tk.END, tag.get_text() + "\n")
            elif tag.name in ["b", "strong"]:
                text_area.insert(tk.END, tag.get_text(), "bold")
            elif tag.name in ["i", "em"]:
                text_area.insert(tk.END, tag.get_text(), "italic")
            elif tag.name == "li":
                text_area.insert(tk.END, "• " + tag.get_text() + "\n")
            elif tag.name == "pre":
                code_text = tag.get_text()
                text_area.insert(tk.END, "\n" + code_text + "\n", "code")
            elif tag.name not in ["ul", "ol"]: # Ignore list containers
                 text_area.insert(tk.END, tag.get_text())

        text_area.configure(state="disabled")

    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
            self._set_child_widgets_state(widget, state)

    def toggle_input_mode(self):
        mode = self.input_mode_var.get()
        if mode == "scrape":
            self._set_child_widgets_state(self.scraper_frame, tk.NORMAL)
            self._set_child_widgets_state(self.upload_frame, tk.DISABLED)
        else: # "upload"
            self._set_child_widgets_state(self.scraper_frame, tk.DISABLED)
            self._set_child_widgets_state(self.upload_frame, tk.NORMAL)

    def add_files(self):
        filetypes = [("All Supported", "*.pdf *.docx *.txt"), ("PDF", "*.pdf"), ("Word", "*.docx"), ("Text", "*.txt")]
        files = filedialog.askopenfilenames(title="Select Documents", filetypes=filetypes)
        for f in files:
            if f not in self.file_listbox.get(0, tk.END):
                self.file_listbox.insert(tk.END, f)
    
    def clear_files(self):
        self.file_listbox.delete(0, tk.END)

    def load_initial_settings(self):
        config = self._load_config_file(); prompt_template = self._load_prompt_file()
        api_cfg = config.get('api', {}); llm_cfg = config.get('llm', {})
        llm_params = llm_cfg.get('parameters', {})
        self.url_var.set("https://docs.python.org/3/"); self.limit_var.set(10)
        self.api_key_file_var.set(api_cfg.get('key_file', 'geminaikey'))
        self.model_name_var.set(llm_cfg.get('model_name', 'gemini-1.5-pro'))
        self.temperature_var.set(llm_params.get('temperature', 0.5))
        self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
        self.prompt_text.insert(tk.END, prompt_template)

    def _load_config_file(self, filepath="config.yml"):
        try:
            with open(filepath, 'r') as f: return yaml.safe_load(f)
        except: return {}

    def _load_prompt_file(self, filepath="prompt.md"):
        try:
            with open(filepath, 'r') as f: return f.read()
        except: return "Default prompt: Could not find prompt.md"

    def browse_api_key(self):
        filepath = filedialog.askopenfilename(title="Select API Key File");
        if filepath: self.api_key_file_var.set(filepath)
    
    def load_prompt_from_file(self):
        filepath = filedialog.askopenfilename(title="Select Prompt File", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if filepath:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, self._load_prompt_file(filepath))

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
        self.start_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.preview_button.config(state=tk.DISABLED)
        self.final_notes_content = ""
        threading.Thread(target=self.run_full_process, daemon=True).start()

    def run_full_process(self):
        try:
            api_key_file = self.api_key_file_var.get(); model_name = self.model_name_var.get(); prompt = self.prompt_text.get("1.0", tk.END).strip()
            if not all([api_key_file, model_name, prompt]):
                messagebox.showerror("Input Error", "API Key File, Model, and Prompt must be set."); return
            
            api_key = APIKeyManager.get_api_key(api_key_file)
            if not api_key: logging.error("API Key is invalid. Halting."); return

            source_data, source_name, mode = {}, "", self.input_mode_var.get()

            if mode == "scrape":
                logging.info("--- Starting in Web Scrape Mode ---")
                url = self.url_var.get()
                if not url: messagebox.showerror("Input Error", "URL cannot be empty."); return
                limit = self.limit_var.get(); delay = 0.5
                scraper = WebsiteScraper(url, limit, "Mozilla/5.0", 15, delay)
                source_data = scraper.crawl()
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

            llm_config = {'model_name': model_name, 'parameters': {'temperature': self.temperature_var.get(), 'max_output_tokens': self.max_tokens_var.get()}}
            generator = EnhancedNoteGenerator(api_key, llm_config, prompt)
            self.final_notes_content = generator.generate_comprehensive_notes(source_data, source_name)
            
            logging.info("--- Study Guide Generation Complete! ---")
            logging.info("Ready to save the file.")
            self.save_button.config(state=tk.NORMAL)
            self.preview_button.config(state=tk.NORMAL)

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            messagebox.showerror("Runtime Error", f"An error occurred:\n\n{e}")
        finally:
            self.start_button.config(state=tk.NORMAL)


if __name__ == '__main__':
    root = tk.Tk()
    app = ScraperApp(root)
    root.mainloop()
