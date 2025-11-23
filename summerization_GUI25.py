import os
import time
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple
import re
from time import sleep
import json
from datetime import datetime
import html2text
import random
import shutil
import webbrowser
import tempfile
import sys
import collections

# --- Backend Standard Library Imports ---
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import yaml

# --- Dependency Availability Flags ---
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Backend Third-Party Imports ---
try:
    from bs4 import BeautifulSoup
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import fitz  # PyMuPDF
    import docx
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Some critical dependencies are missing: {e}. The application will run in demo mode.")

# --- Markdown Preview Dependencies ---
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Warning: markdown package not available. Install with: pip install markdown")

# --- Dark Theme Configuration ---
DARK_THEME = {
    'bg': '#1e1e1e', 'fg': '#ffffff', 'select_bg': '#264f78', 'select_fg': '#ffffff', 'insert_bg': '#ffffff',
    'frame_bg': '#2d2d2d', 'button_bg': '#404040', 'button_fg': '#ffffff', 'entry_bg': '#3c3c3c',
    'entry_fg': '#ffffff', 'text_bg': '#1e1e1e', 'text_fg': '#d4d4d4', 'scrollbar_bg': '#404040',
    'scrollbar_fg': '#686868', 'accent': '#0078d4', 'success': '#16c60c', 'warning': '#ffcc02',
    'error': '#e74856', 'border': '#404040'
}

# --- USER_AGENTS constant ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

class DarkThemeManager:
    @staticmethod
    def configure_dark_theme(root):
        style = ttk.Style(root)
        try:
            if sys.platform == "win32": style.theme_use('vista')
            elif sys.platform == "darwin": style.theme_use('aqua')
            else: style.theme_use('clam')
        except tk.TclError: style.theme_use('clam')
        style.configure('TFrame', background=DARK_THEME['frame_bg'])
        style.configure('TLabel', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'])
        style.configure('TLabelFrame', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'])
        style.configure('TLabelFrame.Label', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'])
        style.configure('TButton', background=DARK_THEME['button_bg'], foreground=DARK_THEME['button_fg'], borderwidth=1, focuscolor='none')
        style.map('TButton', background=[('active', DARK_THEME['accent']), ('pressed', DARK_THEME['select_bg'])])
        style.configure('Accent.TButton', background=DARK_THEME['accent'], foreground='white', font=('Segoe UI', 10, 'bold'))
        style.map('Accent.TButton', background=[('active', '#106ebe'), ('pressed', '#005a9e')])
        style.configure('TEntry', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'], insertcolor=DARK_THEME['insert_bg'])
        style.configure('TNotebook', background=DARK_THEME['frame_bg'], borderwidth=0)
        style.configure('TNotebook.Tab', background=DARK_THEME['button_bg'], foreground=DARK_THEME['fg'], padding=[12, 8], font=('Segoe UI', 9))
        style.map('TNotebook.Tab', background=[('selected', DARK_THEME['accent']), ('active', DARK_THEME['select_bg'])])
        style.configure('TCheckbutton', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'], focuscolor='none')
        style.configure('TRadiobutton', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'], focuscolor='none')
        root.configure(bg=DARK_THEME['bg'])

    @staticmethod
    def configure_text_widget(widget):
        widget.configure(bg=DARK_THEME['text_bg'], fg=DARK_THEME['text_fg'], insertbackground=DARK_THEME['insert_bg'],
                         selectbackground=DARK_THEME['select_bg'], selectforeground=DARK_THEME['select_fg'],
                         relief='flat', borderwidth=1, highlightthickness=1, highlightcolor=DARK_THEME['accent'],
                         highlightbackground=DARK_THEME['border'])
        widget.tag_configure('error', foreground=DARK_THEME['error'], font=('Segoe UI', 9, 'bold'))
        widget.tag_configure('warning', foreground=DARK_THEME['warning'])
        widget.tag_configure('success', foreground=DARK_THEME['success'], font=('Segoe UI', 9, 'bold'))
        widget.tag_configure('info', foreground=DARK_THEME['accent'])

    @staticmethod
    def configure_listbox(widget):
        widget.configure(bg=DARK_THEME['text_bg'], fg=DARK_THEME['text_fg'],
                         selectbackground=DARK_THEME['select_bg'], selectforeground=DARK_THEME['select_fg'],
                         relief='flat', borderwidth=1, highlightthickness=1, highlightcolor=DARK_THEME['accent'],
                         highlightbackground=DARK_THEME['border'])

class MarkdownPreviewWidget(ttk.Frame):
    def __init__(self, parent, app_instance):
        super().__init__(parent)
        self.app = app_instance
        self.markdown_processor = markdown.Markdown(extensions=['fenced_code', 'tables']) if MARKDOWN_AVAILABLE else None
        self.create_preview_widgets()
        self.current_content = ""

    def create_preview_widgets(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 10), padx=5)
        ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh_preview).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="🌐 Open in Browser", command=self.app.prompt_and_open_in_browser).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="💾 Export HTML", command=self.export_html).pack(side=tk.LEFT)
        self.preview_mode = tk.StringVar(value="rendered")
        ttk.Radiobutton(toolbar, text="Source", variable=self.preview_mode, value="source", command=self.switch_preview_mode).pack(side=tk.RIGHT)
        ttk.Radiobutton(toolbar, text="Rendered", variable=self.preview_mode, value="rendered", command=self.switch_preview_mode).pack(side=tk.RIGHT, padx=(0, 5))
        self.create_preview_area()
        self.show_placeholder()

    def create_preview_area(self):
        self.preview_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=('Segoe UI', 10), state='disabled')
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        DarkThemeManager.configure_text_widget(self.preview_text)
        self.configure_markdown_tags()

    def configure_markdown_tags(self):
        self.preview_text.tag_configure('h1', font=('Segoe UI', 18, 'bold'), foreground='#4CAF50', spacing1=12, spacing3=6)
        self.preview_text.tag_configure('h2', font=('Segoe UI', 16, 'bold'), foreground='#2196F3', spacing1=10, spacing3=5)
        self.preview_text.tag_configure('h3', font=('Segoe UI', 13, 'bold'), foreground='#FF9800', spacing1=8, spacing3=4)
        self.preview_text.tag_configure('code_inline', font=('Consolas', 9), background='#2d2d30', foreground='#dcdcaa')
        self.preview_text.tag_configure('code_block', font=('Consolas', 9), background='#1e1e1e', lmargin1=20, lmargin2=20, wrap='none')
        self.preview_text.tag_configure('list_item', lmargin1=20, lmargin2=40)
        self.preview_text.tag_configure('quote', lmargin1=20, lmargin2=20, background='#252526')

    def show_placeholder(self):
        placeholder = "# 📚 Study Guide Preview\n\nWelcome!\n\n1.  **Choose your source**: Web, YouTube, or Local Files.\n2.  **Configure AI**: Set your API key.\n3.  **Click \"Start Generation\"**.\n\nYour formatted guide will appear here."
        self.update_preview(placeholder)

    def update_preview(self, content):
        self.current_content = content
        self.refresh_preview()

    def refresh_preview(self):
        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', tk.END)
        if self.preview_mode.get() == "source" or not MARKDOWN_AVAILABLE:
            self.preview_text.insert(tk.END, self.current_content)
        else:
            self.render_markdown_content(self.current_content)
        self.preview_text.config(state='disabled')

    def render_markdown_content(self, content):
        lines = content.split('\n')
        in_code_block = False
        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
                self.preview_text.insert(tk.END, line + '\n', 'code_block')
                continue
            if in_code_block:
                self.preview_text.insert(tk.END, line + '\n', 'code_block')
                continue

            if line.startswith('# '): self.preview_text.insert(tk.END, line[2:] + '\n', 'h1')
            elif line.startswith('## '): self.preview_text.insert(tk.END, line[3:] + '\n', 'h2')
            elif line.startswith('### '): self.preview_text.insert(tk.END, line[4:] + '\n', 'h3')
            elif line.startswith(('* ', '- ')) or re.match(r'^\d+\.\s', line): self.preview_text.insert(tk.END, '• ' + re.sub(r'^\* |^- |\d+\.\s', '', line) + '\n', 'list_item')
            elif line.startswith('> '): self.preview_text.insert(tk.END, line[2:] + '\n', 'quote')
            else: self.render_inline_formatting(line + '\n')

    def render_inline_formatting(self, text):
        parts = re.split(r'(`.*?`)', text)
        for part in parts:
            if part.startswith('`') and part.endswith('`'):
                self.preview_text.insert(tk.END, part[1:-1], 'code_inline')
            else:
                self.preview_text.insert(tk.END, part)

    def switch_preview_mode(self):
        self.refresh_preview()

    def open_in_browser(self):
        if not self.current_content:
            messagebox.showinfo("No Content", "No study guide content to display.")
            return
        try:
            html_content = self.generate_html_content(self.current_content)
            temp_file_path = Path(tempfile.gettempdir()) / f"study_guide_{random.randint(1000,9999)}.html"
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            webbrowser.open(temp_file_path.as_uri())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open in browser:\n{e}")

    def export_html(self):
        if not self.current_content:
            messagebox.showinfo("No Content", "No study guide content to export.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html")])
        if not filepath: return
        try:
            html_content = self.generate_html_content(self.current_content)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            messagebox.showinfo("Success", f"HTML exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export HTML:\n{e}")

    def generate_html_content(self, markdown_content):
        html_body = self.markdown_processor.convert(markdown_content) if self.markdown_processor else f"<pre>{markdown_content}</pre>"
        return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Study Guide</title><style>
        body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;line-height:1.6;max-width:800px;margin:2rem auto;padding:2rem;background-color:#1e1e1e;color:#d4d4d4}}
        h1,h2,h3{{color:#4285f4}} pre{{background-color:#2d2d30;padding:1em;border-radius:5px;overflow-x:auto}}
        code{{font-family:monospace;background-color:#262626;padding:0.2em 0.4em;border-radius:3px}}
        blockquote{{border-left:4px solid #34a853;margin:0;padding:0.5em 1em;color:#aaa}}
        </style></head><body>{html_body}</body></html>"""

class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str) -> Optional[str]:
        if not Path(filepath).is_file(): return None
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f: key = f.read().strip()
            return key if key else None
        except Exception: return None

    @staticmethod
    def get_google_search_credentials(key_path: str, cx_path: str) -> tuple[Optional[str], Optional[str]]:
        try:
            with open(Path(key_path), 'r') as f: api_key = f.read().strip()
            with open(Path(cx_path), 'r') as f: cx_id = f.read().strip()
            return (api_key, cx_id) if api_key and cx_id else (None, None)
        except Exception: return None, None

class LocalDocumentLoader:
    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def load_and_extract_text(self) -> dict[str, str]:
        content = {}
        for path_str in self.file_paths:
            p = Path(path_str)
            try:
                if p.suffix.lower() == ".pdf": content[p.name] = "".join(page.get_text() for page in fitz.open(p))
                elif p.suffix.lower() == ".docx": content[p.name] = "\n".join(para.text for para in docx.Document(p).paragraphs)
                elif p.suffix.lower() == ".txt": content[p.name] = p.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logging.error(f"Failed to process file '{p.name}': {e}")
        return content

class WebsiteScraper:
    def __init__(self, base_url: str, max_pages: int, deep_crawl: bool, crawl_depth: int):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.deep_crawl = deep_crawl
        self.crawl_depth = crawl_depth if crawl_depth > 0 else float('inf')
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        self.visited_urls = set()

    def _get_page_content(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text if 'text/html' in response.headers.get('Content-Type', '') else None
        except requests.RequestException as e:
            logging.warning(f"Request failed for {url}: {e}")
            return None

    def _parse_content_and_links(self, html: str, page_url: str) -> Tuple[str, List[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup.select('script, style, nav, footer, header'): element.decompose()
        main_content = soup.select_one('main, article, [role="main"]') or soup.body
        text = html2text.html2text(str(main_content))
        links = {urljoin(page_url, a['href']).split('#') for a in soup.find_all('a', href=True)
                 if urlparse(urljoin(page_url, a['href'])).netloc == self.domain}
        return text, list(links)

    def crawl(self) -> dict[str, str]:
        scraped_content = {}
        urls_to_visit = collections.deque([(self.base_url, 0)])
        while urls_to_visit and len(scraped_content) < self.max_pages:
            url, depth = urls_to_visit.popleft()
            if url in self.visited_urls: continue
            self.visited_urls.add(url)
            logging.info(f"Scraping [Depth:{depth}]: {url}")
            html = self._get_page_content(url)
            if not html: continue
            text, new_links = self._parse_content_and_links(html, url)
            scraped_content[url] = text
            if self.deep_crawl and depth < self.crawl_depth:
                for link in new_links:
                    if link not in self.visited_urls:
                        urls_to_visit.append((link, depth + 1))
            sleep(0.5)
        return scraped_content

class EnhancedResearchQueryGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        self.output_parser = StrOutputParser()

    def generate_queries(self, content: dict, num_queries: int) -> List[str]:
        context = "\n".join(list(content.values()))[:8000]
        prompt = ChatPromptTemplate.from_template(
            "Based on the following text, generate {num_queries} diverse and specific search queries to find tutorials, "
            "best practices, and common problems. Return as a JSON list.\n\nTEXT: {context}\n\nJSON:"
        )
        chain = prompt | self.llm | self.output_parser
        try:
            response = chain.invoke({"context": context, "num_queries": num_queries})
            queries = json.loads(response.strip())
            return queries if isinstance(queries, list) else []
        except (json.JSONDecodeError, TypeError):
            return ["main topic tutorial", "common errors in topic", "best practices for topic"]

class EnhancedGoogleSearchResearcher:
    def __init__(self, api_key: Optional[str] = None, cx_id: Optional[str] = None):
        self.api_key = api_key
        self.cx_id = cx_id
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})

    def search(self, queries: list[str], num_results: int) -> list[str]:
        all_urls = set()
        for query in queries:
            try:
                if self.api_key and self.cx_id:
                    params = {'key': self.api_key, 'cx': self.cx_id, 'q': query, 'num': num_results}
                    response = self.session.get("https://www.googleapis.com/customsearch/v1", params=params)
                    results = response.json().get('items', [])
                    all_urls.update(item['link'] for item in results)
                else: # Fallback to DuckDuckGo
                    params = {'q': query, 'format': 'json'}
                    response = self.session.get("https://api.duckduckgo.com/", params=params)
                    results = response.json().get('RelatedTopics', [])
                    all_urls.update(topic['FirstURL'] for topic in results if 'FirstURL' in topic)
                sleep(1)
            except Exception as e:
                logging.error(f"Search failed for '{query}': {e}")
        return list(all_urls)

class EnhancedNoteGenerator:
    def __init__(self, api_key: str, model_config: dict, prompt_template: str):
        self.llm = ChatGoogleGenerativeAI(model=model_config['name'], google_api_key=api_key, **model_config['params'])
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.output_parser = StrOutputParser()

    def generate(self, source_data: dict, source_name: str) -> str:
        content = "\n\n---\n\n".join(f"Source: {url}\n\n{text}" for url, text in source_data.items())
        chain = self.prompt | self.llm | self.output_parser
        try:
            notes = chain.invoke({"content": content, "source_name": source_name})
            return f"# Study Guide for: {source_name}\n\n{notes}"
        except Exception as e:
            return f"# Generation Error\n\nAn error occurred: {e}"

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record):
        self.log_queue.put(self.format(record))

class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Research & Study Guide Generator")
        self.root.geometry("1400x900")
        DarkThemeManager.configure_dark_theme(self.root)
        self.is_processing = False
        self.final_notes_content = ""
        self.create_widgets()
        self.setup_logging()
        self.load_initial_settings()

    def create_widgets(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        settings_frame = ttk.Frame(main_paned, width=420); main_paned.add(settings_frame, weight=0)
        settings_notebook = ttk.Notebook(settings_frame); settings_notebook.pack(fill="both", expand=True)

        tabs = {name: ttk.Frame(settings_notebook, padding=15) for name in ["Source", "Research", "AI", "Prompt", "Output"]}
        self.create_source_tab(tabs["Source"])
        self.create_research_tab(tabs["Research"])
        self.create_ai_tab(tabs["AI"])
        self.create_prompt_tab(tabs["Prompt"])
        self.create_output_tab(tabs["Output"])
        for name, tab in tabs.items(): settings_notebook.add(tab, text=name)

        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0), side="bottom")
        self.start_button = ttk.Button(button_frame, text="🚀 Start Generation", command=self.start_task_thread, style="Accent.TButton")
        self.start_button.pack(fill=tk.X, ipady=5, pady=(0, 5))
        self.save_button = ttk.Button(button_frame, text="💾 Save Study Guide", command=self.save_notes, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, ipady=2)
        self.progress_var = tk.StringVar(value="Ready"); ttk.Label(button_frame, textvariable=self.progress_var).pack(pady=5)

        content_notebook = ttk.Notebook(main_paned); main_paned.add(content_notebook, weight=1)
        self.markdown_preview = MarkdownPreviewWidget(content_notebook, self)
        self.log_text_widget = scrolledtext.ScrolledText(content_notebook, state='disabled', wrap=tk.WORD, font=("Consolas", 9))
        DarkThemeManager.configure_text_widget(self.log_text_widget)
        content_notebook.add(self.markdown_preview, text="📖 Study Guide")
        content_notebook.add(self.log_text_widget, text="📋 Logs")

    def create_source_tab(self, parent):
        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=10)
        self.deep_crawl_var = tk.BooleanVar(value=False)
        self.crawl_depth_var = tk.IntVar(value=2)
        self.multimodal_upload_var = tk.BooleanVar(value=False)

        mode_frame = ttk.LabelFrame(parent, text="Input Mode", padding=10); mode_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(mode_frame, text="Web Scraper", variable=self.input_mode_var, value="scrape", command=self.toggle_input_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="YouTube Video", variable=self.input_mode_var, value="youtube", command=self.toggle_input_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Local Files", variable=self.input_mode_var, value="upload", command=self.toggle_input_mode).pack(anchor=tk.W)

        self.url_input_frame = ttk.LabelFrame(parent, text="URL Input", padding=10); self.url_input_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(self.url_input_frame, textvariable=self.url_var).pack(fill=tk.X)
        self.scraper_options_frame = ttk.Frame(self.url_input_frame); self.scraper_options_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(self.scraper_options_frame, text="Deep Crawl", variable=self.deep_crawl_var).pack(side=tk.LEFT)
        ttk.Spinbox(self.scraper_options_frame, from_=1, to=100, textvariable=self.limit_var, width=5).pack(side=tk.RIGHT)
        ttk.Label(self.scraper_options_frame, text="Max Pages:").pack(side=tk.RIGHT)

        self.upload_frame = ttk.LabelFrame(parent, text="Local Files", padding=10); self.upload_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(self.upload_frame, text="Multimodal Analysis (Images/Video)", variable=self.multimodal_upload_var).pack(anchor=tk.W)
        btn_frame = ttk.Frame(self.upload_frame); btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self.clear_files).pack(side=tk.RIGHT)
        self.file_listbox = tk.Listbox(self.upload_frame, height=4); DarkThemeManager.configure_listbox(self.file_listbox)
        self.file_listbox.pack(fill=tk.X, expand=True)
        self.toggle_input_mode()

    def create_research_tab(self, parent):
        self.research_enabled_var = tk.BooleanVar(value=False)
        self.research_queries_var = tk.IntVar(value=5)
        self.research_results_var = tk.IntVar(value=3)
        ttk.Checkbutton(parent, text="Enable AI-Powered Research", variable=self.research_enabled_var).pack(anchor=tk.W)
        ttk.Label(parent, text="Number of search queries to generate:").pack(anchor=tk.W, pady=(5,0))
        ttk.Spinbox(parent, from_=3, to=10, textvariable=self.research_queries_var).pack(anchor=tk.W)
        ttk.Label(parent, text="Search results to use per query:").pack(anchor=tk.W, pady=(5,0))
        ttk.Spinbox(parent, from_=1, to=8, textvariable=self.research_results_var).pack(anchor=tk.W)

    def create_ai_tab(self, parent):
        self.api_key_file_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        ttk.Label(parent, text="Gemini API Key File:").pack(anchor=tk.W)
        key_frame = ttk.Frame(parent); key_frame.pack(fill=tk.X)
        ttk.Entry(key_frame, textvariable=self.api_key_file_var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(key_frame, text="Browse...", command=self.browse_api_key).pack(side=tk.RIGHT)
        ttk.Label(parent, text="Model Name:").pack(anchor=tk.W, pady=(10,0))
        ttk.Entry(parent, textvariable=self.model_name_var).pack(fill=tk.X)
        ttk.Label(parent, text="Temperature:").pack(anchor=tk.W, pady=(10,0))
        ttk.Scale(parent, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var).pack(fill=tk.X)
        ttk.Label(parent, text="Max Output Tokens:").pack(anchor=tk.W, pady=(10,0))
        ttk.Spinbox(parent, from_=1024, to=16384, increment=1024, textvariable=self.max_tokens_var).pack(anchor=tk.W)

    def create_prompt_tab(self, parent):
        self.prompt_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=10)
        DarkThemeManager.configure_text_widget(self.prompt_text)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)
        btn_frame = ttk.Frame(parent); btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Load Prompt", command=self.load_prompt_from_file).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Reset to Default", command=self.reset_prompt_to_default).pack(side=tk.RIGHT)

    def create_output_tab(self, parent):
        self.hugo_enabled_var = tk.BooleanVar(value=False)
        self.hugo_dir_var = tk.StringVar()
        self.hugo_content_dir_var = tk.StringVar(value="posts")
        hugo_frame = ttk.LabelFrame(parent, text="Hugo Integration", padding=10); hugo_frame.pack(fill=tk.X)
        ttk.Checkbutton(hugo_frame, text="Enable Hugo Integration", variable=self.hugo_enabled_var).pack(anchor=tk.W)
        ttk.Label(hugo_frame, text="Hugo Project Directory:").pack(anchor=tk.W)
        hugo_dir_frame = ttk.Frame(hugo_frame); hugo_dir_frame.pack(fill=tk.X)
        ttk.Entry(hugo_dir_frame, textvariable=self.hugo_dir_var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(hugo_dir_frame, text="Browse...", command=self.browse_hugo_dir).pack(side=tk.RIGHT)
        ttk.Label(hugo_frame, text="Content Subdirectory (e.g., 'posts'):").pack(anchor=tk.W)
        ttk.Entry(hugo_frame, textvariable=self.hugo_content_dir_var).pack(fill=tk.X)

    def setup_logging(self):
        log_queue = queue.Queue()
        handler = QueueHandler(log_queue)
        logging.basicConfig(level=logging.INFO, handlers=[handler], format='%(levelname)s: %(message)s')
        self.root.after(100, self.poll_log_queue, log_queue)

    def poll_log_queue(self, log_queue):
        while True:
            try:
                record = log_queue.get(block=False)
                self.log_text_widget.config(state='normal')
                self.log_text_widget.insert(tk.END, record + '\n')
                self.log_text_widget.config(state='disabled')
                self.log_text_widget.yview(tk.END)
            except queue.Empty:
                break
        self.root.after(100, self.poll_log_queue, log_queue)

    def start_task_thread(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "A task is already in progress.")
            return

        api_key = APIKeyManager.get_gemini_api_key(self.api_key_file_var.get())
        if not api_key:
            messagebox.showerror("API Key Error", "Please provide a valid Gemini API key file.")
            return

        self.is_processing = True
        self.start_button.config(state=tk.DISABLED, text="Processing...")
        self.save_button.config(state=tk.DISABLED)
        self.log_text_widget.config(state='normal'); self.log_text_widget.delete('1.0', tk.END); self.log_text_widget.config(state='disabled')

        mode = self.input_mode_var.get()
        target_func = None
        if mode == 'scrape' or mode == 'upload' and not self.multimodal_upload_var.get():
            target_func = self.run_text_based_process
        elif mode == 'youtube' or mode == 'upload' and self.multimodal_upload_var.get():
            target_func = self.run_multimodal_process
        
        if target_func:
            threading.Thread(target=target_func, daemon=True).start()
        else:
            messagebox.showerror("Error", "Invalid processing mode selected.")
            self._reset_ui_state()

    def run_text_based_process(self):
        try:
            # 1. Collect initial content
            self.progress_var.set("Collecting content...")
            mode = self.input_mode_var.get()
            source_data, source_name = {}, ""
            if mode == 'scrape':
                url = self.url_var.get()
                if not url: raise ValueError("URL cannot be empty.")
                source_name = urlparse(url).netloc
                scraper = WebsiteScraper(url, self.limit_var.get(), self.deep_crawl_var.get(), self.crawl_depth_var.get())
                source_data = scraper.crawl()
            elif mode == 'upload':
                files = self.file_listbox.get(0, tk.END)
                if not files: raise ValueError("No files selected.")
                source_name = f"{len(files)} local documents"
                loader = LocalDocumentLoader(files)
                source_data = loader.load_and_extract_text()
            
            if not source_data: raise ValueError("No content was extracted.")

            # 2. (Optional) Research
            if self.research_enabled_var.get():
                self.progress_var.set("Conducting research...")
                api_key = APIKeyManager.get_gemini_api_key(self.api_key_file_var.get())
                query_gen = EnhancedResearchQueryGenerator(api_key)
                queries = query_gen.generate_queries(source_data, self.research_queries_var.get())
                
                searcher = EnhancedGoogleSearchResearcher() # Using fallback for simplicity
                research_urls = searcher.search(queries, self.research_results_var.get())
                
                if research_urls:
                    research_scraper = WebsiteScraper("http://research.local", len(research_urls), False, 0)
                    research_content = research_scraper.crawl(additional_urls=research_urls)
                    source_data.update(research_content)

            # 3. Generate Notes
            self.progress_var.set("Generating study guide...")
            api_key = APIKeyManager.get_gemini_api_key(self.api_key_file_var.get())
            model_config = {'name': self.model_name_var.get(), 'params': {'temperature': self.temperature_var.get(), 'max_output_tokens': self.max_tokens_var.get()}}
            prompt = self.prompt_text.get("1.0", tk.END)
            generator = EnhancedNoteGenerator(api_key, model_config, prompt)
            self.final_notes_content = generator.generate(source_data, source_name)
            
            self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
            self.root.after(0, self._finalize_generation)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self._reset_ui_state()

    def run_multimodal_process(self):
        if not GEMINI_AVAILABLE:
            messagebox.showerror("Dependency Error", "The 'google-generativeai' package is required for multimodal analysis.")
            self._reset_ui_state()
            return
            
        try:
            self.progress_var.set("Configuring API...")
            api_key = APIKeyManager.get_gemini_api_key(self.api_key_file_var.get())
            genai.configure(api_key=api_key)

            prompt_parts = [self.prompt_text.get("1.0", tk.END)]
            mode = self.input_mode_var.get()
            
            temp_dir = Path(tempfile.mkdtemp())
            uploaded_file_refs = []

            if mode == 'youtube':
                url = self.url_var.get()
                if not url: raise ValueError("YouTube URL is required.")
                self.progress_var.set("Downloading video...")
                video_path = temp_dir / "video.mp4"
                subprocess.run(['yt-dlp', '-f', 'best[ext=mp4]', '-o', str(video_path), url], check=True)
                file_paths_to_process = [video_path]
            else: # Upload mode
                file_paths_to_process = self.file_listbox.get(0, tk.END)

            if not file_paths_to_process: raise ValueError("No files to process.")

            # Process all files
            for file_path in file_paths_to_process:
                self.progress_var.set(f"Uploading {Path(file_path).name}...")
                file_ref = genai.upload_file(path=file_path)
                uploaded_file_refs.append(file_ref)
                prompt_parts.append(file_ref)
            
            self.progress_var.set("Generating analysis...")
            model = genai.GenerativeModel(model_name=self.model_name_var.get())
            response = model.generate_content(prompt_parts)
            self.final_notes_content = response.text

            self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
            self.root.after(0, self._finalize_generation)

        except Exception as e:
            logging.error(f"Multimodal processing failed: {e}")
            messagebox.showerror("Multimodal Error", str(e))
        finally:
            # Cleanup
            if 'uploaded_file_refs' in locals():
                for ref in uploaded_file_refs: genai.delete_file(ref.name)
            if 'temp_dir' in locals() and temp_dir.exists(): shutil.rmtree(temp_dir)
            self._reset_ui_state()

    def _reset_ui_state(self):
        self.is_processing = False
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"))
        self.root.after(0, lambda: self.progress_var.set("Ready"))
        
    def _finalize_generation(self):
        self.progress_var.set("Generation Complete!")
        if self.hugo_enabled_var.get():
            self.handle_hugo_integration()
        else:
            self.save_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Study guide generated successfully! You can now save it or export it.")

    def save_notes(self):
        if not self.final_notes_content:
            messagebox.showwarning("No Content", "There is no study guide to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if not filepath: return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.final_notes_content)
            messagebox.showinfo("Success", f"Study guide saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the file: {e}")

    def handle_hugo_integration(self):
        self.progress_var.set("Creating Hugo post...")
        hugo_dir_str = self.hugo_dir_var.get().strip()
        if not hugo_dir_str:
            messagebox.showerror("Hugo Error", "Hugo project directory is not set.")
            self._reset_ui_state()
            return
            
        hugo_root = self._find_hugo_root(Path(hugo_dir_str))
        if not hugo_root:
            messagebox.showerror("Hugo Error", f"Could not verify a Hugo project at '{hugo_dir_str}'.")
            self._reset_ui_state()
            return

        if not shutil.which("hugo"):
            messagebox.showerror("Hugo Error", "Hugo command not found. Please ensure Hugo is installed and in your system's PATH.")
            self._reset_ui_state()
            return

        try:
            # Extract title from the first H1 header
            title_match = re.search(r'^#\s+(.+)$', self.final_notes_content, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else f"Study Guide - {datetime.now().strftime('%Y-%m-%d')}"
            
            filename = self.sanitize_filename(title)
            content_subdir = self.hugo_content_dir_var.get().strip() or "posts"
            post_path_arg = f"{content_subdir}/{filename}.md"
            
            # Use 'hugo new' to create the post with frontmatter
            subprocess.run(["hugo", "new", post_path_arg], cwd=hugo_root, check=True, capture_output=True)
            
            full_post_path = hugo_root / "content" / post_path_arg
            
            # Read the generated frontmatter and append the content
            with open(full_post_path, 'r+', encoding='utf-8') as f:
                content_with_frontmatter = f.read()
                f.seek(0)
                f.write(content_with_frontmatter + "\n" + self.final_notes_content)
            
            messagebox.showinfo("Hugo Success", f"Successfully created new post:\n{full_post_path}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Hugo Error", f"Hugo command failed:\n{e.stderr.decode()}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during Hugo integration: {e}")
        finally:
            self._reset_ui_state()

    def sanitize_filename(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', '-', text) # Replace spaces with hyphens
        text = re.sub(r'[^\w-]', '', text) # Remove non-alphanumeric chars except hyphens
        return text.strip('-')[:60] # Limit length

    def _find_hugo_root(self, start_path: Path) -> Optional[Path]:
        current_path = start_path.resolve()
        for _ in range(5): # Search up 5 levels
            if (current_path / "content").is_dir() and any((current_path / f).exists() for f in ["hugo.toml", "config.toml"]):
                return current_path
            if current_path.parent == current_path: break
            current_path = current_path.parent
        return None

    def load_initial_settings(self):
        try:
            with open("config.yml", "r") as f:
                config = yaml.safe_load(f)
            self.api_key_file_var.set(config.get('api', {}).get('key_file', ''))
            self.model_name_var.set(config.get('llm', {}).get('model_name', 'gemini-1.5-flash'))
            self.temperature_var.set(config.get('llm', {}).get('parameters', {}).get('temperature', 0.5))
            self.max_tokens_var.set(config.get('llm', {}).get('parameters', {}).get('max_output_tokens', 8192))
            self.hugo_dir_var.set(config.get('hugo',{}).get('directory',''))
        except (FileNotFoundError, yaml.YAMLError):
            self.reset_prompt_to_default() # Load defaults if no config
        
        if not self.prompt_text.get("1.0", tk.END).strip():
            self.reset_prompt_to_default()

    def load_prompt_from_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert("1.0", f.read())

    def reset_prompt_to_default(self):
        default_prompt = "You are a helpful AI assistant. Based on the provided content, create a detailed study guide in Markdown format. The guide should include a summary, key concepts with definitions, and practical examples.\n\nCONTENT:\n---\n{content}\n---"
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", default_prompt)

    def toggle_input_mode(self):
        mode = self.input_mode_var.get()
        is_url_mode = mode in ['scrape', 'youtube']
        is_upload_mode = mode == 'upload'
        
        for child in self.url_input_frame.winfo_children(): child.configure(state=tk.NORMAL if is_url_mode else tk.DISABLED)
        for child in self.upload_frame.winfo_children(): child.configure(state=tk.NORMAL if is_upload_mode else tk.DISABLED)
        
    def add_files(self):
        files = filedialog.askopenfilenames()
        for f in files: self.file_listbox.insert(tk.END, f)
    def clear_files(self): self.file_listbox.delete(0, tk.END)
    def browse_api_key(self):
        filepath = filedialog.askopenfilename()
        if filepath: self.api_key_file_var.set(filepath)
    def browse_hugo_dir(self):
        directory = filedialog.askdirectory()
        if directory: self.hugo_dir_var.set(directory)

    def prompt_and_open_in_browser(self):
        """Calls the Markdown preview widget's method to render and open the content in a browser."""
        if hasattr(self, 'markdown_preview'):
            self.markdown_preview.open_in_browser()
            
    def show_demo_content(self):
        """Displays a placeholder message when dependencies are missing."""
        demo_text = f"""
# 🚀 Demo Mode
This is a demonstration because one or more critical Python packages (like 'langchain', 'fitz', or 'docx') are not installed.
Please install the required dependencies to enable full functionality.
**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.final_notes_content = demo_text
        self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
        self.root.after(0, self._finalize_generation)


def main():
    root = tk.Tk()
    app = AdvancedScraperApp(root)
    def on_closing():
        if app.is_processing and not messagebox.askokcancel("Quit", "A task is running. Quit anyway?"):
            return
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
