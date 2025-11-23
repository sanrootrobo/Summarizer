import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple, Set
import re
import argparse
from time import sleep
import json
from datetime import datetime
import html2text
import random
import shutil
import subprocess
import os
from collections import defaultdict, deque
import webbrowser
import tempfile
import sys

# --- Backend Standard Library Imports ---
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import yaml

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
    from markdown.extensions import codehilite, tables, fenced_code, toc
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Warning: markdown package not available. Install with: pip install markdown")

# --- Dark Theme Configuration ---
DARK_THEME = {
    'bg': '#1e1e1e',
    'fg': '#ffffff',
    'select_bg': '#264f78',
    'select_fg': '#ffffff',
    'insert_bg': '#ffffff',
    'frame_bg': '#2d2d2d',
    'button_bg': '#404040',
    'button_fg': '#ffffff',
    'entry_bg': '#3c3c3c',
    'entry_fg': '#ffffff',
    'text_bg': '#1e1e1e',
    'text_fg': '#d4d4d4',
    'scrollbar_bg': '#404040',
    'scrollbar_fg': '#686868',
    'accent': '#0078d4',
    'success': '#16c60c',
    'warning': '#ffcc02',
    'error': '#e74856',
    'border': '#404040'
}

# --- USER_AGENTS constant for the Playwright researcher ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

class DarkThemeManager:
    """Manages dark theme styling for the application"""
    
    @staticmethod
    def configure_dark_theme(root):
        """Configure dark theme for the entire application, building on native themes."""
        style = ttk.Style(root)
        
        # Set a base theme to get native-looking widgets
        try:
            if sys.platform == "win32":
                style.theme_use('vista')
            elif sys.platform == "darwin":
                style.theme_use('aqua')
            else:
                style.theme_use('clam')
        except tk.TclError:
            style.theme_use('clam') # Fallback
        
        # Configure ttk widget styles for dark mode
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
        style.configure('TScale', background=DARK_THEME['frame_bg'], troughcolor=DARK_THEME['button_bg'], borderwidth=0)
        style.configure('TSpinbox', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'])
        style.configure('TPanedwindow', background=DARK_THEME['frame_bg'])
        
        root.configure(bg=DARK_THEME['bg'])

    @staticmethod
    def configure_text_widget(widget):
        """Configure text-like widgets with dark theme"""
        widget.configure(bg=DARK_THEME['text_bg'], fg=DARK_THEME['text_fg'], insertbackground=DARK_THEME['insert_bg'], selectbackground=DARK_THEME['select_bg'], selectforeground=DARK_THEME['select_fg'], relief='flat', borderwidth=1, highlightthickness=1, highlightcolor=DARK_THEME['accent'], highlightbackground=DARK_THEME['border'])
        widget.tag_configure('error', foreground=DARK_THEME['error'], font=('Segoe UI', 9, 'bold'))
        widget.tag_configure('warning', foreground=DARK_THEME['warning'])
        widget.tag_configure('success', foreground=DARK_THEME['success'], font=('Segoe UI', 9, 'bold'))
        widget.tag_configure('info', foreground=DARK_THEME['accent'])
    
    @staticmethod
    def configure_listbox(widget):
        """Configure listbox with dark theme"""
        widget.configure(bg=DARK_THEME['text_bg'], fg=DARK_THEME['text_fg'], selectbackground=DARK_THEME['select_bg'], selectforeground=DARK_THEME['select_fg'], relief='flat', borderwidth=1, highlightthickness=1, highlightcolor=DARK_THEME['accent'], highlightbackground=DARK_THEME['border'])

class MarkdownPreviewWidget(ttk.Frame):
    """Enhanced markdown preview widget with dark theme styling"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.markdown_processor = None
        if MARKDOWN_AVAILABLE:
            self.markdown_processor = markdown.Markdown(extensions=['codehilite', 'tables', 'fenced_code', 'toc'])
        self.create_preview_widgets()
        self.current_content = ""

    def create_preview_widgets(self):
        """Create the preview interface"""
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 10), padx=5)
        ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh_preview).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="🌐 Open in Browser", command=self.open_in_browser).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="💾 Export HTML", command=self.export_html).pack(side=tk.LEFT)
        self.preview_mode = tk.StringVar(value="rendered")
        ttk.Radiobutton(toolbar, text="Source", variable=self.preview_mode, value="source", command=self.switch_preview_mode).pack(side=tk.RIGHT)
        ttk.Radiobutton(toolbar, text="Rendered", variable=self.preview_mode, value="rendered", command=self.switch_preview_mode).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Label(toolbar, text="Preview:").pack(side=tk.RIGHT, padx=(10, 5))
        
        self.create_preview_area()
        self.show_placeholder()

    def create_preview_area(self):
        """Create the main preview area"""
        self.preview_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=('Segoe UI', 10), state='disabled')
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        DarkThemeManager.configure_text_widget(self.preview_text)
        self.configure_markdown_tags()

    def configure_markdown_tags(self):
        """Configure text tags for markdown-like rendering"""
        self.preview_text.tag_configure('h1', font=('Segoe UI', 18, 'bold'), foreground='#4CAF50', spacing1=12, spacing3=6)
        self.preview_text.tag_configure('h2', font=('Segoe UI', 16, 'bold'), foreground='#2196F3', spacing1=10, spacing3=5)
        self.preview_text.tag_configure('h3', font=('Segoe UI', 13, 'bold'), foreground='#FF9800', spacing1=8, spacing3=4)
        self.preview_text.tag_configure('h4', font=('Segoe UI', 11, 'bold'), foreground='#9C27B0', spacing1=6, spacing3=3)
        self.preview_text.tag_configure('bold', font=('Segoe UI', 10, 'bold'))
        self.preview_text.tag_configure('italic', font=('Segoe UI', 10, 'italic'))
        self.preview_text.tag_configure('code_inline', font=('Consolas', 9), background='#2d2d30', foreground='#dcdcaa')
        self.preview_text.tag_configure('code_block', font=('Consolas', 9), background='#1e1e1e', foreground='#d4d4d4', lmargin1=20, lmargin2=20, spacing1=5, spacing3=5, wrap='none')
        self.preview_text.tag_configure('list_item', lmargin1=20, lmargin2=40)
        self.preview_text.tag_configure('quote', lmargin1=20, lmargin2=20, background='#252526', foreground='#cccccc', spacing1=2, spacing3=2)
        self.preview_text.tag_configure('link', foreground='#4CAF50', underline=True)
        self.preview_text.tag_configure('separator', spacing1=10, spacing3=10, overstrike=True)

    def show_placeholder(self):
        """Show placeholder content when no study guide is available"""
        placeholder = "# 📚 Study Guide Preview\n\nWelcome to the Enhanced Research & Study Guide Generator!\n\n## 🚀 Quick Start\n\n1.  **Choose your source**: Web scraping or local documents.\n2.  **Configure research**: Enable web and YouTube research (optional).\n3.  **Set up AI model**: Add your Gemini API key.\n4.  **Click \"Start Generation\"** to create your study guide.\n\nYour generated study guide will appear here with rich formatting!\n\n---\n\n*Ready when you are!*"
        self.update_preview(placeholder)

    def update_preview(self, content): self.current_content = content; self.refresh_preview()
    def refresh_preview(self):
        if not self.current_content: return
        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', tk.END)
        if self.preview_mode.get() == "source": self.preview_text.insert(tk.END, self.current_content)
        else: self.render_markdown_content(self.current_content)
        self.preview_text.config(state='disabled')

    def render_markdown_content(self, content):
        # This is a simplified renderer. A full implementation is complex.
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line: self.preview_text.insert(tk.END, '\n'); i += 1; continue
            if line.startswith('# '): self.preview_text.insert(tk.END, line[2:] + '\n', 'h1')
            elif line.startswith('## '): self.preview_text.insert(tk.END, line[3:] + '\n', 'h2')
            elif line.startswith('### '): self.preview_text.insert(tk.END, line[4:] + '\n', 'h3')
            elif line.startswith('#### '): self.preview_text.insert(tk.END, line[5:] + '\n', 'h4')
            elif line.startswith('```'):
                self.preview_text.insert(tk.END, '\n')
                i += 1
                code_content = []
                while i < len(lines) and not lines[i].startswith('```'): code_content.append(lines[i]); i += 1
                if code_content: self.preview_text.insert(tk.END, '\n'.join(code_content) + '\n', 'code_block')
                self.preview_text.insert(tk.END, '\n')
            elif line.startswith('---') or line.startswith('==='): self.preview_text.insert(tk.END, ' ' * 80 + '\n', 'separator')
            elif line.startswith(('* ', '- ')) or re.match(r'^\d+\.\s', line): self.preview_text.insert(tk.END, '• ' + re.sub(r'^\* |^- |\d+\.\s', '', line) + '\n', 'list_item')
            elif line.startswith('> '): self.preview_text.insert(tk.END, line[2:] + '\n', 'quote')
            else: self.render_inline_formatting(line + '\n')
            i += 1

    def render_inline_formatting(self, text):
        parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\[.*?\]\(.*?\))', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'): self.preview_text.insert(tk.END, part[2:-2], 'bold')
            elif part.startswith('*') and part.endswith('*'): self.preview_text.insert(tk.END, part[1:-1], 'italic')
            elif part.startswith('`') and part.endswith('`'): self.preview_text.insert(tk.END, part[1:-1], 'code_inline')
            elif part.startswith('[') and '](' in part and part.endswith(')'): self.preview_text.insert(tk.END, part[1:part.index('](')], 'link')
            else: self.preview_text.insert(tk.END, part)

    def switch_preview_mode(self): self.refresh_preview()
    def open_in_browser(self):
        if not self.current_content: messagebox.showinfo("No Content", "No study guide content to display."); return
        try:
            html_content = self.generate_html_content(self.current_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f: f.write(html_content); temp_path = f.name
            webbrowser.open(f'file://{os.path.abspath(temp_path)}')
            self.parent.after(5000, lambda: os.unlink(temp_path))
        except Exception as e: messagebox.showerror("Error", f"Failed to open in browser:\n{e}")

    def export_html(self):
        if not self.current_content: messagebox.showinfo("No Content", "No study guide content to export."); return
        initial_filename = f"study_guide_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        filepath = filedialog.asksaveasfilename(initialfile=initial_filename, defaultextension=".html", filetypes=[("HTML files", "*.html"), ("All files", "*.*")])
        if not filepath: return
        try:
            html_content = self.generate_html_content(self.current_content)
            with open(filepath, 'w', encoding='utf-8') as f: f.write(html_content)
            messagebox.showinfo("Success", f"HTML exported to:\n{filepath}")
        except Exception as e: messagebox.showerror("Error", f"Failed to export HTML:\n{e}")

    def generate_html_content(self, markdown_content):
        html_body = markdown.markdown(markdown_content, extensions=['fenced_code', 'tables', 'sane_lists']) if MARKDOWN_AVAILABLE else f"<pre>{markdown_content}</pre>"
        return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Study Guide</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;line-height:1.6;max-width:800px;margin:2rem auto;padding:2rem;background-color:#1e1e1e;color:#d4d4d4}}
h1,h2,h3{{line-height:1.2;margin-top:1.5em}}h1{{color:#34a853}}h2{{color:#4285f4;border-bottom:1px solid #444}}h3{{color:#fbbc05}}
code{{background-color:#2d2d30;color:#dcdcaa;padding:.2em .4em;margin:0;font-size:.85em;border-radius:3px;font-family:monospace}}
pre{{background-color:#1e1e1e;border:1px solid #404040;border-radius:5px;padding:1em;overflow-x:auto}}pre code{{background:0 0;padding:0;font-size:1em}}
blockquote{{border-left:4px solid #34a853;margin:0;padding:0.5em 1em;color:#aaa}}a{{color:#4285f4}}hr{{border:none;height:1px;background-color:#404040;margin:2em 0}}
</style></head><body>{html_body}</body></html>"""

# --- START: FIXED AND IMPLEMENTED BACKEND CLASSES ---
class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str) -> Optional[str]:
        if not Path(filepath).is_file(): logging.error(f"Gemini API key file not found at {filepath}"); return None
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f: key = f.read().strip()
            if not key: raise ValueError("Key file is empty.")
            logging.info(f"🔑 Gemini API key loaded successfully."); return key
        except Exception as e: logging.error(f"❌ Failed to load Gemini API key: {e}"); return None

class LocalDocumentLoader:
    def load(self, file_paths: List[str]) -> Dict[str, str]:
        content = {}
        for file_path in file_paths:
            path = Path(file_path)
            try:
                logging.info(f"📄 Reading local file: {path.name}")
                if path.suffix == '.pdf': text = self._read_pdf(path)
                elif path.suffix == '.docx': text = self._read_docx(path)
                elif path.suffix == '.txt': text = self._read_txt(path)
                else: logging.warning(f"⚠️ Unsupported file type: {path.name}"); continue
                content[path.name] = text
            except Exception as e: logging.error(f"❌ Error reading {path.name}: {e}")
        return content
    def _read_pdf(self, path: Path) -> str:
        with fitz.open(path) as doc: return "".join(page.get_text() for page in doc)
    def _read_docx(self, path: Path) -> str:
        return "\n".join(para.text for para in docx.Document(path).paragraphs)
    def _read_txt(self, path: Path) -> str:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()

class WebScraper:
    """Crawls and scrapes text content from a website up to a specified limit."""
    def crawl_and_scrape(self, start_url: str, limit: int) -> Dict[str, str]:
        if limit == 0: return {}
        
        scraped_content = {}
        queue = deque([start_url])
        visited: Set[str] = {start_url}
        start_domain = urlparse(start_url).netloc

        while queue and len(scraped_content) < limit:
            url = queue.popleft()
            logging.info(f"🌐 Scraping ({len(scraped_content)+1}/{limit}): {url}")
            try:
                response = requests.get(url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=10)
                response.raise_for_status()

                # Use html2text for clean text extraction
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                text = h.handle(response.text)
                scraped_content[url] = text
                
                # Find new links to crawl if we still need more pages
                if len(scraped_content) < limit:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        abs_link = urljoin(url, link['href'])
                        # Basic validation: stay on the same domain, check if not visited, and avoid anchors/files
                        if (urlparse(abs_link).netloc == start_domain and 
                            abs_link not in visited and 
                            urlparse(abs_link).scheme in ['http', 'https']):
                            visited.add(abs_link)
                            queue.append(abs_link)
                sleep(0.5) # Be respectful
            except requests.RequestException as e:
                logging.warning(f"⚠️ Could not scrape {url}: {e}")
        
        logging.info(f"✅ Scraped a total of {len(scraped_content)} pages.")
        return scraped_content

# ... (Other backend classes like EnhancedResearchQueryGenerator, PlaywrightResearcher, YouTubeResearcher, StudyGuideGenerator remain the same)
class EnhancedResearchQueryGenerator:
    def __init__(self, api_key: str, model_name: str):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template("Based on the following text, generate a list of {num_queries} concise and diverse Google search queries to find supplementary information. Return ONLY a numbered list of queries, one per line.\n\nText:\n\"{content}\"")
        self.chain = self.prompt | self.llm | StrOutputParser()
    def generate(self, content: str, num_queries: int) -> List[str]:
        try:
            logging.info("🧠 Generating research queries with AI...")
            content_snippet = (content[:3000] + '...') if len(content) > 3000 else content
            response = self.chain.invoke({"content": content_snippet, "num_queries": num_queries})
            queries = [q.strip() for q in re.findall(r'^\d+\.\s*(.*)', response, re.MULTILINE)]
            if not queries: return [q for q in response.strip().split('\n') if q]
            logging.info(f"✅ Generated {len(queries)} research queries."); return queries
        except Exception as e: logging.error(f"❌ Failed to generate research queries: {e}"); return []
class PlaywrightResearcher:
    def search_and_scrape(self, query: str, num_pages: int) -> Dict[str, str]:
        content = {}; logging.info(f"🎭 Searching with Playwright for: '{query}'")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=random.choice(USER_AGENTS))
                page.goto(f"https://duckduckgo.com/?q={query}&ia=web", timeout=30000)
                page.wait_for_selector('a[data-testid="result-title-a"]', timeout=20000)
                links = page.eval_on_selector_all('a[data-testid="result-title-a"]', 'els=>els.map(e=>e.href)')
                for i, link in enumerate(links[:num_pages]):
                    try:
                        logging.info(f"  -> Scraping result {i+1}: {link}")
                        page.goto(link, timeout=20000, wait_until='domcontentloaded')
                        h = html2text.HTML2Text(); h.ignore_links = True; h.ignore_images = True
                        content[link] = h.handle(page.content())
                        sleep(random.uniform(1, 2))
                    except Exception as e: logging.warning(f"  -> Timeout/error scraping {link}: {e}")
                browser.close()
        except Exception as e: logging.error(f"❌ Playwright research failed for '{query}': {e}")
        return content
class YouTubeResearcher:
    def get_transcripts(self, query: str, num_videos: int) -> Dict[str, str]:
        logging.info(f"📺 Searching YouTube for: '{query}'")
        ydl_opts = {'format':'bestaudio/best','noplaylist':True,'quiet':True,'default_search':f"ytsearch{num_videos}",'writesubtitles':True,'writeautomaticsub':True,'subtitleslangs':['en'],'skip_download':True}
        transcripts = {}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(query, download=False)
                if 'entries' not in result: return {}
                for video in result['entries']:
                    title = video.get('title', 'N/A')
                    logging.info(f"  -> Analyzing video: '{title}'")
                    if video.get('automatic_captions', {}).get('en'):
                        sub_url = video['automatic_captions']['en'][0]['url']
                        response = requests.get(sub_url)
                        transcripts[video.get('webpage_url')] = " ".join(re.findall(r'>(.*?)<', response.text))
        except Exception as e: logging.error(f"❌ YouTube research failed for '{query}': {e}")
        return transcripts
class StudyGuideGenerator:
    def __init__(self, api_key: str, model_name: str, temp: float, max_tokens: int):
        self.llm = ChatGoogleGenerativeAI(model=model_name,google_api_key=api_key,temperature=temp,max_output_tokens=max_tokens)
        self.chain = ChatPromptTemplate.from_template("{prompt_template}") | self.llm | StrOutputParser()
    def generate(self, prompt_template: str, context: str, source_identifier: str) -> str:
        try:
            logging.info("🤖 Generating final study guide with AI...")
            # A more robust template injection
            final_prompt = prompt_template.format(content=context, website_url=source_identifier)
            response = self.chain.invoke({"prompt_template": final_prompt})
            logging.info("✅ Study guide generated successfully."); return response
        except Exception as e:
            logging.error(f"❌ Failed to generate study guide: {e}", exc_info=True)
            return f"Error: Could not generate the study guide. Please check the logs.\n\nDetails: {e}"

# --- END: FIXED AND IMPLEMENTED BACKEND CLASSES ---

class QueueHandler(logging.Handler):
    def __init__(self, log_queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record): self.log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] {self.format(record)}")

class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Research & Study Guide Generator")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)
        DarkThemeManager.configure_dark_theme(self.root)

        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=1) # Default to 1 (single page)
        self.research_enabled_var = tk.BooleanVar(value=False)
        self.web_research_enabled_var = tk.BooleanVar(value=True)
        self.yt_research_enabled_var = tk.BooleanVar(value=False)
        self.research_pages_var = tk.IntVar(value=3)
        self.research_queries_var = tk.IntVar(value=4)
        self.yt_videos_per_query_var = tk.IntVar(value=2)
        self.api_key_file_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        self.final_notes_content = ""
        self.config = {}
        self.is_processing = False

        self.create_widgets()
        self.load_initial_settings()
        self.toggle_input_mode()
        self.toggle_research_panel()
        self.setup_logging()

    def create_widgets(self):
        """Creates the main UI layout with a two-panel design."""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Panel: Settings ---
        settings_frame = ttk.Frame(main_paned, width=420)
        main_paned.add(settings_frame, weight=0)

        settings_notebook = ttk.Notebook(settings_frame)
        settings_notebook.pack(fill="both", expand=True, pady=(0, 10))
        
        source_tab, research_tab, ai_tab, prompt_tab = (ttk.Frame(settings_notebook, padding=15) for _ in range(4))
        self.create_source_tab(source_tab)
        self.create_research_tab(research_tab)
        self.create_ai_tab(ai_tab)
        self.create_prompt_tab(prompt_tab)
        settings_notebook.add(source_tab, text="📄 Source")
        settings_notebook.add(research_tab, text="🔍 Research")
        settings_notebook.add(ai_tab, text="🤖 AI Model")
        settings_notebook.add(prompt_tab, text="📝 Prompt")

        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        self.start_button = ttk.Button(button_frame, text="🚀 Start Generation", command=self.start_task_thread, style="Accent.TButton")
        self.start_button.pack(fill=tk.X, ipady=5, pady=(0, 5))
        self.save_button = ttk.Button(button_frame, text="💾 Save Study Guide", command=self.save_notes, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, ipady=2)
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(button_frame, textvariable=self.progress_var, foreground=DARK_THEME['accent'], anchor='center').pack(pady=(5, 0), fill=tk.X)

        # --- Right Panel: Content (Preview & Logs) ---
        content_notebook = ttk.Notebook(main_paned)
        main_paned.add(content_notebook, weight=1)

        self.markdown_preview = MarkdownPreviewWidget(content_notebook)
        self.log_text_widget = scrolledtext.ScrolledText(content_notebook, state='disabled', wrap=tk.WORD, font=("Consolas", 9))
        DarkThemeManager.configure_text_widget(self.log_text_widget)
        
        content_notebook.add(self.markdown_preview, text="📖 Study Guide Preview")
        content_notebook.add(self.log_text_widget, text="📋 Process Logs")

    def create_source_tab(self, parent):
        mode_frame = ttk.LabelFrame(parent, text="📥 Input Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(mode_frame, text="🌐 Web Scraper", variable=self.input_mode_var, value="scrape", command=self.toggle_input_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="📁 Local Documents", variable=self.input_mode_var, value="upload", command=self.toggle_input_mode).pack(anchor=tk.W)

        self.scraper_frame = ttk.LabelFrame(parent, text="🕷️ Web Scraper Settings", padding=10)
        self.scraper_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(self.scraper_frame, text="Target URL:").pack(fill=tk.X, anchor='w')
        ttk.Entry(self.scraper_frame, textvariable=self.url_var, font=("Segoe UI", 9)).pack(fill=tk.X, pady=(2, 8))
        ttk.Label(self.scraper_frame, text="Page Limit (1 = only target URL):").pack(fill=tk.X, anchor='w')
        ttk.Spinbox(self.scraper_frame, from_=1, to=1000, textvariable=self.limit_var, width=10).pack(fill=tk.X, pady=(2, 0))

        self.upload_frame = ttk.LabelFrame(parent, text="📂 Local Document Settings", padding=10)
        self.upload_frame.pack(fill=tk.X)
        btn_frame = ttk.Frame(self.upload_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="➕ Add Files", command=self.add_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(btn_frame, text="🗑️ Clear List", command=self.clear_files).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        listbox_frame = ttk.Frame(self.upload_frame); listbox_frame.pack(fill=tk.BOTH, expand=True)
        self.file_listbox = tk.Listbox(listbox_frame, height=5, font=("Segoe UI", 8))
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        DarkThemeManager.configure_listbox(self.file_listbox)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_research_tab(self, parent):
        ttk.Checkbutton(parent, text="🔬 Enable AI-Powered Research (Beta)", variable=self.research_enabled_var, command=self.toggle_research_panel).pack(anchor=tk.W, pady=(0,15))
        
        self.web_research_panel = ttk.LabelFrame(parent, text="🌐 Web Research", padding=10)
        self.web_research_panel.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(self.web_research_panel, text="Enable Web Search", variable=self.web_research_enabled_var).pack(anchor=tk.W)
        status_text = f"Method: Playwright ({'Available' if PLAYWRIGHT_AVAILABLE else 'Not Installed'})"
        status_color = DARK_THEME['success'] if PLAYWRIGHT_AVAILABLE else DARK_THEME['error']
        ttk.Label(self.web_research_panel, text=status_text, foreground=status_color).pack(anchor=tk.W, padx=20, pady=(2,5))
        ttk.Label(self.web_research_panel, text="Pages to Scrape per Query:").pack(anchor=tk.W)
        ttk.Spinbox(self.web_research_panel, from_=1, to=10, textvariable=self.research_pages_var, width=10).pack(fill=tk.X, pady=(2, 0))

        self.yt_research_panel = ttk.LabelFrame(parent, text="📺 YouTube Research", padding=10)
        self.yt_research_panel.pack(fill=tk.X)
        self.yt_checkbutton = ttk.Checkbutton(self.yt_research_panel, text="Enable Video Transcript Analysis", variable=self.yt_research_enabled_var)
        self.yt_checkbutton.pack(anchor=tk.W)
        status_text_yt = f"Method: yt-dlp ({'Available' if YT_DLP_AVAILABLE else 'Not Installed'})"
        status_color_yt = DARK_THEME['success'] if YT_DLP_AVAILABLE else DARK_THEME['error']
        ttk.Label(self.yt_research_panel, text=status_text_yt, foreground=status_color_yt).pack(anchor=tk.W, padx=20, pady=(2,5))
        ttk.Label(self.yt_research_panel, text="Videos to Analyze per Query:").pack(anchor=tk.W)
        ttk.Spinbox(self.yt_research_panel, from_=1, to=5, textvariable=self.yt_videos_per_query_var, width=10).pack(fill=tk.X, pady=(2, 0))

    def create_ai_tab(self, parent):
        api_frame = ttk.LabelFrame(parent, text="🔑 API Configuration", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(api_frame, text="Gemini API Key File:").pack(fill=tk.X, anchor='w')
        key_frame = ttk.Frame(api_frame)
        key_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Entry(key_frame, textvariable=self.api_key_file_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(key_frame, text="📁", width=3, command=self.browse_api_key).pack(side=tk.RIGHT, padx=(5, 0))

        model_frame = ttk.LabelFrame(parent, text="🤖 Model Settings", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(model_frame, text="Model Name:").pack(fill=tk.X, anchor='w')
        ttk.Entry(model_frame, textvariable=self.model_name_var, font=("Segoe UI", 9)).pack(fill=tk.X, pady=(2, 8))
        self.temp_label = ttk.Label(model_frame, text=f"Temperature: {self.temperature_var.get():.1f}")
        self.temp_label.pack(fill=tk.X, anchor='w')
        ttk.Scale(model_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var, command=self.update_temperature_label).pack(fill=tk.X, pady=(2, 8))
        ttk.Label(model_frame, text="Max Output Tokens:").pack(fill=tk.X, anchor='w')
        ttk.Spinbox(model_frame, from_=1024, to=32768, increment=1024, textvariable=self.max_tokens_var, width=10).pack(fill=tk.X, pady=(2, 0))
    
    def create_prompt_tab(self, parent):
        control_frame = ttk.Frame(parent); control_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(control_frame, text="Load Prompt", command=self.load_prompt_from_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(control_frame, text="Reset to Default", command=self.reset_prompt_to_default).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        self.prompt_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=("Consolas", 10))
        self.prompt_text.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        DarkThemeManager.configure_text_widget(self.prompt_text)
    
    def setup_logging(self):
        self.log_queue = queue.Queue()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        self.queue_handler = QueueHandler(self.log_queue); self.queue_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.queue_handler); logging.getLogger().setLevel(logging.INFO)
        self.root.after(100, self.poll_log_queue)

    def poll_log_queue(self):
        while True:
            try: record = self.log_queue.get(block=False)
            except queue.Empty: break
            else:
                self.log_text_widget.config(state='normal')
                tag = ''
                if "ERROR" in record or "❌" in record: tag = 'error'
                elif "WARNING" in record or "⚠️" in record: tag = 'warning'
                elif "✅" in record or "SUCCESS" in record or "🔑" in record: tag = 'success'
                elif "🔍" in record or "📺" in record or "🌐" in record or "🧠" in record or "🤖" in record: tag = 'info'
                self.log_text_widget.insert(tk.END, record + '\n', tag)
                self.log_text_widget.config(state='disabled')
                self.log_text_widget.yview(tk.END)
        self.root.after(100, self.poll_log_queue)

    def run_full_process(self):
        try:
            if not DEPENDENCIES_AVAILABLE: self.show_demo_content(); return

            self.progress_var.set("Validating inputs..."); logging.info("🚀 Starting Generation Process...")
            api_key = APIKeyManager.get_gemini_api_key(self.api_key_file_var.get())
            if not api_key: messagebox.showerror("API Key Error", f"Could not load Gemini API key."); return

            mode, source_data, source_identifier = self.input_mode_var.get(), {}, "N/A"
            
            self.progress_var.set("Collecting initial content...")
            if mode == "scrape":
                url = self.url_var.get().strip()
                if not url: messagebox.showerror("Input Error", "URL cannot be empty."); return
                if not url.startswith(('http://', 'https://')): url = 'https://' + url; self.url_var.set(url)
                source_data = WebScraper().crawl_and_scrape(url, self.limit_var.get())
                source_identifier = urlparse(url).netloc
            else:
                file_paths = list(self.file_listbox.get(0, tk.END))
                if not file_paths: messagebox.showerror("Input Error", "Please add at least one document."); return
                source_data = LocalDocumentLoader().load(file_paths)
                source_identifier = f"{len(file_paths)} local document(s)"

            if not source_data: messagebox.showwarning("No Content", "Could not extract text from the source(s)."); return
            
            initial_content = "\n\n---\n\n".join(source_data.values())
            all_content_list = list(source_data.values())

            if self.research_enabled_var.get():
                self.progress_var.set("Generating research queries...")
                queries = EnhancedResearchQueryGenerator(api_key, self.model_name_var.get()).generate(initial_content, self.research_queries_var.get())
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    if self.web_research_enabled_var.get() and PLAYWRIGHT_AVAILABLE:
                        for q in queries: futures.append(executor.submit(PlaywrightResearcher().search_and_scrape, q, self.research_pages_var.get()))
                    if self.yt_research_enabled_var.get() and YT_DLP_AVAILABLE:
                        for q in queries: futures.append(executor.submit(YouTubeResearcher().get_transcripts, q, self.yt_videos_per_query_var.get()))
                    for i, future in enumerate(as_completed(futures)):
                        self.progress_var.set(f"Researching... ({i+1}/{len(futures)})")
                        try: all_content_list.extend(future.result().values())
                        except Exception as e: logging.error(f"❌ Research task failed: {e}")

            self.progress_var.set("Generating study guide...")
            final_context = "\n\n--- NEW SOURCE ---\n\n".join(all_content_list)
            generator = StudyGuideGenerator(api_key, self.model_name_var.get(), self.temperature_var.get(), self.max_tokens_var.get())
            self.final_notes_content = generator.generate(self.prompt_text.get("1.0", tk.END), final_context, source_identifier)
            
            self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
            self.progress_var.set("Generation Complete!"); logging.info("🎉 Study Guide Generation Complete!")
            self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("Success!", "Study guide generated successfully!"))
        except Exception as e:
            logging.error(f"❌ Unexpected error in main process: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"))
            if not self.final_notes_content: self.root.after(0, lambda: self.progress_var.set("Ready"))
            else: self.root.after(0, lambda: self.progress_var.set("Complete - Ready to Save"))

    # --- Helper & UI Methods (mostly unchanged) ---
    def update_temperature_label(self, value): self.temp_label.configure(text=f"Temperature: {float(value):.1f}")
    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            if isinstance(widget, (ttk.Frame, ttk.LabelFrame)): self._set_child_widgets_state(widget, state)
            else:
                try: widget.configure(state=state)
                except tk.TclError: pass
    def toggle_input_mode(self):
        is_scrape = self.input_mode_var.get() == "scrape"
        self._set_child_widgets_state(self.scraper_frame, tk.NORMAL if is_scrape else tk.DISABLED)
        self._set_child_widgets_state(self.upload_frame, tk.DISABLED if is_scrape else tk.NORMAL)
    def add_files(self):
        files = filedialog.askopenfilenames(title="Select Documents", filetypes=[("Supported Files", "*.pdf *.docx *.txt"), ("All files", "*.*")])
        for f in files:
            if f not in self.file_listbox.get(0, tk.END): self.file_listbox.insert(tk.END, f)
    def clear_files(self): self.file_listbox.delete(0, tk.END)
    def browse_api_key(self):
        filepath = filedialog.askopenfilename(title="Select Gemini API Key File", filetypes=[("Key files", "*.key"), ("Text files", "*.txt")])
        if filepath: self.api_key_file_var.set(filepath)
    def load_prompt_from_file(self):
        filepath = filedialog.askopenfilename(title="Select Prompt File", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
                self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, content)
                logging.info(f"Loaded prompt from: {Path(filepath).name}")
            except Exception as e: messagebox.showerror("Error", f"Failed to load prompt: {e}")
    def toggle_research_panel(self):
        state = tk.NORMAL if self.research_enabled_var.get() else tk.DISABLED
        self._set_child_widgets_state(self.web_research_panel, state)
        self._set_child_widgets_state(self.yt_research_panel, state)
        if not PLAYWRIGHT_AVAILABLE: self.web_research_panel.winfo_children()[0].config(state=tk.DISABLED) # the checkbutton
        if not YT_DLP_AVAILABLE: self.yt_checkbutton.config(state=tk.DISABLED)
    def load_initial_settings(self):
        config = self._load_config_file()
        prompt_template = self._load_prompt_file("prompt.md")
        api_settings = config.get('api', {}); llm_settings = config.get('llm', {}); llm_params = llm_settings.get('parameters', {})
        self.api_key_file_var.set(api_settings.get('key_file', 'gemini_api.key'))
        self.model_name_var.set(llm_settings.get('model_name', 'gemini-1.5-flash-latest'))
        self.temperature_var.set(llm_params.get('temperature', 0.5))
        self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
        if prompt_template: self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, prompt_template)
        else: self.reset_prompt_to_default()
        self.update_temperature_label(self.temperature_var.get()); self._check_dependencies()
    def _check_dependencies(self):
        if not DEPENDENCIES_AVAILABLE: messagebox.showwarning("Missing Dependencies", "Core AI/Document libraries missing. App will run in DEMO mode.")
    def reset_prompt_to_default(self): self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, "You are an expert educational content creator. Generate a comprehensive study guide in Markdown based on the provided content.\n\nStructure your response with:\n1. **Executive Summary**\n2. **Main Topics** (use ## and ###)\n3. **Key Points & Definitions**\n4. **Practical Examples**\n5. **Further Learning Resources**\n\nContent to analyze:\n{content}\n\nSource: {website_url}")
    def save_notes(self):
        if not self.final_notes_content: messagebox.showwarning("No Content", "No study guide to save."); return
        source_name = urlparse(self.url_var.get()).netloc or "website" if self.input_mode_var.get() == "scrape" else "local_docs"
        initial_filename = f"study_guide_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        filepath = filedialog.asksaveasfilename(initialfile=initial_filename, defaultextension=".md", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if not filepath: logging.info("💾 Save cancelled by user."); return
        try:
            with open(filepath, "w", encoding="utf-8") as f: f.write(self.final_notes_content)
            logging.info(f"💾 Study guide saved: {filepath}"); messagebox.showinfo("Success", f"Study guide saved to:\n{filepath}")
        except IOError as e: logging.error(f"❌ Failed to save file: {e}"); messagebox.showerror("Save Error", f"Could not save file: {e}")
    def start_task_thread(self):
        if self.is_processing: messagebox.showwarning("Processing", "A task is already running."); return
        self.is_processing = True; self.start_button.config(state=tk.DISABLED, text="⏳ Processing..."); self.save_button.config(state=tk.DISABLED); self.final_notes_content = ""; self.progress_var.set("Initializing...")
        self.log_text_widget.config(state='normal'); self.log_text_widget.delete('1.0', tk.END); self.log_text_widget.config(state='disabled')
        threading.Thread(target=self.run_full_process, daemon=True).start()
    def show_demo_content(self):
        demo_content = f"# 🚀 Demo Mode\n\nThis is a demonstration because one or more critical dependencies (like LangChain, PyMuPDF, etc.) are not installed. Please install all required packages to enable full functionality.\n\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.root.after(0, lambda: self.markdown_preview.update_preview(demo_content))
        self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
        self.progress_var.set("Demo content loaded!")
    def _load_config_file(self, filepath="config.yml"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError): return {}
    def _load_prompt_file(self, filepath="prompt.md"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return f.read()
        except FileNotFoundError: return ""

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    root = tk.Tk()
    app = AdvancedScraperApp(root)
    def on_closing():
        if app.is_processing and messagebox.askokcancel("Quit", "A task is running. Quit anyway?"): root.destroy()
        elif not app.is_processing: root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
