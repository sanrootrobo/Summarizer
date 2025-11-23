import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Tuple
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
import webbrowser
import tempfile

# --- Backend Standard Library Imports ---
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import yaml # You need to 'pip install pyyaml'

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
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


# --- Backend Third-Party Imports ---
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz # PyMuPDF, you need 'pip install PyMuPDF'
import docx # you need 'pip install python-docx'

# --- USER_AGENTS constant for the Playwright researcher ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
]

# --- START: Integrated UI Enhancement Classes ---

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

class DarkThemeManager:
    """Manages dark theme styling for the application"""

    @staticmethod
    def configure_dark_theme(root):
        """Configure dark theme for the entire application"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass # Fallback to default if clam is not available
        
        style.configure('TFrame', background=DARK_THEME['frame_bg'])
        style.configure('TLabel', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'])
        style.configure('TLabelFrame', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'])
        style.configure('TLabelFrame.Label', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'])
        style.configure('TButton', background=DARK_THEME['button_bg'], foreground=DARK_THEME['button_fg'], borderwidth=1, focuscolor='none')
        style.map('TButton', background=[('active', DARK_THEME['accent']), ('pressed', DARK_THEME['select_bg'])])
        style.configure('Accent.TButton', background=DARK_THEME['accent'], foreground='white', font=('Segoe UI', 9, 'bold'))
        style.map('Accent.TButton', background=[('active', '#106ebe'), ('pressed', '#005a9e')])
        style.configure('TEntry', fieldbackground=DARK_THEME['entry_bg'], background=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'], borderwidth=1, insertcolor=DARK_THEME['insert_bg'])
        style.configure('TNotebook', background=DARK_THEME['frame_bg'], borderwidth=0)
        style.configure('TNotebook.Tab', background=DARK_THEME['button_bg'], foreground=DARK_THEME['fg'], padding=[12, 8])
        style.map('TNotebook.Tab', background=[('selected', DARK_THEME['accent']), ('active', DARK_THEME['select_bg'])])
        style.configure('TCheckbutton', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'], focuscolor='none')
        style.configure('TRadiobutton', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'], focuscolor='none')
        root.configure(bg=DARK_THEME['bg'])

    @staticmethod
    def configure_text_widget(widget):
        """Configure text-like widgets with dark theme"""
        widget.configure(bg=DARK_THEME['text_bg'], fg=DARK_THEME['text_fg'], insertbackground=DARK_THEME['insert_bg'], selectbackground=DARK_THEME['select_bg'], selectforeground=DARK_THEME['select_fg'], relief='flat', borderwidth=1, highlightthickness=1, highlightcolor=DARK_THEME['accent'], highlightbackground=DARK_THEME['border'])
        widget.tag_configure('error', foreground=DARK_THEME['error'])
        widget.tag_configure('warning', foreground=DARK_THEME['warning'])
        widget.tag_configure('success', foreground=DARK_THEME['success'])
        widget.tag_configure('info', foreground=DARK_THEME['accent'])

class MarkdownPreviewWidget(ttk.Frame):
    """Enhanced markdown preview widget with dark theme styling"""

    def __init__(self, parent):
        super().__init__(parent)
        self.markdown_processor = None
        if MARKDOWN_AVAILABLE:
            self.markdown_processor = markdown.Markdown(extensions=['codehilite', 'tables', 'fenced_code', 'toc'])
        
        self.create_preview_widgets()
        self.current_content = ""

    def create_preview_widgets(self):
        """Create the preview interface"""
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(5, 10), padx=5)

        ttk.Button(toolbar, text="🔄 Refresh Preview", command=self.refresh_preview).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="🌐 Open in Browser", command=self.open_in_browser).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="💾 Export HTML", command=self.export_html).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Preview Mode:").pack(side=tk.RIGHT, padx=(10, 5))
        self.preview_mode = tk.StringVar(value="rendered")
        ttk.Radiobutton(toolbar, text="Rendered", variable=self.preview_mode, value="rendered", command=self.switch_preview_mode).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Radiobutton(toolbar, text="Source", variable=self.preview_mode, value="source", command=self.switch_preview_mode).pack(side=tk.RIGHT)
        
        self.create_preview_area()
        self.show_placeholder()

    def create_preview_area(self):
        """Create the main preview area"""
        self.preview_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=('Segoe UI', 10), state='disabled', height=20)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))
        DarkThemeManager.configure_text_widget(self.preview_text)
        self.configure_markdown_tags()

    def configure_markdown_tags(self):
        """Configure text tags for markdown-like rendering"""
        self.preview_text.tag_configure('h1', font=('Segoe UI', 16, 'bold'), foreground='#4CAF50', spacing1=10, spacing3=5)
        self.preview_text.tag_configure('h2', font=('Segoe UI', 14, 'bold'), foreground='#2196F3', spacing1=8, spacing3=4)
        self.preview_text.tag_configure('h3', font=('Segoe UI', 12, 'bold'), foreground='#FF9800', spacing1=6, spacing3=3)
        self.preview_text.tag_configure('bold', font=('Segoe UI', 10, 'bold'))
        self.preview_text.tag_configure('italic', font=('Segoe UI', 10, 'italic'))
        self.preview_text.tag_configure('code_inline', font=('Consolas', 9), background='#2d2d30', foreground='#dcdcaa')
        self.preview_text.tag_configure('code_block', font=('Consolas', 9), background='#1e1e1e', foreground='#d4d4d4', lmargin1=20, lmargin2=20, spacing1=5, spacing3=5, wrap='none')
        self.preview_text.tag_configure('list_item', lmargin1=20, lmargin2=40)
        self.preview_text.tag_configure('quote', lmargin1=20, lmargin2=20, background='#252526', foreground='#cccccc')
        self.preview_text.tag_configure('separator', spacing1=10, spacing3=10, overstrike=True)

    def show_placeholder(self):
        """Show placeholder content when no study guide is available"""
        placeholder = """# 📚 Study Guide Preview

Welcome to the Research & Study Guide Generator!

1.  **Choose your source**: Scrape a URL or upload local files.
2.  **Enable research** (optional) to find related content online.
3.  **Configure your AI model** and prompt.
4.  Click **"Start Generation"**.

Your generated study guide will appear here with rich formatting!
"""
        self.update_preview(placeholder)

    def update_preview(self, content):
        self.current_content = content
        self.refresh_preview()

    def refresh_preview(self):
        if not self.current_content: return
        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', tk.END)
        if self.preview_mode.get() == "source":
            self.preview_text.insert(tk.END, self.current_content)
        else:
            self.render_markdown_content(self.current_content)
        self.preview_text.config(state='disabled')

    def render_markdown_content(self, content):
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line: self.preview_text.insert(tk.END, '\n'); i += 1; continue
            
            if line.startswith('# '): self.preview_text.insert(tk.END, line[2:] + '\n', 'h1')
            elif line.startswith('## '): self.preview_text.insert(tk.END, line[3:] + '\n', 'h2')
            elif line.startswith('### '): self.preview_text.insert(tk.END, line[4:] + '\n', 'h3')
            elif line.startswith('```'):
                self.preview_text.insert(tk.END, '\n')
                i += 1
                code_content = [lines[i]] if i < len(lines) else []
                while i + 1 < len(lines) and not lines[i+1].startswith('```'):
                    i += 1
                    code_content.append(lines[i])
                i += 1
                self.preview_text.insert(tk.END, '\n'.join(code_content) + '\n', 'code_block')
                self.preview_text.insert(tk.END, '\n')
            elif line.startswith('---') or line.startswith('==='): self.preview_text.insert(tk.END, ' ' * 80 + '\n', 'separator')
            elif line.startswith(('* ','- ')) or re.match(r'^\d+\.\s', line): self.preview_text.insert(tk.END, '• ' + re.sub(r'^\* |^- |\d+\.\s', '', line) + '\n', 'list_item')
            elif line.startswith('> '): self.preview_text.insert(tk.END, line[2:] + '\n', 'quote')
            else: self.render_inline_formatting(line + '\n')
            i += 1

    def render_inline_formatting(self, text):
        parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'): self.preview_text.insert(tk.END, part[2:-2], 'bold')
            elif part.startswith('*') and part.endswith('*'): self.preview_text.insert(tk.END, part[1:-1], 'italic')
            elif part.startswith('`') and part.endswith('`'): self.preview_text.insert(tk.END, part[1:-1], 'code_inline')
            else: self.preview_text.insert(tk.END, part)

    def switch_preview_mode(self):
        self.refresh_preview()

    def open_in_browser(self):
        if not self.current_content: messagebox.showinfo("No Content", "No study guide to display."); return
        try:
            html_content = self.generate_html_content(self.current_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                temp_path = f.name
            webbrowser.open(f'file://{os.path.abspath(temp_path)}')
            self.master.after(5000, lambda: os.unlink(temp_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open in browser:\n{e}")

    def export_html(self):
        if not self.current_content: messagebox.showinfo("No Content", "No study guide to export."); return
        initial_filename = f"study_guide_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        filepath = filedialog.asksaveasfilename(initialfile=initial_filename, defaultextension=".html", filetypes=[("HTML files", "*.html")])
        if not filepath: return
        try:
            html_content = self.generate_html_content(self.current_content)
            with open(filepath, 'w', encoding='utf-8') as f: f.write(html_content)
            messagebox.showinfo("Success", f"HTML exported to:\n{filepath}")
        except Exception as e: messagebox.showerror("Error", f"Failed to export HTML:\n{e}")

    def generate_html_content(self, markdown_content):
        if MARKDOWN_AVAILABLE:
            html_body = self.markdown_processor.convert(markdown_content)
        else:
            html_body = "<pre>" + html2text.html2text(markdown_content) + "</pre>"
        
        return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Study Guide</title><style>
body{{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 2rem auto; padding: 2rem; background-color: #1e1e1e; color: #d4d4d4;}}
h1,h2,h3{{line-height:1.2;margin-top:1.5em;}} h1{{color:#34a853;}} h2{{color:#4285f4; border-bottom: 1px solid #444;}} h3{{color:#fbbc05;}}
code{{background-color:#2d2d30;color:#dcdcaa;padding:.2em .4em;margin:0;font-size:.85em;border-radius:3px;font-family:monospace;}}
pre{{background-color:#1e1e1e;border:1px solid #404040;border-radius:5px;padding:1em;overflow-x:auto;}} pre code{{background:none;padding:0;font-size:1em;}}
blockquote{{border-left:4px solid #34a853; margin:0; padding: 0.5em 1em; color: #aaa;}} hr{{border:none;height:1px;background-color:#404040;margin:2em 0;}}
</style></head><body>{html_body}</body></html>"""

# --- END: Integrated UI Enhancement Classes ---


# --- Backend Classes (Unchanged) ---
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

class YouTubeResearcher:
    def __init__(self):
        if not YT_DLP_AVAILABLE: raise ImportError("YouTube research requires 'yt-dlp'.")
        logging.info("🔎 Initialized YouTube Researcher.")
    def _extract_transcript_from_file(self, file_path: Path) -> str:
        if not file_path.exists(): return ""
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        lines = [line for line in content.splitlines() if not line.strip().startswith('WEBVTT') and '-->' not in line and line.strip()]
        return " ".join(dict.fromkeys(lines))
    def download_transcript(self, video_id: str) -> Optional[str]:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        temp_dir = Path(f"./temp_subs_{video_id}_{random.randint(1000, 9999)}")
        temp_dir.mkdir(exist_ok=True)
        try:
            with yt_dlp.YoutubeDL({'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en', 'en-US'], 'skip_download': True, 'outtmpl': str(temp_dir / '%(id)s'), 'quiet': True, 'logtostderr': False}) as ydl: ydl.download([video_url])
            subtitle_file = next(temp_dir.glob('*.vtt'), None)
            return self._extract_transcript_from_file(subtitle_file) if subtitle_file else None
        except Exception: return None
        finally:
            if temp_dir.exists(): shutil.rmtree(temp_dir)
    def get_transcripts_for_queries(self, queries: List[str], max_videos: int) -> Dict[str, str]:
        all_videos = {}
        for query in queries:
            try:
                result = subprocess.run(["yt-dlp", "--dump-json", "--default-search", f"ytsearch{max_videos}", query], capture_output=True, text=True, check=True, encoding='utf-8')
                for line in result.stdout.strip().split('\n'):
                    meta = json.loads(line)
                    if meta.get('id') and meta.get('title') and meta.get('subtitles'): all_videos[meta['id']] = meta['title']
            except Exception as e: logging.error(f"❌ yt-dlp search failed for '{query}': {e}")
            sleep(1.0)
        if not all_videos: return {}
        transcripts = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_id = {executor.submit(self.download_transcript, vid): vid for vid in all_videos}
            for future in future_to_id:
                video_id = future_to_id[future]
                try:
                    transcript = future.result()
                    if transcript: transcripts[f"https://www.youtube.com/watch?v={video_id}"] = f"Video Title: {all_videos[video_id]}\n\nTranscript:\n{transcript}"
                except Exception as e: logging.error(f"Error processing video {video_id}: {e}")
        return transcripts

class PlaywrightResearcher:
    def __init__(self):
        self.search_engine_url = "https://duckduckgo.com/"
        logging.info("🤖 Initialized Playwright-based researcher.")
    def search_and_extract_urls(self, queries: List[str], exclude_domain: str) -> List[str]:
        all_urls = set()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=random.choice(USER_AGENTS))
            page = context.new_page()
            try:
                for i, query in enumerate(queries):
                    logging.info(f"🤖 Playwright searching [{i+1}/{len(queries)}]: {query}")
                    try:
                        page.goto(self.search_engine_url, timeout=20000)
                        page.locator('input[name="q"]').fill(query); page.locator('input[name="q"]').press("Enter")
                        page.wait_for_selector("div#links", timeout=15000)
                        found = {link.get_attribute('href') for link in page.locator('h2 > a').all()}
                        all_urls.update({url for url in found if url and exclude_domain not in url and not any(url.endswith(e) for e in ['.pdf', '.zip'])})
                        sleep(random.uniform(2.0, 4.0))
                    except Exception as e: logging.warning(f"❌ Playwright error on query '{query}': {e}")
            finally: browser.close()
        return list(all_urls)

class WebsiteScraper:
    def __init__(self, base_url: str, max_pages: int, user_agent: str, timeout: int, delay: float):
        self.base_url = base_url; self.domain = urlparse(base_url).netloc; self.max_pages = max_pages
        self.session = requests.Session(); self.session.headers.update({'User-Agent': user_agent})
        self.timeout = timeout; self.delay = delay; self.scraped_content = {}; self.visited_urls = set()
    def _get_page_content(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''): return response.text
        except requests.RequestException as e: logging.warning(f"Request failed for {url}: {e}")
        return None
    def _parse_content_and_links(self, html: str, page_url: str) -> tuple[str, list[str]]:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup.select('script, style, nav, footer, header, aside, form, .ad'): element.decompose()
        main = soup.select_one('main, article, [role="main"], .main-content, .content') or soup.body
        text_maker = html2text.HTML2Text(); text_maker.body_width = 0; text_maker.ignore_links = True
        md = re.sub(r'\n{3,}', '\n\n', text_maker.handle(str(main)) if main else "").strip()
        links = {urljoin(page_url, a['href']).split('#')[0] for a in soup.find_all('a', href=True) if urlparse(urljoin(page_url, a['href'])).netloc == self.domain}
        return md, list(links)
    def crawl(self, additional_urls: Optional[List[str]] = None) -> dict[str, str]:
        urls_to_visit = {self.base_url}
        if additional_urls: urls_to_visit.update(additional_urls)
        while urls_to_visit:
            if 0 < self.max_pages <= len(self.visited_urls): logging.info(f"Reached limit of {self.max_pages} pages."); break
            url = urls_to_visit.pop()
            if url in self.visited_urls: continue
            self.visited_urls.add(url)
            log_prefix = "Researching" if additional_urls and url in additional_urls else "Scraping"
            logging.info(f"[{len(self.visited_urls)}/{self.max_pages or '∞'}] {log_prefix}: {url}")
            html = self._get_page_content(url)
            if html:
                text, new_links = self._parse_content_and_links(html, url)
                if text and len(text.strip()) > 150: self.scraped_content[url] = text
                if not (additional_urls and url in additional_urls): urls_to_visit.update(new_links)
            sleep(self.delay)
        return self.scraped_content

class EnhancedNoteGenerator:
    def __init__(self, api_key: str, llm_config: dict, prompt_template_string: str):
        self.llm = ChatGoogleGenerativeAI(model=llm_config['model_name'], google_api_key=api_key, **llm_config['parameters'])
        self.prompt = ChatPromptTemplate.from_template(prompt_template_string)
        self.chain = self.prompt | self.llm | StrOutputParser()
    def generate_comprehensive_notes(self, source_data: dict[str, str], source_name: str) -> str:
        if not source_data: return "No content provided."
        logging.info(f"Generating notes from {len(source_data)} sources...")
        full_content = "".join(f"\n\n--- SOURCE: {name} ---\n{text}" for name, text in source_data.items())
        try: return self.chain.invoke({"content": full_content, "website_url": source_name})
        except Exception as e:
            logging.error(f"Error during note generation: {e}")
            return f"# Generation Error\nAn error occurred: `{e}`"

# --- GUI Application ---
class QueueHandler(logging.Handler):
    def __init__(self, log_queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record): self.log_queue.put(self.format(record))

class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Research & Study Guide Generator")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # --- Apply Dark Theme ---
        DarkThemeManager.configure_dark_theme(self.root)

        # --- TKINTER VARIABLES ---
        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=5)
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
        self.config = {}

        self.create_widgets()
        self.load_initial_settings()
        self.toggle_input_mode()
        self.toggle_research_panel()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        settings_pane = ttk.Frame(main_frame, width=400); settings_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # --- MODIFIED: Right pane is now a notebook for preview and logs ---
        content_pane = ttk.Notebook(main_frame); content_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.preview_widget = MarkdownPreviewWidget(content_pane)
        log_frame = ttk.Frame(content_pane) # Frame to hold the log widget
        
        content_pane.add(self.preview_widget, text="📖 Study Guide Preview")
        content_pane.add(log_frame, text="📋 Process Logs")
        
        # --- END MODIFICATION ---

        notebook = ttk.Notebook(settings_pane); notebook.pack(fill="both", expand=True)
        source_tab, self.research_tab, ai_tab, prompt_tab = ttk.Frame(notebook, padding=10), ttk.Frame(notebook, padding=10), ttk.Frame(notebook, padding=10), ttk.Frame(notebook, padding=10)
        
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
        
        master_frame = ttk.LabelFrame(self.research_tab, text="Master Control", padding=5); master_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(master_frame, text="Enable AI Research (Alpha)", variable=self.research_enabled_var, command=self.toggle_research_panel).pack(anchor=tk.W)
        self.web_research_panel = ttk.LabelFrame(self.research_tab, text="Web Research", padding=10); self.web_research_panel.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(self.web_research_panel, text="Enable Web Search", variable=self.web_research_enabled_var).pack(anchor=tk.W)
        self.g_api_radio = ttk.Radiobutton(self.web_research_panel, text="Google API / DDG (Fast)", variable=self.web_research_method_var, value="google_api"); self.g_api_radio.pack(anchor=tk.W, padx=10)
        self.playwright_radio = ttk.Radiobutton(self.web_research_panel, text="Playwright (Slow, Robust)", variable=self.web_research_method_var, value="playwright"); self.playwright_radio.pack(anchor=tk.W, padx=10)
        ttk.Label(self.web_research_panel, text="Research Pages to Scrape:").pack(anchor=tk.W, pady=(5,0))
        ttk.Spinbox(self.web_research_panel, from_=1, to=50, textvariable=self.research_pages_var).pack(fill=tk.X, padx=10)
        self.yt_research_panel = ttk.LabelFrame(self.research_tab, text="YouTube Research", padding=10); self.yt_research_panel.pack(fill=tk.X, pady=5)
        self.yt_checkbutton = ttk.Checkbutton(self.yt_research_panel, text="Enable Transcript Analysis", variable=self.yt_research_enabled_var); self.yt_checkbutton.pack(anchor=tk.W)
        ttk.Label(self.yt_research_panel, text="Videos to Check per Query:").pack(anchor=tk.W, pady=(5,0))
        ttk.Spinbox(self.yt_research_panel, from_=1, to=10, textvariable=self.yt_videos_per_query_var).pack(fill=tk.X, padx=10)
        
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

        ttk.Button(prompt_tab, text="Load Prompt from File...", command=self.load_prompt_from_file).pack(fill=tk.X, pady=(0,5))
        self.prompt_text = scrolledtext.ScrolledText(prompt_tab, wrap=tk.WORD, height=10); self.prompt_text.pack(fill=tk.BOTH, expand=True)
        DarkThemeManager.configure_text_widget(self.prompt_text) # Apply dark theme to prompt editor

        notebook.add(source_tab, text="Source")
        notebook.add(self.research_tab, text="Research")
        notebook.add(ai_tab, text="AI Model")
        notebook.add(prompt_tab, text="Prompt")

        self.start_button = ttk.Button(settings_pane, text="Start Generation", command=self.start_task_thread, style="Accent.TButton"); self.start_button.pack(fill=tk.X, pady=(10, 0), ipady=5)
        self.save_button = ttk.Button(settings_pane, text="Save Study Guide As...", command=self.save_notes, state=tk.DISABLED); self.save_button.pack(fill=tk.X, pady=5)
        
        self.log_text_widget = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD); self.log_text_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        DarkThemeManager.configure_text_widget(self.log_text_widget) # Apply dark theme to log widget

        self.log_queue = queue.Queue(); self.queue_handler = QueueHandler(self.log_queue)
        logging.getLogger().addHandler(self.queue_handler); logging.getLogger().setLevel(logging.INFO)
        self.root.after(100, self.poll_log_queue)

    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            try: widget.configure(state=state)
            except tk.TclError: pass
            if isinstance(widget, (ttk.Frame, ttk.LabelFrame)): self._set_child_widgets_state(widget, state)

    def toggle_input_mode(self):
        is_scrape = self.input_mode_var.get() == "scrape"
        self._set_child_widgets_state(self.scraper_frame, tk.NORMAL if is_scrape else tk.DISABLED)
        self._set_child_widgets_state(self.upload_frame, tk.DISABLED if is_scrape else tk.NORMAL)

    def add_files(self):
        files = filedialog.askopenfilenames(title="Select Documents", filetypes=[("All Supported", "*.pdf *.docx *.txt")])
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
        self.config = self._load_config_file()
        prompt_template = self._load_prompt_file()
        if self.config:
            logging.info("📝 Loading settings from config.yml...")
            api_settings, llm_settings = self.config.get('api', {}), self.config.get('llm', {})
            llm_params = llm_settings.get('parameters', {})
            self.api_key_file_var.set(api_settings.get('key_file', 'gemini_api.key'))
            self.model_name_var.set(llm_settings.get('model_name', 'gemini-1.5-flash'))
            self.temperature_var.set(llm_params.get('temperature', 0.5))
            self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
            google_search_settings = api_settings.get('google_search', {})
            self.google_api_key_file_var.set(google_search_settings.get('key_file', 'google_api.key'))
            self.google_cx_file_var.set(google_search_settings.get('cx_file', 'google_cx.key'))
            logging.info("✅ Default settings loaded.")
        if prompt_template:
            self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, prompt_template)
            logging.info("✅ Default prompt loaded from prompt.md.")
        if not PLAYWRIGHT_AVAILABLE:
            self.playwright_radio.config(state=tk.DISABLED)
            if self.web_research_method_var.get() == 'playwright': self.web_research_method_var.set('google_api')
        if not YT_DLP_AVAILABLE:
            self.yt_checkbutton.config(state=tk.DISABLED); self.yt_research_enabled_var.set(False)

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
        # Clear preview and logs before starting
        self.preview_widget.update_preview("")
        self.log_text_widget.config(state='normal'); self.log_text_widget.delete('1.0', tk.END); self.log_text_widget.config(state='disabled')
        threading.Thread(target=self.run_full_process, daemon=True).start()

    def run_full_process(self):
        try:
            api_key = APIKeyManager.get_gemini_api_key(self.api_key_file_var.get())
            if not api_key: messagebox.showerror("API Key Error", "The Gemini API key is missing or invalid."); return

            source_data, source_name, mode = {}, "", self.input_mode_var.get()
            scraper_config = self.config.get('scraper', {})
            
            if mode == "scrape":
                url = self.url_var.get();
                if not url: messagebox.showerror("Input Error", "URL cannot be empty."); return
                scraper = WebsiteScraper(url, self.limit_var.get(), random.choice(USER_AGENTS), 15, scraper_config.get('rate_limit_delay', 0.5))
                source_data = scraper.crawl()
                source_name = urlparse(url).netloc
            else:
                file_paths = self.file_listbox.get(0, tk.END)
                if not file_paths: messagebox.showerror("Input Error", "Please add at least one document."); return
                source_data = LocalDocumentLoader(file_paths).load_and_extract_text()
                source_name = f"{len(file_paths)} local documents"

            if not source_data: logging.warning("No text extracted from initial source. Halting."); return

            if self.research_enabled_var.get():
                query_generator = ResearchQueryGenerator(api_key)
                main_topic = query_generator.extract_main_topic(source_data)
                research_queries = query_generator.generate_research_queries(main_topic, 5)

                if self.web_research_enabled_var.get():
                    if self.web_research_method_var.get() == "playwright": researcher = PlaywrightResearcher()
                    else:
                        g_key, g_cx = APIKeyManager.get_google_search_credentials(self.google_api_key_file_var.get(), self.google_cx_file_var.get())
                        researcher = GoogleSearchResearcher(g_key, g_cx)
                    research_urls = researcher.search_and_extract_urls(research_queries, source_name if mode == "scrape" else "")
                    if research_urls:
                        urls_to_scrape = research_urls[:self.research_pages_var.get()]
                        research_scraper = WebsiteScraper("http://research.local", len(urls_to_scrape), random.choice(USER_AGENTS), 15, scraper_config.get('rate_limit_delay', 0.5))
                        source_data.update(research_scraper.crawl(additional_urls=urls_to_scrape))

                if self.yt_research_enabled_var.get():
                    source_data.update(YouTubeResearcher().get_transcripts_for_queries(research_queries, self.yt_videos_per_query_var.get()))

            llm_config = {'model_name': self.model_name_var.get(), 'parameters': {'temperature': self.temperature_var.get(), 'max_output_tokens': self.max_tokens_var.get()}}
            generator = EnhancedNoteGenerator(api_key, llm_config, self.prompt_text.get("1.0", tk.END).strip())
            self.final_notes_content = generator.generate_comprehensive_notes(source_data, source_name)
            
            # --- MODIFIED: Update the preview widget ---
            self.root.after(0, self.preview_widget.update_preview, self.final_notes_content)
            
            logging.info("--- Study Guide Generation Complete! ---")
            self.save_button.config(state=tk.NORMAL)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            messagebox.showerror("Runtime Error", f"An error occurred:\n\n{e}")
        finally:
            self.start_button.config(state=tk.NORMAL)
            
    def _load_config_file(self, filepath="config.yml"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.warning(f"Could not load '{filepath}': {e}"); return {}
    def _load_prompt_file(self, filepath="prompt.md"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return f.read()
        except FileNotFoundError: return ""

if __name__ == '__main__':
    root = tk.Tk()
    app = AdvancedScraperApp(root)
    root.mainloop()
