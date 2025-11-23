import os
import time
import subprocess
import google.generativeai as genai
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
import webbrowser
import tempfile
import sys
import collections
from itertools import cycle

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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

class DarkThemeManager:
    """Manages dark theme styling for the application"""
    # ... (code is unchanged)
    @staticmethod
    def configure_dark_theme(root):
        style = ttk.Style(root)
        try:
            if sys.platform == "win32": style.theme_use('vista')
            elif sys.platform == "darwin": style.theme_use('aqua')
            else: style.theme_use('clam')
        except tk.TclError: style.theme_use('clam')
        style.configure('TFrame', background=DARK_THEME['frame_bg']); style.configure('TLabel', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg']); style.configure('TLabelFrame', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg']); style.configure('TLabelFrame.Label', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg']); style.configure('TButton', background=DARK_THEME['button_bg'], foreground=DARK_THEME['button_fg'], borderwidth=1, focuscolor='none'); style.map('TButton', background=[('active', DARK_THEME['accent']), ('pressed', DARK_THEME['select_bg'])]); style.configure('Accent.TButton', background=DARK_THEME['accent'], foreground='white', font=('Segoe UI', 10, 'bold')); style.map('Accent.TButton', background=[('active', '#106ebe'), ('pressed', '#005a9e')]); style.configure('TEntry', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'], insertcolor=DARK_THEME['insert_bg']); style.configure('TNotebook', background=DARK_THEME['frame_bg'], borderwidth=0); style.configure('TNotebook.Tab', background=DARK_THEME['button_bg'], foreground=DARK_THEME['fg'], padding=[12, 8], font=('Segoe UI', 9)); style.map('TNotebook.Tab', background=[('selected', DARK_THEME['accent']), ('active', DARK_THEME['select_bg'])]); style.configure('TCheckbutton', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'], focuscolor='none'); style.configure('TRadiobutton', background=DARK_THEME['frame_bg'], foreground=DARK_THEME['fg'], focuscolor='none'); style.configure('TScale', background=DARK_THEME['frame_bg'], troughcolor=DARK_THEME['button_bg'], borderwidth=0); style.configure('TSpinbox', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg']); style.configure('TPanedwindow', background=DARK_THEME['frame_bg']); root.configure(bg=DARK_THEME['bg'])
    @staticmethod
    def configure_text_widget(widget): widget.configure(bg=DARK_THEME['text_bg'], fg=DARK_THEME['text_fg'], insertbackground=DARK_THEME['insert_bg'], selectbackground=DARK_THEME['select_bg'], selectforeground=DARK_THEME['select_fg'], relief='flat', borderwidth=1, highlightthickness=1, highlightcolor=DARK_THEME['accent'], highlightbackground=DARK_THEME['border']); widget.tag_configure('error', foreground=DARK_THEME['error'], font=('Segoe UI', 9, 'bold')); widget.tag_configure('warning', foreground=DARK_THEME['warning']); widget.tag_configure('success', foreground=DARK_THEME['success'], font=('Segoe UI', 9, 'bold')); widget.tag_configure('info', foreground=DARK_THEME['accent'])
    @staticmethod
    def configure_listbox(widget): widget.configure(bg=DARK_THEME['text_bg'], fg=DARK_THEME['text_fg'], selectbackground=DARK_THEME['select_bg'], selectforeground=DARK_THEME['select_fg'], relief='flat', borderwidth=1, highlightthickness=1, highlightcolor=DARK_THEME['accent'], highlightbackground=DARK_THEME['border'])

class MarkdownPreviewWidget(ttk.Frame):
    # ... (code is unchanged)
    def __init__(self, parent, app_instance): super().__init__(parent); self.parent = parent; self.app = app_instance; self.markdown_processor = None; self.create_preview_widgets(); self.current_content = ""
    def create_preview_widgets(self): toolbar = ttk.Frame(self); toolbar.pack(fill=tk.X, pady=(0, 10), padx=5); ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh_preview).pack(side=tk.LEFT, padx=(0, 5)); ttk.Button(toolbar, text="🌐 Open in Browser", command=self.app.prompt_and_open_in_browser).pack(side=tk.LEFT, padx=(0, 5)); ttk.Button(toolbar, text="💾 Export HTML", command=self.export_html).pack(side=tk.LEFT); self.preview_mode = tk.StringVar(value="rendered"); ttk.Radiobutton(toolbar, text="Source", variable=self.preview_mode, value="source", command=self.switch_preview_mode).pack(side=tk.RIGHT); ttk.Radiobutton(toolbar, text="Rendered", variable=self.preview_mode, value="rendered", command=self.switch_preview_mode).pack(side=tk.RIGHT, padx=(0, 5)); ttk.Label(toolbar, text="Preview:").pack(side=tk.RIGHT, padx=(10, 5)); self.create_preview_area(); self.show_placeholder()
    def create_preview_area(self): self.preview_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=('Segoe UI', 10), state='disabled'); self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5)); DarkThemeManager.configure_text_widget(self.preview_text); self.configure_markdown_tags()
    def configure_markdown_tags(self): self.preview_text.tag_configure('h1', font=('Segoe UI', 18, 'bold'), foreground='#4CAF50', spacing1=12, spacing3=6); self.preview_text.tag_configure('h2', font=('Segoe UI', 16, 'bold'), foreground='#2196F3', spacing1=10, spacing3=5); self.preview_text.tag_configure('h3', font=('Segoe UI', 13, 'bold'), foreground='#FF9800', spacing1=8, spacing3=4); self.preview_text.tag_configure('h4', font=('Segoe UI', 11, 'bold'), foreground='#9C27B0', spacing1=6, spacing3=3); self.preview_text.tag_configure('bold', font=('Segoe UI', 10, 'bold')); self.preview_text.tag_configure('italic', font=('Segoe UI', 10, 'italic')); self.preview_text.tag_configure('code_inline', font=('Consolas', 9), background='#2d2d30', foreground='#dcdcaa'); self.preview_text.tag_configure('code_block', font=('Consolas', 9), background='#1e1e1e', foreground='#d4d4d4', lmargin1=20, lmargin2=20, spacing1=5, spacing3=5, wrap='none'); self.preview_text.tag_configure('list_item', lmargin1=20, lmargin2=40); self.preview_text.tag_configure('quote', lmargin1=20, lmargin2=20, background='#252526', foreground='#cccccc', spacing1=2, spacing3=2); self.preview_text.tag_configure('link', foreground='#4CAF50', underline=True); self.preview_text.tag_configure('separator', spacing1=10, spacing3=10, overstrike=True)
    def show_placeholder(self): placeholder = "# 📚 Study Guide Preview..."; self.update_preview(placeholder)
    def update_preview(self, content): self.current_content = content; self.refresh_preview()
    def refresh_preview(self):
        if not self.current_content: return
        self.preview_text.config(state='normal'); self.preview_text.delete('1.0', tk.END)
        if self.preview_mode.get() == "source": self.preview_text.insert(tk.END, self.current_content)
        else: self.render_markdown_content(self.current_content)
        self.preview_text.config(state='disabled')
    def render_markdown_content(self, content):
        lines = content.split('\n'); i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line: self.preview_text.insert(tk.END, '\n'); i += 1; continue
            if line.startswith('# '): self.preview_text.insert(tk.END, line[2:] + '\n', 'h1')
            elif line.startswith('## '): self.preview_text.insert(tk.END, line[3:] + '\n', 'h2')
            elif line.startswith('### '): self.preview_text.insert(tk.END, line[4:] + '\n', 'h3')
            elif line.startswith('#### '): self.preview_text.insert(tk.END, line[5:] + '\n', 'h4')
            elif line.startswith('```'):
                self.preview_text.insert(tk.END, '\n'); i += 1; code_content = []
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
    def open_in_browser(self, content_to_open=None):
        content = content_to_open if content_to_open is not None else self.current_content
        if not content: messagebox.showinfo("No Content", "No study guide content to display."); return
        try:
            html_content = self.generate_html_content(content); temp_file_path = Path(tempfile.gettempdir()) / f"study_guide_{random.randint(1000,9999)}.html"
            with open(temp_file_path, 'w', encoding='utf-8') as f: f.write(html_content)
            webbrowser.open(temp_file_path.as_uri()); self.parent.after(5000, lambda: os.unlink(temp_file_path) if os.path.exists(temp_file_path) else None)
        except Exception as e: messagebox.showerror("Error", f"Failed to open in browser:\n{e}")
    def export_html(self):
        if not self.current_content: messagebox.showinfo("No Content", "No study guide content to export."); return
        filepath = filedialog.asksaveasfilename(initialfile=f"study_guide_{datetime.now().strftime('%Y%m%d_%H%M')}.html", defaultextension=".html", filetypes=[("HTML files", "*.html"), ("All files", "*.*")])
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: f.write(self.generate_html_content(self.current_content))
            messagebox.showinfo("Success", f"HTML exported to:\n{filepath}")
        except Exception as e: messagebox.showerror("Error", f"Failed to export HTML:\n{e}")
    def generate_html_content(self, markdown_content):
        html_body = markdown.markdown(markdown_content, extensions=['fenced_code', 'tables', 'sane_lists']) if MARKDOWN_AVAILABLE else f"<pre>{markdown_content}</pre>"
        return f"""...HTML TEMPLATE..."""

class APIKeyManager:
    # ... (code is unchanged)
    @staticmethod
    def get_gemini_api_keys(filepath: str) -> List[str]:
        if not Path(filepath).is_file(): return []
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f: keys = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            if not keys: raise ValueError("Key file is empty or all lines are commented out.")
            return keys
        except Exception as e: logging.error(f"❌ Failed to load Gemini API keys: {e}"); return []
    @staticmethod
    def get_google_search_credentials(key_path: str, cx_path: str) -> tuple[str | None, str | None]:
        try:
            with open(Path(key_path), 'r') as f: api_key = f.read().strip()
            with open(Path(cx_path), 'r') as f: cx_id = f.read().strip()
            if not api_key or not cx_id: raise ValueError("Key or CX is empty")
            return api_key, cx_id
        except Exception as e: return None, None

class LocalDocumentLoader:
    # ... (code is unchanged)
    def __init__(self, file_paths: list[str]): self.file_paths = file_paths
    def _read_txt(self, p: Path) -> str: return p.read_text(encoding='utf-8', errors='ignore')
    def _read_pdf(self, p: Path) -> str:
        with fitz.open(p) as doc: return "".join(page.get_text() for page in doc)
    def _read_docx(self, p: Path) -> str: return "\n".join(para.text for para in docx.Document(p).paragraphs)
    def load_and_extract_text(self) -> dict[str, str]:
        content = {}
        for path_str in self.file_paths:
            p, name = Path(path_str), Path(path_str).name
            try:
                if name.lower().endswith(".pdf"): content[name] = self._read_pdf(p)
                elif name.lower().endswith(".docx"): content[name] = self._read_docx(p)
                elif name.lower().endswith(".txt"): content[name] = self._read_txt(p)
            except Exception as e: logging.error(f"Failed to process file '{name}': {e}")
        return content

class WebsiteScraper:
    # ... (code is unchanged)
    def __init__(self, base_url: str, max_pages: int, user_agent: str, request_timeout: int, rate_limit_delay: float, gemini_api_key: Optional[str] = None, deep_crawl: bool = False, crawl_depth: int = 0):
        # ... init logic ...
        pass
    def crawl(self, additional_urls: Optional[List[str]] = None) -> dict[str, str]:
        # ... crawl logic ...
        return {}

class EnhancedResearchQueryGenerator:
    # ... (code is unchanged)
    def __init__(self, api_key: str): self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3); self.output_parser = StrOutputParser()
    def extract_main_topic_and_subtopics(self, content: dict[str, str]) -> Dict[str, List[str]]:
        # ... logic ...
        return {"main_topic": "General Topic", "subtopics": []}
    def generate_diverse_research_queries(self, topic_data: Dict[str, List[str]], max_queries: int = 8) -> List[str]:
        # ... logic ...
        return []

class EnhancedGoogleSearchResearcher:
    # ... (code is unchanged)
    def __init__(self, api_key: str = None, cx_id: str = None): self.api_key, self.cx_id = api_key, cx_id; self.session = requests.Session(); self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    def search_and_extract_urls(self, queries: list[str], exclude_domain: str, max_results_per_query: int = 8) -> list[str]:
        # ... logic ...
        return []

class EnhancedPlaywrightResearcher:
    # ... (code is unchanged)
    def __init__(self): self.search_engines = [{"name": "DuckDuckGo", "url": "https://duckduckgo.com/", "input_selector": 'input[name="q"]'}, {"name": "Bing", "url": "https://www.bing.com/", "input_selector": 'input[name="q"]'}]
    def search_and_extract_urls(self, queries: List[str], exclude_domain: str) -> List[str]:
        # ... logic ...
        return []

class EnhancedYouTubeResearcher:
    # ... (code is unchanged, but review for context)
    def __init__(self):
        if not YT_DLP_AVAILABLE: raise ImportError("YouTube research requires 'yt-dlp'. Please install it.")
        self.temp_dirs = []; logging.info("🔎 Initialized Enhanced YouTube Researcher")
    def _clean_transcript_text(self, vtt_content: str) -> str:
        if not vtt_content: return ""
        lines = vtt_content.splitlines(); text_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('WEBVTT') or '-->' in line or line.startswith('NOTE') or re.match(r'^\d+$', line): continue
            line = re.sub(r'<[^>]+>', '', line); line = re.sub(r'&[a-zA-Z]+;', '', line); text_lines.append(line)
        full_text = ' '.join(text_lines); full_text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[SOUND\]', '', full_text, flags=re.IGNORECASE); return re.sub(r'\s+', ' ', full_text).strip()
    def _extract_transcript_from_file(self, file_path: Path) -> str:
        try:
            if not file_path.exists(): return ""
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
            return self._clean_transcript_text(content)
        except Exception as e: logging.warning(f"Failed to read transcript file {file_path}: {e}"); return ""
    def download_transcript(self, video_id: str, title: str = "") -> Optional[str]:
        video_url = f"https://www.youtube.com/watch?v={video_id}"; temp_dir = Path(tempfile.mkdtemp(prefix=f"subs_{video_id}_")); self.temp_dirs.append(temp_dir)
        try:
            ydl_opts = {'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en', 'en-US', 'en-GB'], 'skip_download': True, 'outtmpl': str(temp_dir / '%(id)s'), 'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([video_url])
            subtitle_files = list(temp_dir.glob('*.vtt'))
            if not subtitle_files: return None
            best_sub = sorted(subtitle_files, key=lambda p: ('.auto.' in p.name, '.live_chat.' in p.name))[0]
            transcript = self._extract_transcript_from_file(best_sub)
            return transcript if transcript and len(transcript) > 100 else None
        except Exception as e: logging.warning(f"Failed to download transcript for {video_id}: {e}"); return None
    def get_transcripts_for_playlist(self, video_entries: List[Dict]) -> Dict[str, str]:
        logging.info(f"📺 Found {len(video_entries)} videos. Downloading all transcripts in parallel...")
        transcripts = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_video = {executor.submit(self.download_transcript, entry['id'], entry.get('title')): entry for entry in video_entries}
            for i, future in enumerate(as_completed(future_to_video), 1):
                entry = future_to_video[future]; title = entry.get('title', f"video_{entry['id']}")
                try:
                    transcript = future.result()
                    if transcript: transcripts[title] = transcript; logging.info(f"  ✅ ({i}/{len(video_entries)}) Transcript downloaded: {title[:50]}...")
                    else: logging.warning(f"  ⚠️ ({i}/{len(video_entries)}) Failed transcript: {title[:50]}...")
                except Exception as e: logging.error(f"  ❌ ({i}/{len(video_entries)}) Error on {title}: {e}")
        self._cleanup(); return transcripts
    def _cleanup(self):
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists(): shutil.rmtree(temp_dir)
            except Exception as e: logging.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        self.temp_dirs.clear()

class EnhancedNoteGenerator:
    # ... (code is unchanged)
    def __init__(self, api_key: str, llm_config: dict, prompt_template_string: str): self.llm = ChatGoogleGenerativeAI(model=llm_config['model_name'], google_api_key=api_key, **llm_config['parameters']); self.output_parser = StrOutputParser(); self.prompt = ChatPromptTemplate.from_template(prompt_template_string)
    def _prepare_content_for_generation(self, source_data: dict[str, str]) -> str:
        organized_content = ""
        for source_name, content in source_data.items(): organized_content += f"\n--- SOURCE: {source_name[:80]} ---\n{content}\n"
        return organized_content
    def generate_comprehensive_notes(self, source_data: dict[str, str], source_name: str) -> str:
        if not source_data: return "No content provided."
        logging.info(f"📝 Generating notes for '{source_name[:60]}'")
        organized_content = self._prepare_content_for_generation(source_data)
        try:
            notes = (self.prompt | self.llm | self.output_parser).invoke({"content": organized_content, "website_url": source_name, "source_count": len(source_data)})
            if not notes.strip().startswith(("#", "##")): notes = f"# Study Guide for: {source_name}\n\n{notes}"
            return notes
        except Exception as e: logging.error(f"❌ Error during note generation for {source_name}: {e}"); return f"# Generation Error: {source_name}\n\n**Error:** `{e}`"

class QueueHandler(logging.Handler):
    # ... (code is unchanged)
    def __init__(self, log_queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record): self.log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] {self.format(record)}")

class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Research & Study Guide Generator v3.1 (Sanitized Dirs)")
        # ... (rest of __init__ is unchanged)
        self.root.geometry("1400x900"); self.root.minsize(1100, 700); DarkThemeManager.configure_dark_theme(self.root); self.input_mode_var = tk.StringVar(value="scrape"); self.url_var = tk.StringVar(); self.url_var.trace_add("write", self._update_url_status); self.limit_var = tk.IntVar(value=10); self.deep_crawl_var = tk.BooleanVar(value=False); self.crawl_depth_var = tk.IntVar(value=2); self.research_enabled_var = tk.BooleanVar(value=False); self.web_research_enabled_var = tk.BooleanVar(value=True); self.yt_research_enabled_var = tk.BooleanVar(value=True); self.web_research_method_var = tk.StringVar(value="google_api"); self.research_pages_var = tk.IntVar(value=5); self.research_queries_var = tk.IntVar(value=6); self.google_api_key_file_var = tk.StringVar(); self.google_cx_file_var = tk.StringVar(); self.yt_videos_per_query_var = tk.IntVar(value=3); self.api_key_file_var = tk.StringVar(); self.api_key_file_var.trace_add("write", self.update_api_key_status); self.loaded_api_keys = []; self.model_name_var = tk.StringVar(); self.temperature_var = tk.DoubleVar(); self.max_tokens_var = tk.IntVar(); self.multimodal_upload_var = tk.BooleanVar(value=False); self.auto_save_var = tk.BooleanVar(value=False); self.output_dir_var = tk.StringVar(); self.hugo_enabled_var = tk.BooleanVar(value=False); self.hugo_dir_var = tk.StringVar(); self.hugo_content_dir_var = tk.StringVar(value="posts"); self.final_notes_content = ""; self.last_generated_files = []; self.config = {}; self.is_processing = False; self.create_widgets(); self.load_initial_settings(); self.toggle_input_mode(); self.toggle_research_panel(); self.toggle_auto_save_panel(); self.setup_logging()

    def create_widgets(self):
        # ... (code is unchanged)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL); main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10); settings_frame = ttk.Frame(main_paned, width=420); main_paned.add(settings_frame, weight=0); settings_notebook = ttk.Notebook(settings_frame); settings_notebook.pack(fill="both", expand=True, pady=(0, 10)); source_tab, research_tab, ai_tab, prompt_tab, output_tab = (ttk.Frame(settings_notebook, padding=15) for _ in range(5)); self.create_source_tab(source_tab); self.create_research_tab(research_tab); self.create_ai_tab(ai_tab); self.create_prompt_tab(prompt_tab); self.create_output_tab(output_tab); settings_notebook.add(source_tab, text="📄 Source"); settings_notebook.add(research_tab, text="🔍 Research"); settings_notebook.add(ai_tab, text="🤖 AI Model"); settings_notebook.add(prompt_tab, text="📝 Prompt"); settings_notebook.add(output_tab, text="💾 Output"); button_frame = ttk.Frame(settings_frame); button_frame.pack(fill=tk.X, pady=(10, 0)); self.start_button = ttk.Button(button_frame, text="🚀 Start Generation", command=self.start_task_thread, style="Accent.TButton"); self.start_button.pack(fill=tk.X, ipady=5, pady=(0, 5)); self.save_button = ttk.Button(button_frame, text="💾 Save Study Guide", command=self.save_notes, state=tk.DISABLED); self.save_button.pack(fill=tk.X, ipady=2); self.progress_var = tk.StringVar(value="Ready"); ttk.Label(button_frame, textvariable=self.progress_var, foreground=DARK_THEME['accent'], anchor='center').pack(pady=(5, 0), fill=tk.X); content_notebook = ttk.Notebook(main_paned); main_paned.add(content_notebook, weight=1); self.markdown_preview = MarkdownPreviewWidget(content_notebook, self); self.log_text_widget = scrolledtext.ScrolledText(content_notebook, state='disabled', wrap=tk.WORD, font=("Consolas", 9)); DarkThemeManager.configure_text_widget(self.log_text_widget); content_notebook.add(self.markdown_preview, text="📖 Study Guide Preview"); content_notebook.add(self.log_text_widget, text="📋 Process Logs")

    def create_source_tab(self, parent):
        # ... (code is unchanged)
        mode_frame = ttk.LabelFrame(parent, text="📥 Input Mode", padding=10); mode_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Radiobutton(mode_frame, text="🌐 Web Scraper (Intelligent Crawler)", variable=self.input_mode_var, value="scrape", command=self.toggle_input_mode).pack(anchor=tk.W); ttk.Radiobutton(mode_frame, text="🎬 YouTube (Video or Playlist)", variable=self.input_mode_var, value="youtube_video", command=self.toggle_input_mode).pack(anchor=tk.W); ttk.Radiobutton(mode_frame, text="📁 Local Files (Text or Multimodal)", variable=self.input_mode_var, value="upload", command=self.toggle_input_mode).pack(anchor=tk.W); self.url_input_frame = ttk.LabelFrame(parent, text="🌐 URL Input", padding=10); self.url_input_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Label(self.url_input_frame, text="Target URL:").pack(fill=tk.X, anchor='w'); ttk.Entry(self.url_input_frame, textvariable=self.url_var, font=("Segoe UI", 9)).pack(fill=tk.X, pady=(2, 4)); self.url_status_label = ttk.Label(self.url_input_frame, text="", font=("Segoe UI", 8)); self.url_status_label.pack(anchor='w', pady=(0, 8)); self.scraper_options_frame = ttk.Frame(self.url_input_frame); self.scraper_options_frame.pack(fill=tk.X, pady=(0, 0)); ttk.Label(self.scraper_options_frame, text="Max total pages to scrape:").pack(fill=tk.X, anchor='w'); ttk.Spinbox(self.scraper_options_frame, from_=1, to=1000, textvariable=self.limit_var, width=10).pack(fill=tk.X, pady=(2, 10)); self.deep_crawl_check = ttk.Checkbutton(self.scraper_options_frame, text="Enable Deep Crawl", variable=self.deep_crawl_var, command=self.toggle_depth_setting); self.deep_crawl_check.pack(anchor=tk.W); self.depth_frame = ttk.Frame(self.scraper_options_frame); self.depth_frame.pack(fill=tk.X, padx=(20, 0), pady=(2, 0)); self.depth_label = ttk.Label(self.depth_frame, text="Crawl Depth (0=inf):"); self.depth_label.pack(side=tk.LEFT, padx=(0, 5)); self.depth_spinbox = ttk.Spinbox(self.depth_frame, from_=0, to=50, textvariable=self.crawl_depth_var, width=8); self.depth_spinbox.pack(side=tk.LEFT); self.upload_frame = ttk.LabelFrame(parent, text="📂 Local File Settings", padding=10); self.upload_frame.pack(fill=tk.X); self.multimodal_check = ttk.Checkbutton(self.upload_frame, text="Enable Multimodal Analysis", variable=self.multimodal_upload_var); self.multimodal_check.pack(anchor=tk.W, pady=(0, 10)); btn_frame = ttk.Frame(self.upload_frame); btn_frame.pack(fill=tk.X, pady=(0, 5)); ttk.Button(btn_frame, text="➕ Add Files", command=self.add_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5)); ttk.Button(btn_frame, text="🗑️ Clear List", command=self.clear_files).pack(side=tk.RIGHT, expand=True, fill=tk.X); listbox_frame = ttk.Frame(self.upload_frame); listbox_frame.pack(fill=tk.BOTH, expand=True); self.file_listbox = tk.Listbox(listbox_frame, height=5, font=("Segoe UI", 8)); scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.file_listbox.yview); self.file_listbox.configure(yscrollcommand=scrollbar.set); DarkThemeManager.configure_listbox(self.file_listbox); self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_research_tab(self, parent):
        # ... (code is unchanged)
        master_frame = ttk.LabelFrame(parent, text="🎛️ Master Control", padding=10); master_frame.pack(fill=tk.X, pady=(0, 10)); self.research_checkbutton = ttk.Checkbutton(master_frame, text="🔬 Enable AI-Powered Research (Beta)", variable=self.research_enabled_var, command=self.toggle_research_panel); self.research_checkbutton.pack(anchor=tk.W); settings_frame = ttk.LabelFrame(parent, text="⚙️ Research Settings", padding=10); settings_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Label(settings_frame, text="AI search queries to generate:").pack(anchor=tk.W); ttk.Spinbox(settings_frame, from_=3, to=15, textvariable=self.research_queries_var, width=10).pack(fill=tk.X, pady=(2, 10)); self.web_research_panel = ttk.LabelFrame(parent, text="🌐 Web Research", padding=10); self.web_research_panel.pack(fill=tk.X, pady=(0, 10)); ttk.Checkbutton(self.web_research_panel, text="Enable Web Search", variable=self.web_research_enabled_var).pack(anchor=tk.W, pady=(0, 5)); ttk.Label(self.web_research_panel, text="Search Method:").pack(anchor=tk.W); self.g_api_radio = ttk.Radiobutton(self.web_research_panel, text="🚀 Google API / DuckDuckGo", variable=self.web_research_method_var, value="google_api"); self.g_api_radio.pack(anchor=tk.W, padx=20); self.playwright_radio = ttk.Radiobutton(self.web_research_panel, text="🎭 Playwright Browser", variable=self.web_research_method_var, value="playwright"); self.playwright_radio.pack(anchor=tk.W, padx=20, pady=(0, 5)); ttk.Label(self.web_research_panel, text="Max external pages to scrape:").pack(anchor=tk.W); ttk.Spinbox(self.web_research_panel, from_=1, to=20, textvariable=self.research_pages_var, width=10).pack(fill=tk.X, pady=(2, 0)); self.yt_research_panel = ttk.LabelFrame(parent, text="📺 YouTube Research", padding=10); self.yt_research_panel.pack(fill=tk.X); self.yt_checkbutton = ttk.Checkbutton(self.yt_research_panel, text="Enable Video Transcript Analysis", variable=self.yt_research_enabled_var); self.yt_checkbutton.pack(anchor=tk.W, pady=(0, 5)); ttk.Label(self.yt_research_panel, text="Videos per search query:").pack(anchor=tk.W); ttk.Spinbox(self.yt_research_panel, from_=1, to=5, textvariable=self.yt_videos_per_query_var, width=10).pack(fill=tk.X, pady=(2, 0))

    def create_ai_tab(self, parent):
        # ... (code is unchanged)
        api_frame = ttk.LabelFrame(parent, text="🔑 API Configuration", padding=10); api_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Label(api_frame, text="Gemini API Key File (one key per line for parallel):").pack(fill=tk.X, anchor='w'); key_frame = ttk.Frame(api_frame); key_frame.pack(fill=tk.X, pady=(2, 0)); ttk.Entry(key_frame, textvariable=self.api_key_file_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, expand=True, fill=tk.X); ttk.Button(key_frame, text="📁", width=3, command=self.browse_api_key).pack(side=tk.RIGHT, padx=(5, 0)); self.api_key_status_label = ttk.Label(api_frame, text="No key file loaded.", font=("Segoe UI", 8), foreground=DARK_THEME['warning']); self.api_key_status_label.pack(anchor=tk.W, pady=(5,0)); model_frame = ttk.LabelFrame(parent, text="🤖 Model Settings", padding=10); model_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Label(model_frame, text="Model Name:").pack(fill=tk.X, anchor='w'); self.model_entry = ttk.Entry(model_frame, textvariable=self.model_name_var, font=("Segoe UI", 9)); self.model_entry.pack(fill=tk.X, pady=(2, 8)); self.temp_label = ttk.Label(model_frame, text=f"Temperature: {self.temperature_var.get():.1f}"); self.temp_label.pack(fill=tk.X, anchor='w'); ttk.Scale(model_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var, command=self.update_temperature_label).pack(fill=tk.X, pady=(2, 8)); ttk.Label(model_frame, text="Max Output Tokens:").pack(fill=tk.X, anchor='w'); ttk.Spinbox(model_frame, from_=1024, to=32768, increment=1024, textvariable=self.max_tokens_var, width=10).pack(fill=tk.X, pady=(2, 0))

    def create_prompt_tab(self, parent):
        # ... (code is unchanged)
        control_frame = ttk.Frame(parent); control_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Button(control_frame, text="Load Prompt", command=self.load_prompt_from_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5)); ttk.Button(control_frame, text="Reset to Default", command=self.reset_prompt_to_default).pack(side=tk.RIGHT, expand=True, fill=tk.X); self.prompt_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=("Consolas", 10)); self.prompt_text.pack(fill=tk.BOTH, expand=True, pady=(5,0)); DarkThemeManager.configure_text_widget(self.prompt_text)

    def create_output_tab(self, parent):
        # ... (code is unchanged)
        save_frame = ttk.LabelFrame(parent, text="📂 Automatic Markdown Saving", padding=10); save_frame.pack(fill=tk.X, pady=(0, 10)); self.auto_save_check = ttk.Checkbutton(save_frame, text="✅ Enable Automatic Saving", variable=self.auto_save_var, command=self.toggle_auto_save_panel); self.auto_save_check.pack(anchor=tk.W, pady=(0, 10)); self.output_dir_panel = ttk.Frame(save_frame); self.output_dir_panel.pack(fill=tk.X); ttk.Label(self.output_dir_panel, text="Output Directory:").pack(fill=tk.X, anchor='w'); dir_frame = ttk.Frame(self.output_dir_panel); dir_frame.pack(fill=tk.X, pady=(2, 0)); self.output_dir_entry = ttk.Entry(dir_frame, textvariable=self.output_dir_var, font=("Segoe UI", 9)); self.output_dir_entry.pack(side=tk.LEFT, expand=True, fill=tk.X); self.output_dir_button = ttk.Button(dir_frame, text="📁", width=3, command=self.browse_output_dir); self.output_dir_button.pack(side=tk.RIGHT, padx=(5, 0)); hugo_frame = ttk.LabelFrame(parent, text="🚀 Hugo Integration", padding=10); hugo_frame.pack(fill=tk.X, pady=(15, 0)); ttk.Checkbutton(hugo_frame, text="✅ Enable Hugo Integration", variable=self.hugo_enabled_var).pack(anchor=tk.W, pady=(0, 10)); ttk.Label(hugo_frame, text="Hugo Project Root Directory:").pack(fill=tk.X, anchor='w'); hugo_dir_frame = ttk.Frame(hugo_frame); hugo_dir_frame.pack(fill=tk.X, pady=(2, 8)); ttk.Entry(hugo_dir_frame, textvariable=self.hugo_dir_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, expand=True, fill=tk.X); ttk.Button(hugo_dir_frame, text="📁", width=3, command=self.browse_hugo_dir).pack(side=tk.RIGHT, padx=(5, 0)); ttk.Label(hugo_frame, text="Content Subdirectory:").pack(fill=tk.X, anchor='w'); ttk.Entry(hugo_frame, textvariable=self.hugo_content_dir_var, font=("Segoe UI", 9)).pack(fill=tk.X, pady=(2, 0))

    def setup_logging(self):
        # ... (code is unchanged)
        self.log_queue = queue.Queue(); formatter = logging.Formatter('%(levelname)s: %(message)s'); self.queue_handler = QueueHandler(self.log_queue); self.queue_handler.setFormatter(formatter); logging.getLogger().addHandler(self.queue_handler); logging.getLogger().setLevel(logging.INFO); self.root.after(100, self.poll_log_queue)

    def poll_log_queue(self):
        # ... (code is unchanged)
        while True:
            try: record = self.log_queue.get(block=False)
            except queue.Empty: break
            else:
                self.log_text_widget.config(state='normal'); tag = '';
                if "ERROR" in record or "❌" in record: tag = 'error'
                elif "WARNING" in record or "⚠️" in record: tag = 'warning'
                elif "✅" in record or "SUCCESS" in record or "🔑" in record: tag = 'success'
                elif "🔍" in record or "📺" in record or "🌐" in record or "🧠" in record or "🤖" in record: tag = 'info'
                self.log_text_widget.insert(tk.END, record + '\n', tag); self.log_text_widget.config(state='disabled'); self.log_text_widget.yview(tk.END)
        self.root.after(100, self.poll_log_queue)

    def run_full_process(self):
        # ... (code is unchanged)
        pass # Abbreviated

    def run_youtube_multimodal_analysis(self):
        # ... (code is unchanged)
        pass # Abbreviated

    def generate_notes_from_transcript(self, title: str, transcript: str, api_key: str) -> Tuple[str, str]:
        # ... (code is unchanged)
        pass # Abbreviated

    def run_youtube_playlist_analysis(self, playlist_url):
        """Processes a YouTube playlist using the text-based (transcript) pipeline."""
        logging.info(f"🚀 Starting YouTube Playlist (Transcript) Analysis for: {playlist_url}")
        self.progress_var.set("Fetching playlist info...")

        try:
            # Use extract_flat:True to get playlist metadata without fetching individual video info yet
            with yt_dlp.YoutubeDL({'extract_flat': True, 'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                video_entries = [entry for entry in info.get('entries', []) if entry and entry.get('id')]
                playlist_title = self.sanitize_filename(info.get('title', 'youtube-playlist'))
        except Exception as e:
            messagebox.showerror("Playlist Error", f"Could not fetch playlist videos: {e}"); logging.error(f"❌ Failed to get playlist info: {e}"); self._reset_processing_state(); return

        if not video_entries:
            messagebox.showinfo("No Videos", "Could not find any videos in the provided playlist URL."); self._reset_processing_state(); return

        # Determine the base output directory
        base_output_dir_str = self.output_dir_var.get()
        if self.auto_save_var.get() and base_output_dir_str:
            base_output_dir = Path(base_output_dir_str)
        else:
            base_output_dir_str = filedialog.askdirectory(title=f"Select Base Directory to Save Playlist Folder")
            if not base_output_dir_str: logging.warning("User cancelled directory selection."); self._reset_processing_state(); return
            base_output_dir = Path(base_output_dir_str)

        # Create the final, sanitized subdirectory for this playlist
        final_output_dir = base_output_dir / playlist_title
        final_output_dir.mkdir(exist_ok=True)
        logging.info(f"Playlist files will be saved to: {final_output_dir}")

        # Stage 1: Download all transcripts
        yt_researcher = EnhancedYouTubeResearcher()
        transcripts_map = yt_researcher.get_transcripts_for_playlist(video_entries)

        if not transcripts_map:
            messagebox.showwarning("No Transcripts", "Could not retrieve any valid transcripts from the playlist videos."); self._reset_processing_state(); return

        # Stage 2: Generate notes in parallel from transcripts
        num_videos = len(transcripts_map)
        num_keys = len(self.loaded_api_keys)
        logging.info(f"Starting note generation for {num_videos} transcripts with {num_keys} API key(s).")
        api_key_cycle = cycle(self.loaded_api_keys)
        processed_files = []

        try:
            with ThreadPoolExecutor(max_workers=num_keys) as executor:
                future_to_title = {executor.submit(self.generate_notes_from_transcript, title, transcript, next(api_key_cycle)): title for title, transcript in transcripts_map.items()}

                for i, future in enumerate(as_completed(future_to_title), 1):
                    title = future_to_title[future]
                    self.progress_var.set(f"Generating notes {i}/{num_videos}: {title[:40]}...")
                    try:
                        _, markdown_content = future.result()
                        sanitized_title = self.sanitize_filename(title)
                        filename = f"{i:03d}_{sanitized_title}.md"
                        filepath = final_output_dir / filename # Save in the new subdirectory
                        with open(filepath, 'w', encoding='utf-8') as f: f.write(markdown_content)
                        processed_files.append(str(filepath))
                        logging.info(f"✅ Saved study guide: {filename}")
                    except Exception as e:
                        logging.error(f"❌ Failed to process and save notes for video '{title}': {e}")
            
            self.last_generated_files = processed_files
            self.final_notes_content = f"# Playlist Processing Complete\n\nSuccessfully processed and saved {len(processed_files)} out of {len(video_entries)} videos.\n\nFiles are located in:\n`{final_output_dir}`"
            self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
            messagebox.showinfo("Playlist Complete", f"Finished processing the playlist.\n\n{len(processed_files)} study guides saved to:\n{final_output_dir}")
        finally:
            self._reset_processing_state()

    def run_multimodal_upload_analysis(self):
        # ... (code is unchanged)
        pass # Abbreviated
    
    def sanitize_filename(self, text: str) -> str:
        """Converts a string into a URL-friendly and filesystem-safe slug."""
        if not text: return "untitled"
        # Replace common separators with a hyphen
        text = re.sub(r'[\s._|]+', '-', text)
        # Remove any character that is not a letter, a number, a hyphen, or an underscore
        text = re.sub(r'[^\w\s-]', '', text).strip().lower()
        # Collapse consecutive hyphens
        text = re.sub(r'--+', '-', text)
        # Limit length to avoid issues with long file paths
        return text.strip('-_')[:100]

    # ... all other helper functions like _is_youtube_playlist, _update_url_status, start_task_thread etc. are unchanged.
    def _is_youtube_playlist(self, url: str) -> bool:
        if not url: return False
        playlist_regex = re.compile(r'(?:&|\?)list=([a-zA-Z0-9_-]+)')
        return playlist_regex.search(url) is not None
    def _update_url_status(self, *args):
        if self.input_mode_var.get() == "youtube_video":
            url = self.url_var.get()
            if not url.strip(): self.url_status_label.config(text=""); return
            if self._is_youtube_playlist(url): self.url_status_label.config(text="▶️ YouTube Playlist Detected. Will process all video transcripts.", foreground=DARK_THEME['success'])
            else: self.url_status_label.config(text="Single YouTube Video Detected. Will perform multimodal analysis.", foreground=DARK_THEME['info'])
        else: self.url_status_label.config(text="")
    def toggle_depth_setting(self):
        new_state = tk.NORMAL if self.deep_crawl_var.get() else tk.DISABLED
        self.depth_label.config(state=new_state); self.depth_spinbox.config(state=new_state)
    def toggle_input_mode(self, *args):
        mode = self.input_mode_var.get()
        is_scrape, is_yt, is_upload = (mode == "scrape"), (mode == "youtube_video"), (mode == "upload")
        self._set_child_widgets_state(self.url_input_frame, tk.NORMAL if is_scrape or is_yt else tk.DISABLED)
        self._set_child_widgets_state(self.upload_frame, tk.NORMAL if is_upload else tk.DISABLED)
        if is_scrape: self.scraper_options_frame.pack(fill=tk.X, pady=(0, 0)); self.toggle_depth_setting()
        else: self.scraper_options_frame.pack_forget()
        self.research_checkbutton.config(state=tk.NORMAL if is_scrape or is_upload else tk.DISABLED)
        if is_yt: self.research_enabled_var.set(False)
        self.multimodal_check.config(state=tk.NORMAL if is_upload else tk.DISABLED)
        self.toggle_research_panel(); self._update_url_status()
    def update_temperature_label(self, value): self.temp_label.configure(text=f"Temperature: {float(value):.1f}")
    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            if widget in [self.multimodal_check, self.url_status_label]: continue
            if isinstance(widget, (ttk.Frame, ttk.LabelFrame)): self._set_child_widgets_state(widget, state)
            else:
                try: widget.configure(state=state)
                except tk.TclError: pass
    def add_files(self):
        filetypes = [("All Supported Files", "*.pdf *.docx *.txt *.md *.png *.jpg *.jpeg *.webp *.mp4 *.mov *.avi"), ("Documents", "*.pdf *.docx *.txt *.md"), ("Images", "*.png *.jpg *.jpeg *.webp"), ("Videos", "*.mp4 *.mov *.avi *.mpeg"), ("All files", "*.*")]
        files = filedialog.askopenfilenames(title="Select Files for Analysis", filetypes=filetypes)
        for f in files:
            if f not in self.file_listbox.get(0, tk.END): self.file_listbox.insert(tk.END, f)
    def clear_files(self): self.file_listbox.delete(0, tk.END)
    def browse_api_key(self):
        filepath = filedialog.askopenfilename(title="Select Gemini API Key File", filetypes=[("Key files", "*.key"), ("Text files", "*.txt")]);
        if filepath: self.api_key_file_var.set(filepath)
    def update_api_key_status(self, *args):
        filepath = self.api_key_file_var.get()
        self.loaded_api_keys = APIKeyManager.get_gemini_api_keys(filepath) if filepath else []
        num_keys = len(self.loaded_api_keys)
        if num_keys == 0: self.api_key_status_label.config(text="No valid keys found in file.", foreground=DARK_THEME['error'])
        elif num_keys == 1: self.api_key_status_label.config(text="✅ Loaded 1 API Key.", foreground=DARK_THEME['success'])
        else: self.api_key_status_label.config(text=f"✅ Loaded {num_keys} API Keys for parallel processing.", foreground=DARK_THEME['success'])
    def browse_hugo_dir(self):
        filepath = filedialog.askdirectory(title="Select Hugo Project Root Directory");
        if filepath: self.hugo_dir_var.set(filepath)
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Directory to Save Markdown Files");
        if directory: self.output_dir_var.set(directory)
    def load_prompt_from_file(self):
        filepath = filedialog.askopenfilename(title="Select Prompt File", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
                self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, content)
            except Exception as e: messagebox.showerror("Error", f"Failed to load prompt: {e}")
    def toggle_auto_save_panel(self):
        new_state = tk.NORMAL if self.auto_save_var.get() else tk.DISABLED
        self.output_dir_entry.config(state=new_state); self.output_dir_button.config(state=new_state)
    def toggle_research_panel(self):
        panel_state = tk.NORMAL if self.research_enabled_var.get() else tk.DISABLED
        self._set_child_widgets_state(self.web_research_panel, panel_state); self._set_child_widgets_state(self.yt_research_panel, panel_state)
        if not self.research_enabled_var.get(): return
        if not PLAYWRIGHT_AVAILABLE: self.playwright_radio.config(state=tk.DISABLED)
        if not YT_DLP_AVAILABLE: self.yt_checkbutton.config(state=tk.DISABLED)
    def load_initial_settings(self):
        self.config = self._load_config_file()
        prompt_template = self._load_prompt_file()
        # ... logic to load config ...
        self.update_api_key_status(); self.update_temperature_label(self.temperature_var.get()); self._check_dependencies()
    def _check_dependencies(self):
        # ... logic to check deps ...
        pass
    def reset_prompt_to_default(self):
        default_prompt = self._load_prompt_file("prompt.md") or "You are an expert educational content creator..."
        self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, default_prompt)
    def save_notes(self):
        # ... logic to save manually ...
        pass
    def _reset_processing_state(self):
        self.is_processing = False
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"))
        if self.last_generated_files: self.root.after(0, lambda: self.progress_var.set(f"Complete! {len(self.last_generated_files)} file(s) saved."))
        elif self.final_notes_content: self.root.after(0, lambda: self.progress_var.set("Complete! Ready to save."))
        else: self.root.after(0, lambda: self.progress_var.set("Ready"))
    def start_task_thread(self):
        if self.is_processing: messagebox.showwarning("Processing", "A task is already running."); return
        if not self.loaded_api_keys: messagebox.showerror("API Key Error", "Cannot start. Please provide a valid Gemini API key file."); return
        self.is_processing = True; self.start_button.config(state=tk.DISABLED, text="⏳ Processing..."); self.save_button.config(state=tk.DISABLED)
        self.final_notes_content, self.last_generated_files = "", []; self.progress_var.set("Initializing...")
        self.log_text_widget.config(state='normal'); self.log_text_widget.delete('1.0', tk.END); self.log_text_widget.config(state='disabled')
        mode = self.input_mode_var.get(); target_function, args = None, ()
        if mode == "youtube_video":
            url = self.url_var.get().strip()
            if not url: messagebox.showerror("Input Error", "YouTube URL cannot be empty."); self._reset_processing_state(); return
            if self._is_youtube_playlist(url): target_function, args = self.run_youtube_playlist_analysis, (url,)
            else: target_function = self.run_youtube_multimodal_analysis
        elif mode == "upload" and self.multimodal_upload_var.get(): target_function = self.run_multimodal_upload_analysis
        else: target_function = self.run_full_process
        if target_function: threading.Thread(target=target_function, args=args, daemon=True).start()
        else: messagebox.showerror("Internal Error", "Could not determine the correct action."); self._reset_processing_state()
    def _finalize_generation(self):
        if self.auto_save_var.get() and self.output_dir_var.get(): self._perform_auto_save()
        elif self.hugo_enabled_var.get(): self.handle_hugo_integration()
        else:
            self.save_button.config(state=tk.NORMAL)
            if not self.is_processing: messagebox.showinfo("Success!", "Study guide generated successfully!\n\nClick 'Save Study Guide' to save the file.")
    def _perform_auto_save(self): pass # Abbreviated
    def handle_hugo_integration(self): pass # Abbreviated
    def prompt_and_open_in_browser(self): pass # Abbreviated
    def show_demo_content(self): pass # Abbreviated
    def _load_config_file(self, filepath="config.yml"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError): return {}
    def _load_prompt_file(self, filepath="prompt.md"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return f.read()
        except FileNotFoundError: return ""

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root = tk.Tk()
    app = AdvancedScraperApp(root)
    def on_closing():
        if app.is_processing and messagebox.askokcancel("Quit", "A processing task is still running. Are you sure you want to quit?"): root.destroy()
        elif not app.is_processing: root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
