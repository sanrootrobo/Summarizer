import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from collections import defaultdict

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
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz
import docx

# --- USER_AGENTS constant for the Playwright researcher ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str) -> str | None:
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f: 
                key = f.read().strip()
            if not key: 
                raise ValueError("Key is empty")
            logging.info(f"Gemini API key loaded from {filepath}.")
            return key
        except Exception as e:
            logging.error(f"Failed to load Gemini API key from {filepath}: {e}")
            return None

    @staticmethod
    def get_google_search_credentials(key_path: str, cx_path: str) -> tuple[str | None, str | None]:
        try:
            with open(Path(key_path), 'r') as f: 
                api_key = f.read().strip()
            with open(Path(cx_path), 'r') as f: 
                cx_id = f.read().strip()
            if not api_key or not cx_id: 
                raise ValueError("Key or CX is empty")
            logging.info("Google Search credentials loaded successfully.")
            return api_key, cx_id
        except Exception as e:
            logging.warning(f"Could not load Google Search credentials: {e}. Will use fallback.")
            return None, None

class LocalDocumentLoader:
    def __init__(self, file_paths: list[str]): 
        self.file_paths = file_paths
        
    def _read_txt(self, p: Path) -> str: 
        return p.read_text(encoding='utf-8', errors='ignore')
        
    def _read_pdf(self, p: Path) -> str:
        with fitz.open(p) as doc: 
            return "".join(page.get_text() for page in doc)
            
    def _read_docx(self, p: Path) -> str:
        return "\n".join(para.text for para in docx.Document(p).paragraphs)
        
    def load_and_extract_text(self) -> dict[str, str]:
        content = {}
        for path_str in self.file_paths:
            p, name = Path(path_str), Path(path_str).name
            logging.info(f"Processing local file: {name}")
            try:
                if name.lower().endswith(".pdf"): 
                    content[name] = self._read_pdf(p)
                elif name.lower().endswith(".docx"): 
                    content[name] = self._read_docx(p)
                elif name.lower().endswith(".txt"): 
                    content[name] = self._read_txt(p)
                else: 
                    logging.warning(f"Unsupported file type skipped: {name}")
            except Exception as e: 
                logging.error(f"Failed to process file '{name}': {e}")
        logging.info(f"Successfully processed {len(content)} local documents.")
        return content

class EnhancedResearchQueryGenerator:
    """Improved query generator with better topic extraction and diverse query types"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key, 
            temperature=0.3  # Lower temperature for more focused results
        )
        self.output_parser = StrOutputParser()
        
    def extract_main_topic_and_subtopics(self, content: dict[str, str]) -> Dict[str, List[str]]:
        """Extract main topic and related subtopics for better research targeting"""
        logging.info("🧠 Analyzing content structure and extracting topics...")
        
        # Combine content with size limits
        context = ""
        for name, text in list(content.items())[:3]:  # Use top 3 sources
            truncated_text = text[:2000] if len(text) > 2000 else text
            context += f"\n--- {name} ---\n{truncated_text}"
        
        # Limit total context size
        if len(context) > 6000:
            context = context[:6000] + "..."
            
        prompt = ChatPromptTemplate.from_template("""
Analyze this content and extract:
1. Main topic (the primary subject/technology)
2. Key subtopics (3-5 related concepts, features, or areas)

Return as JSON:
{{"main_topic": "topic name", "subtopics": ["subtopic1", "subtopic2", ...]}}

CONTENT:
{context}

JSON Response:
        """)
        
        try:
            response = (prompt | self.llm | self.output_parser).invoke({"context": context})
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                main_topic = result.get('main_topic', 'General Topic')
                subtopics = result.get('subtopics', [])
                logging.info(f"✅ Main topic: {main_topic}")
                logging.info(f"✅ Subtopics: {', '.join(subtopics)}")
                return {"main_topic": main_topic, "subtopics": subtopics}
        except Exception as e:
            logging.error(f"❌ Topic extraction failed: {e}")
            
        # Fallback extraction
        return {"main_topic": "General Topic", "subtopics": []}

    def generate_diverse_research_queries(self, topic_data: Dict[str, List[str]], max_queries: int = 8) -> List[str]:
        """Generate diverse research queries covering multiple aspects"""
        logging.info(f"🧠 Generating diverse research queries...")
        
        main_topic = topic_data["main_topic"]
        subtopics = topic_data["subtopics"]
        
        # Query templates for different research angles
        query_templates = [
            f"{main_topic} complete tutorial guide",
            f"{main_topic} best practices production",
            f"{main_topic} common errors troubleshooting",
            f"{main_topic} advanced techniques tips",
            f"{main_topic} performance optimization",
            f"{main_topic} vs alternatives comparison",
            f"how to use {main_topic} examples",
            f"{main_topic} latest updates 2024",
        ]
        
        # Add subtopic-specific queries
        for subtopic in subtopics[:3]:  # Limit to top 3 subtopics
            query_templates.extend([
                f"{main_topic} {subtopic} tutorial",
                f"{subtopic} {main_topic} implementation",
            ])
        
        # Use LLM to refine and expand queries
        prompt = ChatPromptTemplate.from_template("""
Based on the topic "{main_topic}" and subtopics {subtopics}, generate {max_queries} diverse search queries.
Include queries for: tutorials, troubleshooting, best practices, comparisons, and recent updates.

Return as a JSON array of strings only:
        """)
        
        try:
            response = (prompt | self.llm | self.output_parser).invoke({
                "main_topic": main_topic,
                "subtopics": subtopics,
                "max_queries": max_queries
            })
            
            # Extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                llm_queries = json.loads(json_match.group(0))
                if isinstance(llm_queries, list) and len(llm_queries) > 0:
                    logging.info(f"✅ Generated {len(llm_queries)} LLM queries")
                    return llm_queries[:max_queries]
        except Exception as e:
            logging.warning(f"⚠️ LLM query generation failed, using templates: {e}")
        
        # Fallback to template queries
        selected_queries = query_templates[:max_queries]
        logging.info(f"✅ Using {len(selected_queries)} template queries")
        return selected_queries

class EnhancedGoogleSearchResearcher:
    """Improved Google Search with better error handling and result filtering"""
    
    def __init__(self, api_key: str = None, cx_id: str = None):
        self.api_key = api_key
        self.cx_id = cx_id
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(USER_AGENTS)
        })
        
    def _is_quality_url(self, url: str, exclude_domain: str) -> bool:
        """Enhanced URL quality filtering"""
        if not url or exclude_domain in url:
            return False
            
        # Filter out unwanted file types and low-quality domains
        bad_extensions = ['.pdf', '.zip', '.doc', '.docx', '.ppt', '.pptx']
        low_quality_domains = ['pinterest.com', 'facebook.com', 'twitter.com', 'instagram.com']
        
        if any(url.lower().endswith(ext) for ext in bad_extensions):
            return False
            
        parsed_url = urlparse(url)
        if any(domain in parsed_url.netloc.lower() for domain in low_quality_domains):
            return False
            
        return True
        
    def search_and_extract_urls(self, queries: list[str], exclude_domain: str, max_results_per_query: int = 8) -> list[str]:
        """Enhanced search with better result collection and deduplication"""
        all_urls = set()
        successful_queries = 0
        
        for i, query in enumerate(queries):
            logging.info(f"🔍 Searching [{i+1}/{len(queries)}]: {query}")
            
            try:
                if self.api_key and self.cx_id:
                    # Google Custom Search API
                    params = {
                        'key': self.api_key,
                        'cx': self.cx_id,
                        'q': query,
                        'num': max_results_per_query,
                        'dateRestrict': 'y2'  # Results from last 2 years
                    }
                    response = self.session.get(
                        "https://www.googleapis.com/customsearch/v1", 
                        params=params,
                        timeout=10
                    )
                    response.raise_for_status()
                    results = response.json().get('items', [])
                    query_urls = [item['link'] for item in results]
                    
                else:
                    # DuckDuckGo fallback
                    params = {'q': query, 'kl': 'us-en'}
                    response = self.session.get(
                        "https://html.duckduckgo.com/html/", 
                        params=params,
                        timeout=15
                    )
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    query_urls = []
                    for link in soup.select('a.result__a'):
                        href = link.get('href', '')
                        if 'uddg=' in href:
                            try:
                                actual_url = unquote(href.split('uddg=')[1])
                                query_urls.append(actual_url)
                            except:
                                continue
                
                # Filter and add quality URLs
                quality_urls = [url for url in query_urls if self._is_quality_url(url, exclude_domain)]
                all_urls.update(quality_urls)
                successful_queries += 1
                
                logging.info(f"  ✅ Found {len(quality_urls)} quality URLs")
                
                # Rate limiting
                sleep(random.uniform(1.0, 2.0))
                
            except Exception as e:
                logging.error(f"  ❌ Search failed for '{query}': {e}")
                
        logging.info(f"🔍 Search complete: {len(all_urls)} unique URLs from {successful_queries}/{len(queries)} queries")
        return list(all_urls)

class EnhancedYouTubeResearcher:
    """Significantly improved YouTube research with better search, filtering, and transcript extraction"""
    
    def __init__(self):
        if not YT_DLP_AVAILABLE:
            raise ImportError("YouTube research requires 'yt-dlp'. Please install it.")
        self.temp_dirs = []  # Track temp directories for cleanup
        logging.info("🔎 Initialized Enhanced YouTube Researcher")

    def _clean_transcript_text(self, vtt_content: str) -> str:
        """Enhanced transcript cleaning with better text processing"""
        if not vtt_content:
            return ""
            
        lines = vtt_content.splitlines()
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip VTT headers, timestamps, and empty lines
            if (not line or 
                line.startswith('WEBVTT') or 
                '-->' in line or 
                line.startswith('NOTE') or
                re.match(r'^\d+$', line)):
                continue
                
            # Remove HTML tags and clean text
            line = re.sub(r'<[^>]+>', '', line)
            line = re.sub(r'&[a-zA-Z]+;', '', line)  # Remove HTML entities
            text_lines.append(line)
        
        # Join lines and remove duplicates
        full_text = ' '.join(text_lines)
        
        # Remove common transcript artifacts
        full_text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]', '', full_text)
        full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
        
        return full_text.strip()

    def _extract_transcript_from_file(self, file_path: Path) -> str:
        """Enhanced transcript extraction with better error handling"""
        try:
            if not file_path.exists():
                return ""
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            return self._clean_transcript_text(content)
            
        except Exception as e:
            logging.warning(f"Failed to read transcript file {file_path}: {e}")
            return ""

    def _get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Get video metadata without downloading"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False
                )
                
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date'),
                    'subtitles': info.get('subtitles', {}),
                    'automatic_captions': info.get('automatic_captions', {})
                }
        except Exception:
            return None

    def _has_quality_subtitles(self, metadata: Dict) -> bool:
        """Check if video has good quality subtitles"""
        if not metadata:
            return False
            
        subtitles = metadata.get('subtitles', {})
        auto_captions = metadata.get('automatic_captions', {})
        
        # Prefer manual subtitles over auto-generated
        english_subs = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
        
        for lang in english_subs:
            if lang in subtitles:
                return True
                
        # Accept auto-captions if no manual subtitles
        for lang in english_subs:
            if lang in auto_captions:
                return True
                
        return False

    def download_transcript(self, video_id: str, title: str = "") -> Optional[str]:
        """Enhanced transcript download with better error handling"""
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        temp_dir = Path(f"./temp_subs_{video_id}_{random.randint(1000, 9999)}")
        temp_dir.mkdir(exist_ok=True)
        self.temp_dirs.append(temp_dir)
        
        try:
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],
                'skip_download': True,
                'outtmpl': str(temp_dir / '%(id)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            # Look for subtitle files
            subtitle_files = list(temp_dir.glob('*.vtt'))
            if not subtitle_files:
                return None
            
            # Prefer manual subtitles over auto-generated
            manual_subs = [f for f in subtitle_files if '.auto.' not in f.name]
            subtitle_file = manual_subs[0] if manual_subs else subtitle_files[0]
            
            transcript = self._extract_transcript_from_file(subtitle_file)
            
            if len(transcript) < 100:  # Too short, probably not useful
                return None
                
            return transcript
            
        except Exception as e:
            logging.warning(f"Failed to download transcript for {video_id}: {e}")
            return None

    def search_videos_with_yt_dlp(self, query: str, max_results: int = 10) -> List[Dict]:
        """Enhanced video search with better filtering"""
        try:
            # Use yt-dlp to search
            search_query = f"ytsearch{max_results*2}:{query}"  # Get more results for filtering
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(search_query, download=False)
                
            if not search_results or 'entries' not in search_results:
                return []
                
            videos = []
            for entry in search_results['entries']:
                if not entry or not entry.get('id'):
                    continue
                    
                # Get detailed metadata
                metadata = self._get_video_metadata(entry['id'])
                if not metadata:
                    continue
                    
                # Filter by quality criteria
                duration = metadata.get('duration', 0)
                view_count = metadata.get('view_count', 0)
                
                # Skip very short or very long videos
                if duration < 120 or duration > 3600:  # 2min to 1hour
                    continue
                    
                # Skip videos with very low views (likely low quality)
                if view_count < 1000:
                    continue
                    
                # Must have subtitles
                if not self._has_quality_subtitles(metadata):
                    continue
                    
                videos.append(metadata)
                
                if len(videos) >= max_results:
                    break
                    
            return videos
            
        except Exception as e:
            logging.error(f"Video search failed for '{query}': {e}")
            return []

    def get_transcripts_for_queries(self, queries: List[str], max_videos_per_query: int = 3) -> Dict[str, str]:
        """Enhanced transcript collection with parallel processing and better filtering"""
        logging.info(f"▶️  Starting enhanced YouTube research for {len(queries)} queries...")
        
        all_videos = {}  # video_id -> metadata
        
        # Search for videos
        for i, query in enumerate(queries):
            logging.info(f"🔍 YouTube search [{i+1}/{len(queries)}]: {query}")
            
            videos = self.search_videos_with_yt_dlp(query, max_videos_per_query)
            
            for video in videos:
                video_id = video['id']
                if video_id not in all_videos:
                    all_videos[video_id] = video
                    
            logging.info(f"  ✅ Found {len(videos)} quality videos")
            sleep(1.0)  # Rate limiting
        
        if not all_videos:
            logging.warning("No suitable videos found for transcript extraction")
            return {}
            
        logging.info(f"📺 Found {len(all_videos)} unique videos. Downloading transcripts...")
        
        # Download transcripts in parallel
        transcripts = {}
        
        def download_single_transcript(video_data):
            video_id, metadata = video_data
            transcript = self.download_transcript(video_id, metadata.get('title', ''))
            if transcript:
                return video_id, metadata, transcript
            return None
        
        with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers for YouTube rate limits
            future_to_video = {
                executor.submit(download_single_transcript, item): item[0] 
                for item in all_videos.items()
            }
            
            for future in as_completed(future_to_video):
                try:
                    result = future.result(timeout=60)  # 60 second timeout per video
                    if result:
                        video_id, metadata, transcript = result
                        title = metadata.get('title', 'Unknown Title')
                        duration = metadata.get('duration', 0)
                        view_count = metadata.get('view_count', 0)
                        
                        content = f"""Video Title: {title}
Duration: {duration//60}:{duration%60:02d}
Views: {view_count:,}
URL: https://www.youtube.com/watch?v={video_id}

Transcript:
{transcript}"""
                        
                        transcripts[f"https://www.youtube.com/watch?v={video_id}"] = content
                        logging.info(f"  ✅ Extracted transcript: {title[:50]}...")
                        
                except Exception as e:
                    logging.error(f"  ❌ Error processing video: {e}")
        
        # Cleanup temp directories
        self._cleanup()
        
        logging.info(f"▶️  YouTube research complete: {len(transcripts)} transcripts extracted")
        return transcripts
    
    def _cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        self.temp_dirs.clear()

# [Continue with the rest of the classes - PlaywrightResearcher, WebsiteScraper, etc. - keeping them largely the same but with minor improvements]

class EnhancedPlaywrightResearcher:
    """Enhanced Playwright researcher with better error handling and search diversity"""
    
    def __init__(self):
        self.search_engines = [
            {"name": "DuckDuckGo", "url": "https://duckduckgo.com/", "input_selector": 'input[name="q"]'},
            {"name": "Bing", "url": "https://www.bing.com/", "input_selector": 'input[name="q"]'},
        ]
        logging.info("🤖 Initialized Enhanced Playwright Researcher")

    def search_and_extract_urls(self, queries: List[str], exclude_domain: str) -> List[str]:
        """Enhanced search using multiple search engines"""
        all_urls = set()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=random.choice(USER_AGENTS))
            page = context.new_page()
            
            try:
                for i, query in enumerate(queries):
                    logging.info(f"🤖 Playwright search [{i+1}/{len(queries)}]: {query}")
                    
                    # Try different search engines
                    for engine in self.search_engines:
                        try:
                            page.goto(engine["url"], timeout=20000)
                            page.locator(engine["input_selector"]).fill(query)
                            page.locator(engine["input_selector"]).press("Enter")
                            
                            # Wait for results with multiple possible selectors
                            result_selectors = ["div#links", ".b_results", "[data-testid='result']"]
                            for selector in result_selectors:
                                try:
                                    page.wait_for_selector(selector, timeout=10000)
                                    break
                                except:
                                    continue
                            
                            # Extract links with multiple selectors
                            link_selectors = ["h2 > a", ".b_title > h2 > a", "[data-testid='result-title-a']"]
                            found_urls = set()
                            
                            for selector in link_selectors:
                                try:
                                    links = page.locator(selector).all()
                                    for link in links:
                                        href = link.get_attribute('href')
                                        if href and self._is_useful_url(href, exclude_domain):
                                            found_urls.add(href)
                                except:
                                    continue
                            
                            all_urls.update(found_urls)
                            logging.info(f"  ✅ {engine['name']}: {len(found_urls)} URLs")
                            
                            # Rate limiting between engines
                            sleep(random.uniform(2.0, 3.0))
                            break  # Success, move to next query
                            
                        except Exception as e:
                            logging.warning(f"  ⚠️ {engine['name']} failed: {e}")
                            continue
                    
                    # Rate limiting between queries
                    sleep(random.uniform(3.0, 5.0))
                    
            finally:
                browser.close()
        
        logging.info(f"🤖 Playwright search complete: {len(all_urls)} unique URLs")
        return list(all_urls)

    def _is_useful_url(self, url: str, domain: str) -> bool:
        """Enhanced URL filtering"""
        if not url or domain in url:
            return False
            
        # Filter out unwanted extensions and domains
        bad_extensions = ['.pdf', '.zip', '.doc', '.docx', '.ppt', '.pptx', '.exe']
        bad_domains = ['pinterest.com', 'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com']
        
        url_lower = url.lower()
        if any(url_lower.endswith(ext) for ext in bad_extensions):
            return False
            
        parsed_url = urlparse(url)
        if any(bad_domain in parsed_url.netloc.lower() for bad_domain in bad_domains):
            return False
            
        return True

# [Keep the existing WebsiteScraper class mostly the same, with minor improvements]

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
        
        # Remove unwanted elements
        for element in soup.select('script, style, nav, footer, header, aside, form, .ad, .advertisement'):
            element.decompose()
            
        # Find main content area
        main_content_area = (
            soup.select_one('main, article, [role="main"], .main-content, .content, .post-content') 
            or soup.body
        )
        
        # Convert to markdown
        text_maker = html2text.HTML2Text()
        text_maker.body_width = 0
        text_maker.ignore_links = True
        text_maker.ignore_images = True
        
        markdown_content = text_maker.handle(str(main_content_area)) if main_content_area else ""
        
        # Clean up markdown
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content).strip()
        markdown_content = re.sub(r'[ \t]+', ' ', markdown_content)  # Normalize spaces
        
        # Extract internal links
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(page_url, href).split('#')[0]  # Remove anchors
            if urlparse(full_url).netloc == self.domain:
                links.add(full_url)
        
        return markdown_content, list(links)
    
    def crawl(self, additional_urls: Optional[List[str]] = None) -> dict[str, str]:
        """Enhanced crawling with better URL management"""
        urls_to_visit = {self.base_url}
        
        # Add external research URLs
        if additional_urls:
            urls_to_visit.update(additional_urls)
            logging.info(f"Added {len(additional_urls)} external research URLs to crawl queue")

        processed_count = 0
        while urls_to_visit:
            # Check limits
            if 0 < self.max_pages <= processed_count:
                logging.info(f"Reached scraping limit of {self.max_pages} pages")
                break
                
            url = urls_to_visit.pop()
            if url in self.visited_urls:
                continue
                
            self.visited_urls.add(url)
            processed_count += 1
            
            # Determine log prefix
            is_external = additional_urls and url in additional_urls
            log_prefix = "Research page" if is_external else "Scraping page"
            
            logging.info(f"[{processed_count}/{self.max_pages or '∞'}] {log_prefix}: {url}")

            html = self._get_page_content(url)
            if html:
                text, new_links = self._parse_content_and_links(html, url)
                
                # Only save substantial content
                if text and len(text.strip()) > 200:
                    self.scraped_content[url] = text

                # Only follow internal links for base domain crawling, not for external research
                if not is_external and new_links:
                    # Limit new links to prevent infinite crawling
                    new_links_to_add = list(new_links)[:10]  # Max 10 new links per page
                    urls_to_visit.update(new_links_to_add)

            sleep(self.rate_limit_delay)
            
        logging.info(f"Crawling complete. Scraped {len(self.scraped_content)} valid pages")
        return self.scraped_content

class EnhancedNoteGenerator:
    """Enhanced note generator with better content structuring"""
    
    def __init__(self, api_key: str, llm_config: dict, prompt_template_string: str):
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config['model_name'], 
            google_api_key=api_key, 
            **llm_config['parameters']
        )
        self.output_parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_template(prompt_template_string)

    def _prepare_content_for_generation(self, source_data: dict[str, str]) -> str:
        """Better content preparation with source organization"""
        if not source_data:
            return ""
            
        # Organize content by source type
        web_content = []
        video_content = []
        local_content = []
        
        for source_url, content in source_data.items():
            if 'youtube.com' in source_url:
                video_content.append((source_url, content))
            elif source_url.startswith('http'):
                web_content.append((source_url, content))
            else:
                local_content.append((source_url, content))
        
        # Build organized content string
        organized_content = ""
        
        if local_content:
            organized_content += "\n\n=== LOCAL DOCUMENTS ===\n"
            for name, content in local_content:
                organized_content += f"\n--- SOURCE: {name} ---\n{content[:3000]}\n"
        
        if web_content:
            organized_content += "\n\n=== WEB RESEARCH ===\n"
            for url, content in web_content[:5]:  # Limit web sources
                organized_content += f"\n--- SOURCE: {url} ---\n{content[:2000]}\n"
        
        if video_content:
            organized_content += "\n\n=== VIDEO TRANSCRIPTS ===\n"
            for url, content in video_content:
                organized_content += f"\n--- SOURCE: {url} ---\n{content[:2500]}\n"
        
        return organized_content

    def generate_comprehensive_notes(self, source_data: dict[str, str], source_name: str) -> str:
        """Enhanced note generation with better content handling"""
        if not source_data:
            return "No content was provided to generate notes."
            
        logging.info(f"📝 Generating comprehensive notes from {len(source_data)} sources...")
        
        # Prepare and organize content
        organized_content = self._prepare_content_for_generation(source_data)
        
        # Add metadata
        metadata = f"""
GENERATION METADATA:
- Total Sources: {len(source_data)}
- Source Types: {self._get_source_type_summary(source_data)}
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Primary Source: {source_name}

"""
        
        try:
            # Generate notes using the organized content
            chain = self.prompt | self.llm | self.output_parser
            notes = chain.invoke({
                "content": organized_content,
                "website_url": source_name,
                "source_count": len(source_data),
                "metadata": metadata
            })
            
            # Add metadata to the beginning of notes
            final_notes = metadata + "\n" + "="*80 + "\n\n" + notes
            
            logging.info("✅ Note generation completed successfully")
            return final_notes
            
        except Exception as e:
            logging.error(f"❌ Error during note generation: {e}")
            return f"""# Generation Error

An error occurred while communicating with the AI model:

**Error Details:** `{e}`

**Available Content Summary:**
- Total sources processed: {len(source_data)}
- Content length: {len(organized_content)} characters

Please check your API key and try again.
"""

    def _get_source_type_summary(self, source_data: dict[str, str]) -> str:
        """Get a summary of source types"""
        types = {"Web": 0, "YouTube": 0, "Local": 0}
        
        for source_url in source_data.keys():
            if 'youtube.com' in source_url:
                types["YouTube"] += 1
            elif source_url.startswith('http'):
                types["Web"] += 1
            else:
                types["Local"] += 1
        
        return ", ".join([f"{k}: {v}" for k, v in types.items() if v > 0])

# --- GUI Application with Enhanced Features ---

class QueueHandler(logging.Handler):
    """Enhanced logging handler with better formatting"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        # Add timestamp to log messages
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_record = f"[{timestamp}] {self.format(record)}"
        self.log_queue.put(formatted_record)

class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Research & Study Guide Generator v2.0")
        self.root.geometry("1200x900")
        self.root.minsize(800, 600)

        # --- TKINTER VARIABLES ---
        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=0)
        self.research_enabled_var = tk.BooleanVar(value=False)
        self.web_research_enabled_var = tk.BooleanVar(value=True)
        self.yt_research_enabled_var = tk.BooleanVar(value=False)
        self.web_research_method_var = tk.StringVar(value="google_api")
        self.research_pages_var = tk.IntVar(value=8)
        self.research_queries_var = tk.IntVar(value=6)
        self.google_api_key_file_var = tk.StringVar()
        self.google_cx_file_var = tk.StringVar()
        self.yt_videos_per_query_var = tk.IntVar(value=3)
        self.api_key_file_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        self.final_notes_content = ""
        self.config = {}
        self.is_processing = False

        # --- UI AND LOGGING SETUP ---
        self.create_widgets()
        self.load_initial_settings()
        self.toggle_input_mode()
        self.toggle_research_panel()
        self.setup_logging()

    def create_widgets(self):
        """Enhanced widget creation with better layout"""
        # Main container with better spacing
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create paned window for resizable sections
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for settings
        settings_frame = ttk.Frame(paned_window, width=450)
        paned_window.add(settings_frame, weight=0)
        
        # Right panel for logs
        log_frame = ttk.Frame(paned_window)
        paned_window.add(log_frame, weight=1)

        # Enhanced notebook with better styling
        notebook = ttk.Notebook(settings_frame)
        notebook.pack(fill="both", expand=True, pady=(0, 10))

        # Create tabs
        source_tab = ttk.Frame(notebook, padding=15)
        self.research_tab = ttk.Frame(notebook, padding=15)
        ai_tab = ttk.Frame(notebook, padding=15)
        prompt_tab = ttk.Frame(notebook, padding=15)
        
        self.create_source_tab(source_tab)
        self.create_research_tab(self.research_tab)
        self.create_ai_tab(ai_tab)
        self.create_prompt_tab(prompt_tab)

        # Add tabs to notebook
        notebook.add(source_tab, text="📄 Source")
        notebook.add(self.research_tab, text="🔍 Research")
        notebook.add(ai_tab, text="🤖 AI Model")
        notebook.add(prompt_tab, text="📝 Prompt")

        # Control buttons with better styling
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(
            button_frame, 
            text="🚀 Start Generation", 
            command=self.start_task_thread,
            style="Accent.TButton"
        )
        self.start_button.pack(fill=tk.X, pady=(0, 5))
        
        self.save_button = ttk.Button(
            button_frame, 
            text="💾 Save Study Guide", 
            command=self.save_notes, 
            state=tk.DISABLED
        )
        self.save_button.pack(fill=tk.X)
        
        # Progress indicator
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(button_frame, textvariable=self.progress_var, foreground="blue")
        progress_label.pack(pady=(5, 0))

        # Enhanced log panel
        self.create_log_panel(log_frame)

    def create_source_tab(self, parent):
        """Enhanced source tab with better organization"""
        # Input mode selection
        mode_frame = ttk.LabelFrame(parent, text="📥 Input Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(
            mode_frame, 
            text="🌐 Scrape from URL", 
            variable=self.input_mode_var, 
            value="scrape", 
            command=self.toggle_input_mode
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            mode_frame, 
            text="📁 Upload Local Documents", 
            variable=self.input_mode_var, 
            value="upload", 
            command=self.toggle_input_mode
        ).pack(anchor=tk.W)

        # Web scraper settings
        self.scraper_frame = ttk.LabelFrame(parent, text="🕷️ Web Scraper Settings", padding=10)
        self.scraper_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.scraper_frame, text="Target URL:").pack(fill=tk.X)
        url_entry = ttk.Entry(self.scraper_frame, textvariable=self.url_var, font=("Consolas", 9))
        url_entry.pack(fill=tk.X, pady=(2, 8))
        
        ttk.Label(self.scraper_frame, text="Page Limit (0 = unlimited):").pack(fill=tk.X)
        ttk.Spinbox(
            self.scraper_frame, 
            from_=0, 
            to=1000, 
            textvariable=self.limit_var,
            width=10
        ).pack(fill=tk.X, pady=(2, 0))

        # Local document settings
        self.upload_frame = ttk.LabelFrame(parent, text="📂 Local Document Settings", padding=10)
        self.upload_frame.pack(fill=tk.X)
        
        btn_frame = ttk.Frame(self.upload_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(
            btn_frame, 
            text="➕ Add Files", 
            command=self.add_files
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        ttk.Button(
            btn_frame, 
            text="🗑️ Clear List", 
            command=self.clear_files
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        # File listbox with scrollbar
        listbox_frame = ttk.Frame(self.upload_frame)
        listbox_frame.pack(fill=tk.X)
        
        self.file_listbox = tk.Listbox(listbox_frame, height=4, font=("Consolas", 8))
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_research_tab(self, parent):
        """Enhanced research tab with more options"""
        # Master control
        master_frame = ttk.LabelFrame(parent, text="🎛️ Master Control", padding=10)
        master_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            master_frame, 
            text="🔬 Enable AI-Powered Research (Beta)", 
            variable=self.research_enabled_var, 
            command=self.toggle_research_panel
        ).pack(anchor=tk.W)
        
        # Research settings
        settings_frame = ttk.LabelFrame(parent, text="⚙️ Research Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="Research Queries to Generate:").pack(anchor=tk.W)
        ttk.Spinbox(
            settings_frame, 
            from_=3, 
            to=15, 
            textvariable=self.research_queries_var,
            width=10
        ).pack(fill=tk.X, pady=(2, 10))

        # Web research panel
        self.web_research_panel = ttk.LabelFrame(parent, text="🌐 Web Research", padding=10)
        self.web_research_panel.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            self.web_research_panel, 
            text="Enable Web Search Research", 
            variable=self.web_research_enabled_var
        ).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(self.web_research_panel, text="Search Method:").pack(anchor=tk.W)
        
        self.g_api_radio = ttk.Radiobutton(
            self.web_research_panel, 
            text="🚀 Google API / DuckDuckGo (Fast)", 
            variable=self.web_research_method_var, 
            value="google_api"
        )
        self.g_api_radio.pack(anchor=tk.W, padx=20)
        
        self.playwright_radio = ttk.Radiobutton(
            self.web_research_panel, 
            text="🎭 Playwright Browser (Robust)", 
            variable=self.web_research_method_var, 
            value="playwright"
        )
        self.playwright_radio.pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        ttk.Label(self.web_research_panel, text="Pages to Scrape per Search:").pack(anchor=tk.W)
        ttk.Spinbox(
            self.web_research_panel, 
            from_=3, 
            to=20, 
            textvariable=self.research_pages_var,
            width=10
        ).pack(fill=tk.X, pady=(2, 0))

        # YouTube research panel
        self.yt_research_panel = ttk.LabelFrame(parent, text="📺 YouTube Research", padding=10)
        self.yt_research_panel.pack(fill=tk.X)
        
        self.yt_checkbutton = ttk.Checkbutton(
            self.yt_research_panel, 
            text="Enable Video Transcript Analysis", 
            variable=self.yt_research_enabled_var
        )
        self.yt_checkbutton.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(self.yt_research_panel, text="Videos to Analyze per Query:").pack(anchor=tk.W)
        ttk.Spinbox(
            self.yt_research_panel, 
            from_=1, 
            to=5, 
            textvariable=self.yt_videos_per_query_var,
            width=10
        ).pack(fill=tk.X, pady=(2, 0))

    def create_ai_tab(self, parent):
        """Enhanced AI configuration tab"""
        # API Key section
        api_frame = ttk.LabelFrame(parent, text="🔑 API Configuration", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(api_frame, text="Gemini API Key File:").pack(fill=tk.X)
        key_frame = ttk.Frame(api_frame)
        key_frame.pack(fill=tk.X, pady=(2, 0))
        
        ttk.Entry(
            key_frame, 
            textvariable=self.api_key_file_var, 
            font=("Consolas", 9)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        ttk.Button(
            key_frame, 
            text="📁", 
            width=3, 
            command=self.browse_api_key
        ).pack(side=tk.RIGHT, padx=(5, 0))

        # Model settings
        model_frame = ttk.LabelFrame(parent, text="🤖 Model Settings", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model Name:").pack(fill=tk.X)
        ttk.Entry(
            model_frame, 
            textvariable=self.model_name_var, 
            font=("Consolas", 9)
        ).pack(fill=tk.X, pady=(2, 8))
        
        ttk.Label(model_frame, text=f"Temperature: {self.temperature_var.get():.1f}").pack(fill=tk.X)
        temp_scale = ttk.Scale(
            model_frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.temperature_var,
            command=self.update_temperature_label
        )
        temp_scale.pack(fill=tk.X, pady=(2, 8))
        
        ttk.Label(model_frame, text="Max Output Tokens:").pack(fill=tk.X)
        ttk.Spinbox(
            model_frame, 
            from_=1024, 
            to=32768, 
            increment=1024, 
            textvariable=self.max_tokens_var,
            width=10
        ).pack(fill=tk.X, pady=(2, 0))

    def create_prompt_tab(self, parent):
        """Enhanced prompt configuration tab"""
        # Prompt controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            control_frame, 
            text="📁 Load Prompt from File", 
            command=self.load_prompt_from_file
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        ttk.Button(
            control_frame, 
            text="🔄 Reset to Default", 
            command=self.reset_prompt_to_default
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        # Prompt editor with line numbers
        editor_frame = ttk.LabelFrame(parent, text="✏️ Prompt Editor")
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        self.prompt_text = scrolledtext.ScrolledText(
            editor_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 10),
            height=15
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_log_panel(self, parent):
        """Enhanced log panel with better formatting"""
        log_label_frame = ttk.LabelFrame(parent, text="📋 Process Logs")
        log_label_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log controls
        log_controls = ttk.Frame(log_label_frame)
        log_controls.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        ttk.Button(
            log_controls, 
            text="🗑️ Clear Logs", 
            command=self.clear_logs
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            log_controls, 
            text="💾 Save Logs", 
            command=self.save_logs
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # Log text widget
        self.log_text_widget = scrolledtext.ScrolledText(
            log_label_frame, 
            state='disabled', 
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#f8f9fa",
            fg="#333333"
        )
        self.log_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_logging(self):
        """Enhanced logging setup"""
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        
        # Set up formatter
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.queue_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.getLogger().addHandler(self.queue_handler)
        logging.getLogger().setLevel(logging.INFO)
        
        # Start log polling
        self.root.after(100, self.poll_log_queue)

    # --- Helper Methods ---
    
    def update_temperature_label(self, value):
        """Update temperature label when scale changes"""
        temp_value = float(value)
        # Find the temperature label and update it
        for widget in self.root.winfo_children():
            if hasattr(widget, 'winfo_children'):
                self._update_temp_label_recursive(widget, temp_value)

    def _update_temp_label_recursive(self, parent, temp_value):
        """Recursively find and update temperature label"""
        for child in parent.winfo_children():
            if isinstance(child, ttk.Label) and "Temperature:" in str(child.cget('text')):
                child.configure(text=f"Temperature: {temp_value:.1f}")
                return
            if hasattr(child, 'winfo_children'):
                self._update_temp_label_recursive(child, temp_value)

    def clear_logs(self):
        """Clear the log display"""
        self.log_text_widget.config(state='normal')
        self.log_text_widget.delete('1.0', tk.END)
        self.log_text_widget.config(state='disabled')

    def save_logs(self):
        """Save current logs to file"""
        log_content = self.log_text_widget.get('1.0', tk.END)
        if not log_content.strip():
            messagebox.showinfo("No Logs", "No logs to save.")
            return
            
        filename = f"research_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = filedialog.asksaveasfilename(
            initialfile=filename,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                messagebox.showinfo("Success", f"Logs saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs:\n{e}")

    def reset_prompt_to_default(self):
        """Reset prompt to default template"""
        default_prompt = self._load_prompt_file("prompt.md")
        if default_prompt:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, default_prompt)
        else:
            messagebox.showwarning("No Default", "No default prompt file (prompt.md) found.")

    # --- Existing Methods (Enhanced) ---
    
    def _set_child_widgets_state(self, parent, state):
        """Recursively set widget states"""
        for widget in parent.winfo_children():
            try: 
                widget.configure(state=state)
            except tk.TclError: 
                pass
            self._set_child_widgets_state(widget, state)

    def toggle_input_mode(self):
        """Toggle between scrape and upload modes"""
        mode = self.input_mode_var.get()
        if mode == "scrape":
            self._set_child_widgets_state(self.scraper_frame, tk.NORMAL)
            self._set_child_widgets_state(self.upload_frame, tk.DISABLED)
        else:
            self._set_child_widgets_state(self.scraper_frame, tk.DISABLED)
            self._set_child_widgets_state(self.upload_frame, tk.NORMAL)

    def add_files(self):
        """Add files to the upload list"""
        filetypes = [
            ("All Supported", "*.pdf *.docx *.txt"), 
            ("PDF", "*.pdf"), 
            ("Word", "*.docx"), 
            ("Text", "*.txt")
        ]
        files = filedialog.askopenfilenames(title="Select Documents", filetypes=filetypes)
        for f in files:
            if f not in self.file_listbox.get(0, tk.END): 
                self.file_listbox.insert(tk.END, f)
    
    def clear_files(self): 
        """Clear the file list"""
        self.file_listbox.delete(0, tk.END)
    
    def browse_api_key(self):
        """Browse for API key file"""
        filepath = filedialog.askopenfilename(
            title="Select Gemini API Key File",
            filetypes=[("Key files", "*.key"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath: 
            self.api_key_file_var.set(filepath)
    
    def load_prompt_from_file(self):
        """Load prompt from file"""
        filepath = filedialog.askopenfilename(
            title="Select Prompt File", 
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert(tk.END, content)
                logging.info(f"Loaded prompt from: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load prompt:\n{e}")
            
    def toggle_research_panel(self):
        """Toggle research panel based on master checkbox"""
        state = tk.NORMAL if self.research_enabled_var.get() else tk.DISABLED
        self._set_child_widgets_state(self.web_research_panel, state)
        self._set_child_widgets_state(self.yt_research_panel, state)
        
        # Handle dependency-specific disabling
        if not PLAYWRIGHT_AVAILABLE: 
            self.playwright_radio.config(state=tk.DISABLED)
        if not YT_DLP_AVAILABLE: 
            self.yt_checkbutton.config(state=tk.DISABLED)

    def load_initial_settings(self):
        """Load settings from config.yml and prompt.md"""
        self.config = self._load_config_file()
        prompt_template = self._load_prompt_file()
        
        if self.config:
            logging.info("📝 Loading settings from config.yml...")
            api_settings = self.config.get('api', {})
            llm_settings = self.config.get('llm', {})
            llm_params = llm_settings.get('parameters', {})
            
            # Load API settings
            self.api_key_file_var.set(api_settings.get('key_file', 'gemini_api.key'))
            self.model_name_var.set(llm_settings.get('model_name', 'gemini-1.5-flash'))
            self.temperature_var.set(llm_params.get('temperature', 0.5))
            self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
            
            # Load Google API settings
            google_search_settings = api_settings.get('google_search', {})
            self.google_api_key_file_var.set(google_search_settings.get('key_file', 'google_api.key'))
            self.google_cx_file_var.set(google_search_settings.get('cx_file', 'google_cx.key'))

            logging.info("✅ Configuration loaded successfully")
        else:
            logging.warning("⚠️ Could not load config.yml. Using defaults.")

        # Load default prompt
        if prompt_template:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, prompt_template)
            logging.info("✅ Default prompt loaded from prompt.md")
        else:
            # Set a basic default prompt if file doesn't exist
            default_prompt = """You are an expert educational content creator. Generate a comprehensive study guide based on the provided content.

Structure your response with:
1. **Executive Summary** - Key takeaways in 2-3 paragraphs
2. **Main Topics** - Detailed sections covering core concepts
3. **Key Points** - Important facts and principles
4. **Examples** - Practical applications and use cases
5. **Common Issues** - Problems and troubleshooting
6. **Best Practices** - Recommended approaches
7. **Further Learning** - Next steps and resources

Content to analyze:
{content}

Source: {website_url}
"""
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, default_prompt)
        
        # Check dependencies and warn user
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for optional dependencies and update UI accordingly"""
        if not PLAYWRIGHT_AVAILABLE:
            self.playwright_radio.config(state=tk.DISABLED)
            if self.web_research_method_var.get() == 'playwright':
                self.web_research_method_var.set('google_api')
            logging.warning("⚠️ Playwright not available. Install with: pip install playwright && playwright install")
            
        if not YT_DLP_AVAILABLE:
            self.yt_checkbutton.config(state=tk.DISABLED)
            self.yt_research_enabled_var.set(False)
            logging.warning("⚠️ yt-dlp not available. Install with: pip install yt-dlp")

    def poll_log_queue(self):
        """Enhanced log polling with better formatting"""
        try:
            while True:
                try: 
                    record = self.log_queue.get(block=False)
                except queue.Empty: 
                    break
                else:
                    self.log_text_widget.config(state='normal')
                    
                    # Add color coding for different log levels
                    if "ERROR" in record or "❌" in record:
                        self.log_text_widget.insert(tk.END, record + '\n', 'error')
                    elif "WARNING" in record or "⚠️" in record:
                        self.log_text_widget.insert(tk.END, record + '\n', 'warning')
                    elif "✅" in record or "SUCCESS" in record:
                        self.log_text_widget.insert(tk.END, record + '\n', 'success')
                    else:
                        self.log_text_widget.insert(tk.END, record + '\n')
                    
                    self.log_text_widget.config(state='disabled')
                    self.log_text_widget.yview(tk.END)
        except Exception as e:
            pass  # Ignore polling errors
        finally:
            self.root.after(100, self.poll_log_queue)
    
    def save_notes(self):
        """Enhanced save functionality with better file naming"""
        if not self.final_notes_content: 
            messagebox.showwarning("No Content", "No study guide to save.")
            return
            
        # Generate intelligent filename
        if self.input_mode_var.get() == "scrape":
            source_name = urlparse(self.url_var.get()).netloc or "website"
        else:
            source_name = "local_docs"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        initial_filename = f"study_guide_{source_name}_{timestamp}.md"
        
        filepath = filedialog.asksaveasfilename(
            initialfile=initial_filename, 
            defaultextension=".md", 
            filetypes=[
                ("Markdown", "*.md"), 
                ("Text", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath: 
            logging.info("💾 Save cancelled by user")
            return
            
        try:
            with open(filepath, "w", encoding="utf-8") as f: 
                f.write(self.final_notes_content)
            logging.info(f"💾 Study guide saved successfully: {filepath}")
            messagebox.showinfo("Success", f"Study guide saved to:\n{filepath}")
        except IOError as e: 
            logging.error(f"❌ Failed to save file: {e}")
            messagebox.showerror("Save Error", f"Could not save file:\n\n{e}")

    def start_task_thread(self):
        """Enhanced task starting with better state management"""
        if self.is_processing:
            messagebox.showwarning("Processing", "A task is already running. Please wait for it to complete.")
            return
            
        # Reset state and UI
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED, text="⏳ Processing...")
        self.save_button.config(state=tk.DISABLED)
        self.final_notes_content = ""
        self.progress_var.set("Initializing...")
        
        # Clear previous logs
        self.clear_logs()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.run_full_process, daemon=True)
        processing_thread.start()

    def run_full_process(self):
        """Enhanced main processing function with better error handling and progress tracking"""
        try:
            # Validate inputs
            api_key_file = self.api_key_file_var.get()
            model_name = self.model_name_var.get() 
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            
            if not all([api_key_file, model_name, prompt]):
                messagebox.showerror("Input Error", "API Key File, Model Name, and Prompt must be configured.")
                return
            
            # Load API key
            self.progress_var.set("Loading API credentials...")
            api_key = APIKeyManager.get_gemini_api_key(api_key_file)
            if not api_key: 
                logging.error("❌ Invalid API key. Process halted.")
                messagebox.showerror("API Key Error", "The Gemini API key is missing or invalid.")
                return

            # Initialize variables
            source_data = {}
            source_name = ""
            mode = self.input_mode_var.get()
            delay = self.config.get('scraper', {}).get('rate_limit_delay', 0.5)
            user_agent = random.choice(USER_AGENTS)
            
            # Phase 1: Initial Content Collection
            self.progress_var.set("Collecting initial content...")
            logging.info("🚀 Starting Enhanced Research & Study Guide Generation")
            logging.info("="*60)
            
            if mode == "scrape":
                logging.info("📄 Mode: Web Scraping")
                url = self.url_var.get().strip()
                if not url: 
                    messagebox.showerror("Input Error", "URL cannot be empty.")
                    return
                    
                # Add protocol if missing
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                    self.url_var.set(url)
                    
                limit = self.limit_var.get()
                logging.info(f"🎯 Target: {url}")
                logging.info(f"📊 Page limit: {limit if limit > 0 else 'Unlimited'}")
                
                scraper = WebsiteScraper(url, limit, user_agent, 15, delay)
                source_data = scraper.crawl()
                source_name = urlparse(url).netloc
                
            else:  # upload mode
                logging.info("📁 Mode: Local Document Processing")
                file_paths = list(self.file_listbox.get(0, tk.END))
                if not file_paths: 
                    messagebox.showerror("Input Error", "Please add at least one document.")
                    return
                    
                logging.info(f"📚 Processing {len(file_paths)} local documents")
                loader = LocalDocumentLoader(file_paths)
                source_data = loader.load_and_extract_text()
                source_name = f"{len(file_paths)}_local_documents"

            if not source_data:
                logging.warning("⚠️ No content extracted from initial source. Cannot proceed.")
                messagebox.showwarning("No Content", "No text content could be extracted from the source.")
                return

            logging.info(f"✅ Initial content collection complete: {len(source_data)} sources")

            # Phase 2: AI Research (if enabled)
            if self.research_enabled_var.get():
                self.progress_var.set("Conducting AI research...")
                logging.info("\n🔬 Starting AI Research Phase")
                logging.info("-" * 40)
                
                # Generate research queries
                query_generator = EnhancedResearchQueryGenerator(api_key)
                topic_data = query_generator.extract_main_topic_and_subtopics(source_data)
                research_queries = query_generator.generate_diverse_research_queries(
                    topic_data, 
                    self.research_queries_var.get()
                )
                
                logging.info(f"🎯 Generated {len(research_queries)} research queries")
                for i, query in enumerate(research_queries, 1):
                    logging.info(f"  {i}. {query}")

                # Web research
                if self.web_research_enabled_var.get():
                    self.progress_var.set("Researching web sources...")
                    logging.info("\n🌐 Starting Web Research")
                    
                    exclude_domain = source_name if mode == "scrape" else ""
                    
                    if self.web_research_method_var.get() == "playwright":
                        if PLAYWRIGHT_AVAILABLE:
                            researcher = EnhancedPlaywrightResearcher()
                        else:
                            logging.warning("⚠️ Playwright not available, falling back to API method")
                            google_api_key, google_cx = APIKeyManager.get_google_search_credentials(
                                self.google_api_key_file_var.get(), 
                                self.google_cx_file_var.get()
                            )
                            researcher = EnhancedGoogleSearchResearcher(google_api_key, google_cx)
                    else:
                        google_api_key, google_cx = APIKeyManager.get_google_search_credentials(
                            self.google_api_key_file_var.get(), 
                            self.google_cx_file_var.get()
                        )
                        researcher = EnhancedGoogleSearchResearcher(google_api_key, google_cx)
                    
                    research_urls = researcher.search_and_extract_urls(research_queries, exclude_domain)
                    
                    if research_urls:
                        urls_to_scrape = research_urls[:self.research_pages_var.get()]
                        logging.info(f"🕷️ Scraping top {len(urls_to_scrape)} research URLs")
                        
                        research_scraper = WebsiteScraper(
                            "http://research.local", 
                            len(urls_to_scrape), 
                            user_agent, 
                            15, 
                            delay
                        )
                        research_data = research_scraper.crawl(additional_urls=urls_to_scrape)
                        
                        if research_data:
                            source_data.update(research_data)
                            logging.info(f"✅ Added {len(research_data)} web research sources")
                    else:
                        logging.warning("⚠️ No research URLs found from web search")

                # YouTube research
                if self.yt_research_enabled_var.get():
                    self.progress_var.set("Analyzing YouTube videos...")
                    logging.info("\n📺 Starting YouTube Research")
                    
                    if YT_DLP_AVAILABLE:
                        try:
                            yt_researcher = EnhancedYouTubeResearcher()
                            video_transcripts = yt_researcher.get_transcripts_for_queries(
                                research_queries, 
                                self.yt_videos_per_query_var.get()
                            )
                            
                            if video_transcripts:
                                source_data.update(video_transcripts)
                                logging.info(f"✅ Added {len(video_transcripts)} YouTube transcript sources")
                            else:
                                logging.warning("⚠️ No suitable YouTube videos found")
                                
                        except Exception as e:
                            logging.error(f"❌ YouTube research failed: {e}")
                    else:
                        logging.warning("⚠️ YouTube research skipped - yt-dlp not available")

            # Phase 3: Study Guide Generation
            self.progress_var.set("Generating study guide...")
            logging.info(f"\n📝 Starting Study Guide Generation")
            logging.info("-" * 40)
            logging.info(f"📊 Total sources: {len(source_data)}")
            
            # Categorize sources for better logging
            web_sources = sum(1 for url in source_data.keys() if url.startswith('http') and 'youtube.com' not in url)
            video_sources = sum(1 for url in source_data.keys() if 'youtube.com' in url)
            local_sources = sum(1 for url in source_data.keys() if not url.startswith('http'))
            
            logging.info(f"  📄 Local documents: {local_sources}")
            logging.info(f"  🌐 Web pages: {web_sources}")
            logging.info(f"  📺 Video transcripts: {video_sources}")

            # Configure LLM
            llm_config = {
                'model_name': model_name,
                'parameters': {
                    'temperature': self.temperature_var.get(),
                    'max_output_tokens': self.max_tokens_var.get()
                }
            }

            # Generate notes
            generator = EnhancedNoteGenerator(api_key, llm_config, prompt)
            self.final_notes_content = generator.generate_comprehensive_notes(source_data, source_name)
            
            # Success!
            self.progress_var.set("Generation complete!")
            logging.info("\n🎉 Study Guide Generation Complete!")
            logging.info("="*60)
            logging.info("📋 Summary:")
            logging.info(f"  • Total sources processed: {len(source_data)}")
            logging.info(f"  • Study guide length: {len(self.final_notes_content)} characters")
            logging.info("  • Ready to save!")
            
            self.save_button.config(state=tk.NORMAL)
            messagebox.showinfo(
                "Success!", 
                f"Study guide generated successfully!\n\n"
                f"Sources processed: {len(source_data)}\n"
                f"Content length: {len(self.final_notes_content):,} characters\n\n"
                f"Click 'Save Study Guide' to export your notes."
            )

        except Exception as e:
            logging.error(f"❌ Unexpected error: {e}", exc_info=True)
            messagebox.showerror(
                "Error", 
                f"An unexpected error occurred:\n\n{e}\n\n"
                f"Check the logs for more details."
            )
        finally:
            # Reset UI state
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation")
            if not self.final_notes_content:
                self.progress_var.set("Ready")
            else:
                self.progress_var.set("Complete - Ready to save!")
            
    def _load_config_file(self, filepath="config.yml"):
        """Load configuration from YAML file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.warning(f"Could not load '{filepath}': {e}")
            return {}
        
    def _load_prompt_file(self, filepath="prompt.md"):
        """Load prompt template from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"Prompt file '{filepath}' not found")
            return ""

def main():
    """Main application entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    # Create and run application
    root = tk.Tk()
    
    # Set application icon if available
    try:
        root.iconbitmap('icon.ico')  # Add an icon file if you have one
    except:
        pass
    
    app = AdvancedScraperApp(root)
    
    # Handle window closing
    def on_closing():
        if app.is_processing:
            if messagebox.askokcancel("Quit", "A task is running. Do you want to quit anyway?"):
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the application
    root.mainloop()

if __name__ == '__main__':
    main()
