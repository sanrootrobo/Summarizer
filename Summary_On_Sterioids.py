import os
import requests
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
from concurrent.futures import ThreadPoolExecutor
import re
import argparse
from time import sleep
import json
from datetime import datetime
import html2text

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "gemini-2.5-pro"
SEARCH_MODEL_NAME = "gemini-2.5-flash"  # Lightweight model for query generation
API_KEY_FILE = "geminaikey"
GOOGLE_API_KEY_FILE = "googlesearchapi"
GOOGLE_CX_FILE = "googlecx"
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
MAX_CONTENT_CHARS = 500000
MAX_SEARCH_RESULTS = 10  # Maximum results per search query
MAX_RESEARCH_PAGES = 20  # Maximum additional pages to crawl from search

class APIKeyManager:
    @staticmethod
    def get_gemini_api_key(filepath: str = "geminaikey") -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"Gemini API key file not found at '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10:
                logger.error("Invalid Gemini API key found in file.")
                return None
            logger.info("Gemini API key loaded successfully.")
            return api_key
        except Exception as e:
            logger.error(f"Error reading Gemini API key: {e}")
            return None
    
    @staticmethod
    def get_google_search_credentials() -> Tuple[Optional[str], Optional[str]]:
        """Get Google Search API key and Custom Search Engine ID"""
        try:
            # Get API Key
            api_key_path = Path(GOOGLE_API_KEY_FILE)
            if not api_key_path.exists():
                logger.warning(f"Google Search API key file not found at '{GOOGLE_API_KEY_FILE}'. Skipping research enhancement.")
                return None, None
            
            with open(api_key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            
            # Get Custom Search Engine ID
            cx_path = Path(GOOGLE_CX_FILE)
            if not cx_path.exists():
                logger.warning(f"Google CX file not found at '{GOOGLE_CX_FILE}'. Skipping research enhancement.")
                return None, None
                
            with open(cx_path, 'r', encoding='utf-8') as f:
                cx_id = f.read().strip()
            
            if not api_key or len(api_key) < 10 or not cx_id:
                logger.warning("Invalid Google Search credentials. Skipping research enhancement.")
                return None, None
                
            logger.info("Google Search API credentials loaded successfully.")
            return api_key, cx_id
            
        except Exception as e:
            logger.warning(f"Error reading Google Search credentials: {e}. Skipping research enhancement.")
            return None, None

class ResearchQueryGenerator:
    """Generate intelligent research queries using Gemini Flash"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model=SEARCH_MODEL_NAME,
            google_api_key=api_key,
            temperature=0.7,  # Higher creativity for diverse queries
            max_output_tokens=2048
        )
        self.output_parser = StrOutputParser()
    
    def generate_research_queries(self, scraped_content: Dict[str, str], metadata: Dict[str, Dict], main_url: str) -> List[str]:
        """Generate intelligent research queries based on scraped content"""
        
        # Analyze content to understand the main topics and gaps
        content_summary = self._create_content_summary(scraped_content, metadata)
        
        prompt = ChatPromptTemplate.from_template("""
You are a research assistant specializing in comprehensive documentation analysis. Your task is to generate intelligent Google search queries that will help find additional resources, tutorials, examples, and explanations to enhance the study materials.

**Content Analysis:**
Main Website: {main_url}
Content Summary: {content_summary}

**Your Mission:**
Generate 8-12 strategic Google search queries that will find:
1. Alternative explanations and tutorials
2. Practical examples and use cases  
3. Best practices and patterns
4. Common problems and solutions
5. Comparison with alternatives
6. Advanced techniques and tips
7. Community discussions and insights
8. Official resources not yet covered

**Query Generation Rules:**
- Each query should be 3-8 words long
- Focus on finding content that COMPLEMENTS the existing documentation
- Include queries for different skill levels (beginner, intermediate, advanced)
- Mix specific technical terms with broader conceptual searches
- Include "tutorial", "example", "guide", "best practices" etc. where appropriate
- Avoid queries that would just return the original documentation site

**Output Format:**
Return ONLY a JSON array of search queries, no other text:
["query 1", "query 2", "query 3", ...]

**Example Output:**
["react hooks tutorial examples", "javascript async await best practices", "node.js error handling patterns", "react performance optimization guide"]

Generate research queries:
""")
        
        chain = prompt | self.llm | self.output_parser
        
        try:
            logger.info("Generating intelligent research queries using Gemini Flash...")
            result = chain.invoke({
                "main_url": main_url,
                "content_summary": content_summary
            })
            
            # Parse JSON response
            queries = json.loads(result.strip())
            if isinstance(queries, list):
                logger.info(f"Generated {len(queries)} research queries")
                return queries[:12]  # Limit to prevent excessive API calls
            else:
                logger.warning("Unexpected query format returned")
                return []
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse generated queries - using fallback queries")
            return self._generate_fallback_queries(main_url, content_summary)
        except Exception as e:
            logger.error(f"Error generating research queries: {e}")
            return self._generate_fallback_queries(main_url, content_summary)
    
    def _create_content_summary(self, scraped_content: Dict[str, str], metadata: Dict[str, Dict]) -> str:
        """Create a concise summary of scraped content for query generation"""
        topics = set()
        content_types = set()
        
        for url, meta in metadata.items():
            analysis = meta.get('analysis', {})
            content_types.add(analysis.get('type', 'general'))
            
            # Extract key terms from titles and URLs
            title = meta.get('title', '')
            url_parts = url.lower().split('/')
            
            for part in url_parts + title.lower().split():
                if len(part) > 3 and part.isalpha():
                    topics.add(part)
        
        summary = f"Content types: {', '.join(content_types)}. "
        summary += f"Key topics: {', '.join(list(topics)[:10])}. "
        summary += f"Total pages: {len(scraped_content)}"
        
        return summary
    
    def _generate_fallback_queries(self, main_url: str, content_summary: str) -> List[str]:
        """Generate basic fallback queries if AI generation fails"""
        domain = urlparse(main_url).netloc.replace('www.', '')
        base_topic = domain.split('.')[0]
        
        return [
            f"{base_topic} tutorial examples",
            f"{base_topic} best practices guide",
            f"{base_topic} common mistakes solutions",
            f"{base_topic} advanced techniques",
            f"how to use {base_topic} beginners",
            f"{base_topic} vs alternatives comparison",
            f"{base_topic} troubleshooting guide",
            f"{base_topic} performance optimization"
        ]

class GoogleSearchResearcher:
    """Perform Google searches and extract useful URLs with DuckDuckGo fallback"""
    
    def __init__(self, api_key: str = None, cx_id: str = None):
        self.api_key = api_key
        self.cx_id = cx_id
        self.google_base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
    
    def search_and_extract_urls(self, queries: List[str], exclude_domain: str = None) -> List[str]:
        """Perform searches using Google first, fallback to DuckDuckGo if needed"""
        all_urls = set()
        google_available = bool(self.api_key and self.cx_id)
        
        for i, query in enumerate(queries):
            logger.info(f"🔍 Searching [{i+1}/{len(queries)}]: {query}")
            
            urls_found = False
            
            # Try Google Search first if credentials available
            if google_available:
                try:
                    google_urls = self._search_google(query, exclude_domain)
                    if google_urls:
                        all_urls.update(google_urls)
                        urls_found = True
                        logger.info(f"✅ Google found {len(google_urls)} URLs for: {query}")
                    else:
                        logger.info(f"⚠️ Google returned no results for: {query}")
                except Exception as e:
                    logger.warning(f"❌ Google search failed for '{query}': {e}")
            
            # Fallback to DuckDuckGo if Google failed or not available
            if not urls_found:
                try:
                    logger.info(f"🦆 Using DuckDuckGo fallback for: {query}")
                    ddg_urls = self._search_duckduckgo(query, exclude_domain)
                    if ddg_urls:
                        all_urls.update(ddg_urls)
                        logger.info(f"✅ DuckDuckGo found {len(ddg_urls)} URLs for: {query}")
                    else:
                        logger.info(f"⚠️ DuckDuckGo returned no results for: {query}")
                except Exception as e:
                    logger.warning(f"❌ DuckDuckGo search also failed for '{query}': {e}")
            
            # Be polite to search engines
            sleep(0.5)
        
        filtered_urls = list(all_urls)[:MAX_RESEARCH_PAGES]
        logger.info(f"🎯 Total research URLs collected: {len(filtered_urls)} from {len(queries)} searches")
        return filtered_urls
    
    def _search_google(self, query: str, exclude_domain: str = None) -> List[str]:
        """Search using Google Custom Search API"""
        try:
            params = {
                'key': self.api_key,
                'cx': self.cx_id,
                'q': query,
                'num': min(MAX_SEARCH_RESULTS, 10)
            }
            
            response = self.session.get(self.google_base_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            results = response.json()
            urls = []
            
            if 'items' in results:
                for item in results['items']:
                    url = item.get('link', '')
                    if self._is_useful_url(url, exclude_domain):
                        urls.append(url)
            
            return urls
            
        except requests.RequestException as e:
            raise Exception(f"Google API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from Google: {e}")
    
    def _search_duckduckgo(self, query: str, exclude_domain: str = None) -> List[str]:
        """Search using DuckDuckGo HTML scraping (no API required)"""
        try:
            # DuckDuckGo instant answer API (limited but free)
            ddg_url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'b': '',  # Start from beginning
                'kl': 'us-en',  # Language
                'df': '',  # Date filter
                's': '0',  # Start index
                'v': 'l',  # Layout
                'o': 'json',  # Output format
                'dc': '1'  # More results
            }
            
            # Add search operators for better results
            enhanced_query = f"{query} site:stackoverflow.com OR site:github.com OR site:medium.com OR site:dev.to"
            params['q'] = enhanced_query
            
            response = self.session.get(ddg_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            urls = []
            
            # Parse DuckDuckGo results
            result_links = soup.find_all('a', {'class': 'result__a'})
            
            for link in result_links[:MAX_SEARCH_RESULTS]:
                href = link.get('href', '')
                if href.startswith('/l/?uddg='):
                    # DuckDuckGo wraps URLs, extract the real URL
                    try:
                        # Parse the wrapped URL
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        if 'uddg' in parsed:
                            real_url = urllib.parse.unquote(parsed['uddg'][0])
                            if self._is_useful_url(real_url, exclude_domain):
                                urls.append(real_url)
                    except Exception:
                        continue
                elif href.startswith('http'):
                    # Direct URL
                    if self._is_useful_url(href, exclude_domain):
                        urls.append(href)
            
            # If no results with enhanced query, try simple query
            if not urls and enhanced_query != query:
                params['q'] = query
                response = self.session.get(ddg_url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                result_links = soup.find_all('a', {'class': 'result__a'})
                
                for link in result_links[:MAX_SEARCH_RESULTS]:
                    href = link.get('href', '')
                    if href.startswith('/l/?uddg='):
                        try:
                            import urllib.parse
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                            if 'uddg' in parsed:
                                real_url = urllib.parse.unquote(parsed['uddg'][0])
                                if self._is_useful_url(real_url, exclude_domain):
                                    urls.append(real_url)
                        except Exception:
                            continue
                    elif href.startswith('http'):
                        if self._is_useful_url(href, exclude_domain):
                            urls.append(href)
            
            return urls[:MAX_SEARCH_RESULTS]
            
        except requests.RequestException as e:
            raise Exception(f"DuckDuckGo request failed: {e}")
        except Exception as e:
            raise Exception(f"DuckDuckGo parsing failed: {e}")
    
    def _is_useful_url(self, url: str, exclude_domain: str = None) -> bool:
        """Filter URLs to keep only useful ones for research"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Exclude the original domain if specified
            if exclude_domain and exclude_domain.lower() in domain:
                return False
            
            # Exclude certain file types
            excluded_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.zip', '.tar', '.gz', '.mp3', '.mp4', '.avi']
            if any(path.endswith(ext) for ext in excluded_extensions):
                return False
            
            # Exclude certain domains that typically don't have useful content
            excluded_domains = [
                'youtube.com', 'youtu.be', 'twitter.com', 'x.com', 'facebook.com', 'linkedin.com',
                'pinterest.com', 'instagram.com', 'tiktok.com',  # Social media
                'amazon.com', 'ebay.com', 'shopping.google.com', 'aliexpress.com',  # Shopping
                'ads.google.com', 'doubleclick.net', 'googleadservices.com',  # Ads
                'wikipedia.org'  # Sometimes too general
            ]
            
            if any(excluded in domain for excluded in excluded_domains):
                return False
            
            # Prefer certain high-quality domains
            preferred_domains = [
                'stackoverflow.com', 'github.com', 'medium.com', 'dev.to',
                'hackernoon.com', 'freecodecamp.org', 'css-tricks.com',
                'smashingmagazine.com', 'tutorialspoint.com', 'geeksforgeeks.org'
            ]
            
            preferred_path_indicators = [
                'tutorial', 'guide', 'docs', 'documentation', 'wiki',
                'blog', 'learn', 'course', 'example', 'howto', 'getting-started'
            ]
            
            # Give bonus points to preferred domains/paths
            if (any(preferred in domain for preferred in preferred_domains) or
                any(indicator in path for indicator in preferred_path_indicators)):
                return True
            
            # General filtering - must be HTTP/HTTPS and have reasonable length
            if parsed.scheme not in ['http', 'https']:
                return False
            
            if len(url) > 500:  # Extremely long URLs are usually not useful
                return False
                
            return True
            
        except Exception:
            return False

class ContentAnalyzer:
    """Analyzes scraped content to categorize and prioritize information"""
    
    @staticmethod
    def categorize_content(url: str, content: str) -> Dict[str, any]:
        """Categorize content based on URL patterns and content analysis"""
        categories = {
            'type': 'general',
            'priority': 'medium',
            'contains_code': False,
            'contains_examples': False,
            'is_api_reference': False,
            'is_tutorial': False,
            'is_concept': False,
            'is_research': False,  # New flag for research content
            'word_count': len(content.split())
        }
        
        url_lower = url.lower()
        content_lower = content.lower()
        
        # Check if this is research content (from Google search)
        research_indicators = [
            'stackoverflow.com', 'github.com', 'medium.com', 'dev.to',
            '/tutorial', '/guide', '/example', '/blog'
        ]
        
        if any(indicator in url_lower for indicator in research_indicators):
            categories['is_research'] = True
            categories['priority'] = 'high'  # Research content is valuable
        
        # Categorize by URL patterns
        if any(pattern in url_lower for pattern in ['api', 'reference', 'docs/api']):
            categories['type'] = 'api_reference'
            categories['is_api_reference'] = True
            categories['priority'] = 'high'
        elif any(pattern in url_lower for pattern in ['tutorial', 'guide', 'getting-started', 'quickstart']):
            categories['type'] = 'tutorial'
            categories['is_tutorial'] = True
            categories['priority'] = 'high'
        elif any(pattern in url_lower for pattern in ['concept', 'overview', 'introduction', 'fundamentals']):
            categories['type'] = 'concept'
            categories['is_concept'] = True
            categories['priority'] = 'high'
        elif any(pattern in url_lower for pattern in ['example', 'sample', 'demo']):
            categories['type'] = 'example'
            categories['contains_examples'] = True
            categories['priority'] = 'medium'
        
        # Analyze content
        if any(pattern in content_lower for pattern in ['```', 'code', 'function', 'class', 'import', 'def ']):
            categories['contains_code'] = True
        
        if any(pattern in content_lower for pattern in ['example', 'for instance', 'here\'s how', 'demo']):
            categories['contains_examples'] = True
            
        return categories

class WebsiteScraper:
    def __init__(self, base_url: str, max_pages: int):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.scraped_content = {}
        self.content_metadata = {}
        self.visited_urls = set()
        self.text_maker = html2text.HTML2Text()
        self.text_maker.body_width = 0
        self.text_maker.ignore_links = False
        self.text_maker.ignore_images = True

    def _get_page_content(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
            return None
        except requests.RequestException as e:
            logger.warning(f"Could not fetch {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from various sources"""
        title = soup.find('title')
        if title:
            return title.get_text().strip()
        
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
            
        return "Untitled Page"

    def _parse_content_and_links(self, html: str, page_url: str) -> Tuple[str, List[str], str]:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = self._extract_title(soup)
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Find main content with better priority
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', {'role': 'main'}) or 
            soup.find('div', class_=re.compile(r'content|main|body|documentation')) or 
            soup.find('section', class_=re.compile(r'content|main|docs')) or
            soup.body
        )
        
        if main_content:
            markdown_content = self.text_maker.handle(str(main_content))
        else:
            markdown_content = ""
        
        # Extract links only if we're crawling the same domain
        links = set()
        current_domain = urlparse(page_url).netloc
        if current_domain == self.domain:  # Only extract links from original domain
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(page_url, href)
                parsed_url = urlparse(full_url)
                
                if (parsed_url.netloc == self.domain and 
                    not any(ext in parsed_url.path.lower() for ext in ['.pdf', '.jpg', '.png', '.zip', '.doc', '.mp4'])):
                    clean_url = full_url.split('#')[0].split('?')[0]  # Remove fragments and query params
                    links.add(clean_url)
        
        return markdown_content, list(links), title

    def crawl(self, additional_urls: List[str] = None) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """Crawl the main site and optionally additional research URLs"""
        urls_to_visit = {self.base_url}
        
        # Add research URLs if provided
        if additional_urls:
            logger.info(f"🔬 Adding {len(additional_urls)} research URLs to crawl queue")
            urls_to_visit.update(additional_urls)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            while urls_to_visit:
                if self.max_pages > 0 and len(self.visited_urls) >= self.max_pages:
                    logger.info(f"Reached scraping limit of {self.max_pages} pages. Stopping crawl.")
                    break

                url = urls_to_visit.pop()
                if url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(url)
                
                # Different progress indicators for main vs research content
                current_domain = urlparse(url).netloc
                is_research = current_domain != self.domain
                prefix = "🔬" if is_research else "📄"
                
                progress_prefix = f"[{len(self.visited_urls)}/{self.max_pages}]" if self.max_pages > 0 else f"[{len(self.visited_urls)}]"
                logger.info(f"{progress_prefix} {prefix} Scraping: {url}")
                
                html_content = self._get_page_content(url)
                if not html_content:
                    continue
                
                text, new_links, title = self._parse_content_and_links(html_content, url)
                
                if text and len(text.strip()) > 100:
                    self.scraped_content[url] = text
                    
                    # Analyze and store metadata
                    self.content_metadata[url] = {
                        'title': title,
                        'analysis': ContentAnalyzer.categorize_content(url, text),
                        'scraped_at': datetime.now().isoformat(),
                        'is_research': is_research
                    }
                    
                # Only follow links from the main domain to avoid infinite research crawling
                if not is_research:
                    for link in new_links:
                        if link not in self.visited_urls:
                            urls_to_visit.add(link)
                
                sleep(0.3)  # Be polite to servers
        
        main_pages = sum(1 for meta in self.content_metadata.values() if not meta.get('is_research', False))
        research_pages = sum(1 for meta in self.content_metadata.values() if meta.get('is_research', False))
        
        logger.info(f"Crawling complete. Scraped {main_pages} main pages + {research_pages} research pages = {len(self.scraped_content)} total pages")
        return self.scraped_content, self.content_metadata

class EnhancedNoteGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            google_api_key=api_key, 
            temperature=0.3,
            max_output_tokens=8192
        )
        self.output_parser = StrOutputParser()

    def _create_comprehensive_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
You are an expert technical documentation analyst and educational content creator. Your task is to transform raw documentation AND research materials into comprehensive, multi-layered study materials.

**Your Mission:**
Create exhaustive study notes that serve multiple learning styles and proficiency levels. You have access to both official documentation and supplementary research materials from various sources. Think of this as creating a complete learning resource that could replace multiple sources.

**Content to Process:**
Website: {website_url}
Documentation + Research Content: {content}

**Content Sources Available:**
- Official documentation pages
- Community tutorials and guides  
- Stack Overflow discussions
- GitHub examples and repositories
- Blog posts and articles
- Best practices and patterns

**Required Output Structure:**

# 📚 Comprehensive Study Guide: [Main Topic]

## 📋 Table of Contents
[Generate a detailed TOC with page numbers/sections]

---

## 🎯 Executive Summary
**What is this?** [One-sentence description]
**Who needs this?** [Target audience]  
**Key Benefits:** [3-5 bullet points]
**Time to Master:** [Realistic estimate]
**Learning Prerequisites:** [What you should know first]

---

## 🧠 Core Concepts Explained

### 🔤 ELI5 (Explain Like I'm 5) Section
[Break down complex concepts using simple analogies, metaphors, and everyday examples. Use storytelling where appropriate.]

### 🏗️ Architecture & How It Works
[Technical explanation with diagrams described in text, flow charts, system interactions]

### 🎭 Real-World Analogies
[Multiple analogies comparing technical concepts to familiar real-world scenarios]

### 🤔 Common Misconceptions
[Address frequent misunderstandings with clear corrections]

---

## 📖 Detailed Reference Guide

### 📚 Complete API/Feature Reference
[Organized by category, with every endpoint/feature documented]

### 🛠️ Configuration & Setup  
[Step-by-step instructions, common configurations, troubleshooting]

### 🔧 Advanced Usage Patterns
[Complex scenarios, best practices, optimization techniques]

### 📊 Performance & Scalability
[Benchmarks, optimization tips, scalability considerations]

---

## 💡 Practical Examples & Tutorials

### 🚀 Quick Start (5-minute setup)
[Absolute beginner tutorial with copy-paste code]

### 🎯 Common Use Cases
[8-12 practical scenarios with complete code examples from research sources]

### 🏆 Advanced Projects
[Complex implementations showing professional usage]

### 💼 Production Examples
[Real-world scenarios with error handling, scaling considerations]

### 🔗 Integration Examples
[How to combine with other tools/frameworks, based on community examples]

---

## 🎪 Interactive Learning

### ❓ Self-Assessment Questions
[Multiple choice, short answer, and practical exercises]

### 🧩 Code Challenges
[Progressive difficulty coding exercises]

### 🎲 "What Would Happen If..." Scenarios
[Hypothetical situations to test understanding]

### 🏃‍♂️ Hands-On Labs
[Step-by-step practical exercises]

---

## 📝 Quick Reference Materials

### ⚡ Ultimate Cheat Sheet
[Condensed reference with most-used commands/patterns from all sources]

### 🗂️ Command Reference
[Alphabetical listing of all commands/functions with syntax]

### 🚨 Common Pitfalls & Solutions
[Frequent mistakes from community discussions and how to avoid/fix them]

### 🔍 Troubleshooting Guide  
[Problem → Diagnosis → Solution format, enhanced with community solutions]

### 📱 Quick Reference Cards
[Printable reference materials for different aspects]

---

## 🎨 Visual Learning Aids

### 📊 Concept Maps
[Text-based diagrams showing relationships between concepts]

### 🌊 Workflow Diagrams
[Step-by-step process flows in ASCII art or detailed descriptions]

### 🗺️ Mental Models
[Frameworks for thinking about the technology]

### 🔄 Decision Trees
[When to use what approach and why]

---

## 🚀 From Beginner to Expert Path

### 📈 Learning Progression
[Structured learning path with prerequisites and milestones]

### 🎯 Skill Checkpoints
[What you should know at each level]

### 📚 Curated Resources
[Best books, courses, communities, tools for deeper learning from research]

### 🏅 Certification & Career Paths
[Professional development opportunities]

---

## 🔗 Integration & Ecosystem

### 🤝 Related Technologies
[How this fits with other tools/frameworks, with examples from research]

### 🔌 Popular Integrations
[Common combinations and patterns from community usage]

### 🌐 Community & Support
[Where to get help, contribute, stay updated]

### 🎯 Alternative Solutions
[When to use alternatives, comparison matrix]

---

## 🎖️ Best Practices & Patterns

### ✅ Industry Standards
[Professional patterns and conventions from research sources]

### 🔒 Security Considerations
[Security best practices and common vulnerabilities]

### 📈 Performance Optimization
[Speed and efficiency tips from real-world usage]

### 🧪 Testing Strategies
[How to test effectively, common testing patterns]

---

## 📊 Real-World Case Studies

### 🏢 Industry Examples
[How companies use this technology (from research)]

### 📈 Success Stories
[Notable implementations and their outcomes]

### ⚠️ Lessons Learned
[Common failures and how to avoid them]

---

**Enhanced Content Creation Guidelines:**

1. **Multi-Source Integration**: Seamlessly blend official docs with community wisdom, tutorials, and real-world examples

2. **Research-Enhanced Examples**: Use examples found in research to show multiple approaches to the same problem

3. **Community Insights**: Include common questions, solutions, and patterns discovered from Stack Overflow, GitHub, and forums

4. **Practical Focus**: Every concept should have multiple examples from different sources showing various implementation approaches

5. **Real-World Context**: Show how concepts are actually used in practice, not just theoretical examples

6. **Comparative Analysis**: When research shows multiple approaches, compare and contrast them

7. **Troubleshooting Enhancement**: Use community-discovered issues and solutions to create comprehensive troubleshooting sections

8. **Best Practices Integration**: Combine official recommendations with community-discovered best practices

**Quality Enhancements from Research:**
- Include alternative explanations for difficult concepts
- Show multiple coding patterns for the same task
- Address gaps in official documentation with community solutions
- Provide context about when to use different approaches
- Include performance comparisons and benchmarks where available

Generate comprehensive study notes following this enhanced structure:
""")

    def _chunk_content_intelligently(self, scraped_data: Dict[str, str], metadata: Dict[str, Dict]) -> List[Dict]:
        """Intelligently chunk content based on priority and relationships"""
        chunks = []
        
        # Sort content with research content getting balanced priority
        sorted_content = []
        for url, content in scraped_data.items():
            meta = metadata.get(url, {})
            analysis = meta.get('analysis', {})
            is_research = meta.get('is_research', False)
            
            # Adjust priority for research content
            base_priority = {'high': 3, 'medium': 2, 'low': 1}.get(analysis.get('priority', 'medium'), 2)
            if is_research and analysis.get('contains_examples', False):
                base_priority += 0.5  # Boost research with examples
            
            sorted_content.append({
                'url': url,
                'content': content,
                'title': meta.get('title', 'Untitled'),
                'analysis': analysis,
                'priority_score': base_priority,
                'is_research': is_research
            })
        
        # Sort by priority, mixing main and research content
        sorted_content.sort(key=lambda x: (x['priority_score'], x['analysis'].get('word_count', 0)), reverse=True)
        
        # Create balanced chunks with mix of main and research content
        current_chunk = {
            'content': '',
            'urls': [],
            'types': set(),
            'char_count': 0,
            'main_pages': 0,
            'research_pages': 0
        }
        
        for item in sorted_content:
            content_type = "🔬 RESEARCH" if item['is_research'] else "📄 MAIN"
            content_with_header = f"\n\n--- {content_type}: {item['title']} ({item['url']}) ---\n\n{item['content']}"
            
            if current_chunk['char_count'] + len(content_with_header) > MAX_CONTENT_CHARS:
                if current_chunk['content']:
                    chunks.append(current_chunk)
                
                current_chunk = {
                    'content': content_with_header,
                    'urls': [item['url']],
                    'types': {item['analysis'].get('type', 'general')},
                    'char_count': len(content_with_header),
                    'main_pages': 0 if item['is_research'] else 1,
                    'research_pages': 1 if item['is_research'] else 0
                }
            else:
                current_chunk['content'] += content_with_header
                current_chunk['urls'].append(item['url'])
                current_chunk['types'].add(item['analysis'].get('type', 'general'))
                current_chunk['char_count'] += len(content_with_header)
                
                if item['is_research']:
                    current_chunk['research_pages'] += 1
                else:
                    current_chunk['main_pages'] += 1
        
        if current_chunk['content']:
            chunks.append(current_chunk)
        
        return chunks

    def generate_comprehensive_notes(self, scraped_data: Dict[str, str], metadata: Dict[str, Dict], website_url: str) -> List[str]:
        """Generate comprehensive notes, potentially split into multiple sections"""
        if not scraped_data:
            return ["No content was scraped to generate notes."]
        
        chunks = self._chunk_content_intelligently(scraped_data, metadata)
        prompt = self._create_comprehensive_prompt()
        chain = prompt | self.llm | self.output_parser
        
        generated_sections = []
        
        for i, chunk in enumerate(chunks):
            main_pages = chunk['main_pages']
            research_pages = chunk['research_pages']
            logger.info(f"Generating notes for chunk {i+1}/{len(chunks)} ({main_pages} main + {research_pages} research pages, {chunk['char_count']} chars)")
            logger.info(f"Content types in chunk: {', '.join(chunk['types'])}")
            
            try:
                section_title = f"Part {i+1}" if len(chunks) > 1 else "Complete Guide"
                section_notes = chain.invoke({
                    "content": chunk['content'], 
                    "website_url": website_url
                })
                
                if len(chunks) > 1:
                    section_header = f"\n\n{'='*80}\n# 📖 {section_title} of {len(chunks)}\n"
                    section_header += f"*Main Content: {main_pages} pages | Research Content: {research_pages} pages*\n"
                    section_header += f"{'='*80}\n\n"
                    section_notes = section_header + section_notes
                
                generated_sections.append(section_notes)
                
            except Exception as e:
                logger.error(f"Error generating notes for chunk {i+1}: {e}")
                generated_sections.append(f"Error generating notes for section {i+1}: {e}")
        
        return generated_sections

def save_comprehensive_notes(notes_sections: List[str], url: str, metadata: Dict[str, Dict], research_queries: List[str] = None) -> None:
    """Save notes with enhanced formatting and metadata"""
    if not notes_sections or not any(notes_sections):
        print("No notes were generated to save.")
        return
    
    domain = urlparse(url).netloc
    safe_domain = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_notes_{safe_domain}_{timestamp}.md"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            # Enhanced header with research information
            f.write(f"# 📚 Comprehensive Study Guide (Enhanced with Research)\n\n")
            f.write(f"**Source:** {url}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Analyze content sources
            main_pages = sum(1 for meta in metadata.values() if not meta.get('is_research', False))
            research_pages = sum(1 for meta in metadata.values() if meta.get('is_research', False))
            
            f.write(f"**Main Documentation Pages:** {main_pages}\n")
            f.write(f"**Research Pages:** {research_pages}\n")
            f.write(f"**Total Pages Processed:** {len(metadata)}\n")
            f.write(f"**Document Sections:** {len(notes_sections)}\n\n")
            
            # Research queries used
            if research_queries:
                f.write("**Research Queries Used:**\n")
                for i, query in enumerate(research_queries, 1):
                    f.write(f"{i}. {query}\n")
                f.write("\n")
            
            # Content type analysis
            content_types = {}
            research_sources = set()
            
            for meta in metadata.values():
                analysis = meta.get('analysis', {})
                content_type = analysis.get('type', 'general')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                if meta.get('is_research', False):
                    url_domain = urlparse(list(metadata.keys())[0]).netloc
                    for original_url in metadata.keys():
                        if meta == metadata[original_url]:
                            research_domain = urlparse(original_url).netloc
                            if research_domain != url_domain:
                                research_sources.add(research_domain)
                            break
            
            f.write("**Content Overview:**\n")
            for content_type, count in sorted(content_types.items()):
                f.write(f"- {content_type.replace('_', ' ').title()}: {count} pages\n")
            
            if research_sources:
                f.write(f"\n**Research Sources:** {', '.join(sorted(research_sources))}\n")
            
            f.write(f"\n{'='*100}\n\n")
            
            # Write all sections
            for i, section in enumerate(notes_sections):
                if section.strip():
                    f.write(section)
                    if i < len(notes_sections) - 1:
                        f.write(f"\n\n{'='*100}\n\n")
        
        # Enhanced metadata with research information
        enhanced_metadata = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'main_pages': main_pages,
                'research_pages': research_pages,
                'research_queries': research_queries or [],
                'research_sources': list(research_sources)
            },
            'pages': metadata
        }
        
        metadata_filename = f"metadata_{safe_domain}_{timestamp}.json"
        with open(metadata_filename, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Success! Enhanced comprehensive notes saved to: {filename}")
        print(f"📊 Metadata saved to: {metadata_filename}")
        print(f"📄 Total sections: {len(notes_sections)}")
        print(f"📈 Main pages: {main_pages} | Research pages: {research_pages}")
        if research_sources:
            print(f"🔬 Research sources: {', '.join(sorted(research_sources))}")
        
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")
        print(f"\n❌ Error: Could not write notes to file.")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced documentation scraper with AI-powered research and comprehensive note generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://docs.example.com --limit 50
  %(prog)s https://api.example.com/docs --limit 100 --no-research
  %(prog)s https://tutorial.example.com --limit 0 --research-limit 15

Research Enhancement:
  The script can automatically generate intelligent Google search queries using Gemini Flash
  and crawl additional high-quality resources to enhance the study material.
  
  Required files for research:
  - googlesearchapi: Google Custom Search API key
  - googlecx: Google Custom Search Engine ID
        """
    )
    parser.add_argument("url", type=str, help="The starting URL of the documentation to process")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=30, 
        help="Maximum number of pages to scrape from main site. Set to 0 for no limit. Default is 30."
    )
    parser.add_argument(
        "--research-limit",
        type=int,
        default=MAX_RESEARCH_PAGES,
        help=f"Maximum number of additional research pages to scrape. Default is {MAX_RESEARCH_PAGES}."
    )
    parser.add_argument(
        "--no-research",
        action="store_true",
        help="Disable research enhancement (only scrape the main documentation site)"
    )
    args = parser.parse_args()

    # Validate URL
    try:
        parsed = urlparse(args.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")
    except Exception:
        print("❌ Error: Please provide a valid URL (including http:// or https://)")
        return

    # Load API keys
    gemini_api_key = APIKeyManager.get_gemini_api_key(API_KEY_FILE)
    if not gemini_api_key:
        print(f"❌ FATAL: Could not load Gemini API key from '{API_KEY_FILE}'.")
        print("Please ensure the file exists and contains a valid Google AI API key.")
        return

    # Load Google Search credentials (optional)
    google_api_key, google_cx = None, None
    research_enabled = not args.no_research
    
    if research_enabled:
        google_api_key, google_cx = APIKeyManager.get_google_search_credentials()
        if not google_api_key or not google_cx:
            print("⚠️  Warning: Google Search credentials not found.")
            print("Research enhancement will use DuckDuckGo as fallback search engine.")
            print("For optimal results, create files:")
            print(f"  - {GOOGLE_API_KEY_FILE}: Your Google Custom Search API key")
            print(f"  - {GOOGLE_CX_FILE}: Your Google Custom Search Engine ID")
            # Don't disable research - we can use DuckDuckGo

    # Display configuration
    limit_text = f"page limit: {args.limit}" if args.limit > 0 else "page limit: UNLIMITED ⚠️"
    
    # Enhanced research status display
    if research_enabled:
        if google_api_key and google_cx:
            research_text = f"research limit: {args.research_limit} (Google + DuckDuckGo fallback)"
        else:
            research_text = f"research limit: {args.research_limit} (DuckDuckGo only)"
    else:
        research_text = "research: DISABLED"
    
    print(f"\n{'='*80}")
    print(f"🚀 ENHANCED DOCUMENTATION SCRAPER WITH AI RESEARCH")
    print(f"{'='*80}")
    print(f"📄 Target: {args.url}")
    print(f"⚙️  Main scraping: {limit_text}")
    print(f"🔬 Research enhancement: {research_text}")
    print(f"🤖 AI Models: {MODEL_NAME} (notes) + {SEARCH_MODEL_NAME} (queries)")
    print(f"{'='*80}\n")

    # Phase 1: Initial scraping
    print("📄 Phase 1: Scraping main documentation...")
    scraper = WebsiteScraper(args.url, max_pages=args.limit)
    scraped_data, metadata = scraper.crawl()

    if not scraped_data:
        print("\n❌ Could not find any content from main site to process.")
        print("This might be due to:")
        print("- Website blocking automated access")
        print("- No readable content found")
        print("- Network connectivity issues")
        return

    # Phase 2: Research enhancement (optional)
    research_queries = []
    if research_enabled:
        print(f"\n🔬 Phase 2: AI-powered research enhancement...")
        
        # Generate intelligent research queries
        query_generator = ResearchQueryGenerator(gemini_api_key)
        research_queries = query_generator.generate_research_queries(scraped_data, metadata, args.url)
        
        if research_queries:
            print(f"🎯 Generated {len(research_queries)} research queries:")
            for i, query in enumerate(research_queries, 1):
                print(f"  {i}. {query}")
            
            # Perform searches with Google/DuckDuckGo fallback
            researcher = GoogleSearchResearcher(google_api_key, google_cx)
            exclude_domain = urlparse(args.url).netloc
            research_urls = researcher.search_and_extract_urls(research_queries, exclude_domain)
            
            if research_urls:
                # Limit research URLs
                research_urls = research_urls[:args.research_limit]
                print(f"\n🔍 Crawling {len(research_urls)} research URLs...")
                
                # Create a new scraper instance for research with higher limit
                research_scraper = WebsiteScraper(args.url, max_pages=args.limit + len(research_urls))
                research_scraped_data, research_metadata = research_scraper.crawl(research_urls)
                
                # Merge research data with main data
                scraped_data.update(research_scraped_data)
                metadata.update(research_metadata)
                
                print(f"✅ Research enhancement complete!")
            else:
                print("⚠️  No useful research URLs found from any search engine.")
        else:
            print("⚠️  Could not generate research queries.")

    # Content analysis
    print(f"\n📊 Final Content Analysis:")
    main_pages = sum(1 for meta in metadata.values() if not meta.get('is_research', False))
    research_pages = sum(1 for meta in metadata.values() if meta.get('is_research', False))
    
    content_types = {}
    total_words = 0
    for meta in metadata.values():
        analysis = meta.get('analysis', {})
        content_type = analysis.get('type', 'general')
        is_research = meta.get('is_research', False)
        type_key = f"{content_type} (research)" if is_research else content_type
        content_types[type_key] = content_types.get(type_key, 0) + 1
        total_words += analysis.get('word_count', 0)
    
    print(f"  📄 Main documentation: {main_pages} pages")
    print(f"  🔬 Research content: {research_pages} pages")
    print(f"  📝 Total words: {total_words:,}")
    print("  📑 Content breakdown:")
    for content_type, count in sorted(content_types.items()):
        print(f"    - {content_type.replace('_', ' ').title()}: {count} pages")

    # Phase 3: AI-powered note generation
    print(f"\n{'='*80}")
    print("🧠 Phase 3: Generating comprehensive study materials...")
    print("This enhanced process integrates official docs with research findings.")
    print("Please wait, this may take several minutes...")
    print(f"{'='*80}")
    
    generator = EnhancedNoteGenerator(gemini_api_key)
    notes_sections = generator.generate_comprehensive_notes(scraped_data, metadata, args.url)

    # Save results
    save_comprehensive_notes(notes_sections, args.url, metadata, research_queries)
    
    print(f"\n🎉 Process completed successfully!")
    print("Your enhanced comprehensive study guide includes:")
    print("  📚 Multi-level explanations (ELI5 to Expert)")
    print("  💡 Practical examples from official docs AND community sources")  
    print("  📝 Enhanced cheat sheets with real-world patterns")
    print("  🎯 Self-assessment questions and coding challenges")
    print("  🗺️  Complete learning progression paths")
    print("  🔬 Research-enhanced troubleshooting and best practices")
    
    if research_pages > 0:
        print(f"  ⭐ BONUS: Enhanced with {research_pages} research sources!")

if __name__ == "__main__":
    main()
