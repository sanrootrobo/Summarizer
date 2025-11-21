import os
import requests
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse
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
API_KEY_FILE = "geminaikey"
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
MAX_CONTENT_CHARS = 500000  # Reduced to allow for more detailed prompts

class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str = "geminaikey") -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"API key file not found at '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10:
                logger.error("Invalid API key found in file.")
                return None
            logger.info("API key loaded successfully.")
            return api_key
        except Exception as e:
            logger.error(f"Error reading API key: {e}")
            return None

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
            'word_count': len(content.split())
        }
        
        url_lower = url.lower()
        content_lower = content.lower()
        
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
        
        # Extract links
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(page_url, href)
            parsed_url = urlparse(full_url)
            
            if (parsed_url.netloc == self.domain and 
                not any(ext in parsed_url.path.lower() for ext in ['.pdf', '.jpg', '.png', '.zip', '.doc', '.mp4'])):
                clean_url = full_url.split('#')[0].split('?')[0]  # Remove fragments and query params
                links.add(clean_url)
        
        return markdown_content, list(links), title

    def crawl(self) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        urls_to_visit = {self.base_url}
        
        with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers to be more polite
            while urls_to_visit:
                if self.max_pages > 0 and len(self.visited_urls) >= self.max_pages:
                    logger.info(f"Reached scraping limit of {self.max_pages} pages. Stopping crawl.")
                    break

                url = urls_to_visit.pop()
                if url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(url)
                
                progress_prefix = f"[{len(self.visited_urls)}/{self.max_pages}]" if self.max_pages > 0 else f"[{len(self.visited_urls)}]"
                logger.info(f"{progress_prefix} Scraping: {url}")
                
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
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                for link in new_links:
                    if link not in self.visited_urls:
                        urls_to_visit.add(link)
                
                sleep(0.3)  # Be more polite to servers
        
        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages with content.")
        return self.scraped_content, self.content_metadata

class EnhancedNoteGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            google_api_key=api_key, 
            temperature=0.3,  # Slightly higher for more creative examples
            max_output_tokens=8192
        )
        self.output_parser = StrOutputParser()

    def _create_comprehensive_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
You are an expert technical documentation analyst and educational content creator. Your task is to transform raw documentation into comprehensive, multi-layered study materials.

**Your Mission:**
Create exhaustive study notes that serve multiple learning styles and proficiency levels. Think of this as creating a complete learning resource that could replace the original documentation.

**Content to Process:**
Website: {website_url}
Documentation Content: {content}

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

---

## 🧠 Core Concepts Explained

### 🔤 ELI5 (Explain Like I'm 5) Section
[Break down complex concepts using simple analogies, metaphors, and everyday examples. Use storytelling where appropriate.]

### 🏗️ Architecture & How It Works
[Technical explanation with diagrams described in text, flow charts, system interactions]

### 🎭 Real-World Analogies
[Multiple analogies comparing technical concepts to familiar real-world scenarios]

---

## 📖 Detailed Reference Guide

### 📚 Complete API/Feature Reference
[Organized by category, with every endpoint/feature documented]

### 🛠️ Configuration & Setup
[Step-by-step instructions, common configurations, troubleshooting]

### 🔧 Advanced Usage Patterns
[Complex scenarios, best practices, optimization techniques]

---

## 💡 Practical Examples & Tutorials

### 🚀 Quick Start (5-minute setup)
[Absolute beginner tutorial with copy-paste code]

### 🎯 Common Use Cases
[5-10 practical scenarios with complete code examples]

### 🏆 Advanced Projects
[Complex implementations showing professional usage]

### 💼 Production Examples
[Real-world scenarios with error handling, scaling considerations]

---

## 🎪 Interactive Learning

### ❓ Self-Assessment Questions
[Multiple choice, short answer, and practical exercises]

### 🧩 Code Challenges
[Progressive difficulty coding exercises]

### 🎲 "What Would Happen If..." Scenarios
[Hypothetical situations to test understanding]

---

## 📝 Quick Reference Materials

### ⚡ Cheat Sheet
[Condensed reference with most-used commands/patterns]

### 🗂️ Command Reference
[Alphabetical listing of all commands/functions with syntax]

### 🚨 Common Pitfalls & Solutions
[Frequent mistakes and how to avoid/fix them]

### 🔍 Troubleshooting Guide
[Problem → Diagnosis → Solution format]

---

## 🎨 Visual Learning Aids

### 📊 Concept Maps
[Text-based diagrams showing relationships between concepts]

### 🌊 Workflow Diagrams
[Step-by-step process flows in ASCII art or detailed descriptions]

### 🗺️ Mental Models
[Frameworks for thinking about the technology]

---

## 🚀 From Beginner to Expert Path

### 📈 Learning Progression
[Structured learning path with prerequisites and milestones]

### 🎯 Skill Checkpoints
[What you should know at each level]

### 📚 Additional Resources
[Books, courses, communities, tools for deeper learning]

---

## 🔗 Integration & Ecosystem

### 🤝 Related Technologies
[How this fits with other tools/frameworks]

### 🔌 Common Integrations
[Popular combinations and use patterns]

### 🌐 Community & Support
[Where to get help, contribute, stay updated]

---

**Content Creation Guidelines:**

1. **Multiple Learning Styles:** Include visual (diagrams in text), auditory (explanations), kinesthetic (hands-on examples), and reading/writing components.

2. **Progressive Complexity:** Start simple, build complexity gradually. Each section should be accessible to its intended audience level.

3. **Practical Focus:** Every concept should have at least one practical example. Abstract concepts need multiple examples.

4. **Memory Aids:** Use mnemonics, acronyms, patterns, and memorable analogies throughout.

5. **Error Prevention:** Anticipate common mistakes and address them proactively.

6. **Copy-Paste Ready:** All code examples should be complete and runnable.

7. **Context Switching:** Help readers understand when to use what approach and why.

8. **Future-Proofing:** Include information about evolution, deprecations, and upcoming changes where relevant.

**Tone & Style:**
- Enthusiastic but professional
- Clear and concise while being comprehensive
- Use emojis strategically for visual organization
- Vary sentence structure to maintain engagement
- Include encouraging and motivational language

**Quality Checks:**
- Could a complete beginner understand the ELI5 sections?
- Could an expert use this as a reference?
- Are there enough examples for each concept?
- Is the progression logical and well-paced?
- Would someone prefer this to the original documentation?

Generate comprehensive study notes following this structure:
""")

    def _chunk_content_intelligently(self, scraped_data: Dict[str, str], metadata: Dict[str, Dict]) -> List[Dict]:
        """Intelligently chunk content based on priority and relationships"""
        chunks = []
        
        # Sort content by priority and type
        sorted_content = []
        for url, content in scraped_data.items():
            meta = metadata.get(url, {})
            analysis = meta.get('analysis', {})
            priority_score = {'high': 3, 'medium': 2, 'low': 1}.get(analysis.get('priority', 'medium'), 2)
            
            sorted_content.append({
                'url': url,
                'content': content,
                'title': meta.get('title', 'Untitled'),
                'analysis': analysis,
                'priority_score': priority_score
            })
        
        # Sort by priority, then by content type
        sorted_content.sort(key=lambda x: (x['priority_score'], x['analysis'].get('word_count', 0)), reverse=True)
        
        # Create intelligent chunks
        current_chunk = {
            'content': '',
            'urls': [],
            'types': set(),
            'char_count': 0
        }
        
        for item in sorted_content:
            content_with_header = f"\n\n--- {item['title']} ({item['url']}) ---\n\n{item['content']}"
            
            if current_chunk['char_count'] + len(content_with_header) > MAX_CONTENT_CHARS:
                if current_chunk['content']:  # Don't add empty chunks
                    chunks.append(current_chunk)
                
                current_chunk = {
                    'content': content_with_header,
                    'urls': [item['url']],
                    'types': {item['analysis'].get('type', 'general')},
                    'char_count': len(content_with_header)
                }
            else:
                current_chunk['content'] += content_with_header
                current_chunk['urls'].append(item['url'])
                current_chunk['types'].add(item['analysis'].get('type', 'general'))
                current_chunk['char_count'] += len(content_with_header)
        
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
            logger.info(f"Generating notes for chunk {i+1}/{len(chunks)} ({len(chunk['urls'])} pages, {chunk['char_count']} chars)")
            logger.info(f"Content types in chunk: {', '.join(chunk['types'])}")
            
            try:
                section_title = f"Part {i+1}" if len(chunks) > 1 else "Complete Guide"
                section_notes = chain.invoke({
                    "content": chunk['content'], 
                    "website_url": website_url
                })
                
                if len(chunks) > 1:
                    section_header = f"\n\n{'='*80}\n# 📖 {section_title} of {len(chunks)}\n{'='*80}\n\n"
                    section_notes = section_header + section_notes
                
                generated_sections.append(section_notes)
                
            except Exception as e:
                logger.error(f"Error generating notes for chunk {i+1}: {e}")
                generated_sections.append(f"Error generating notes for section {i+1}: {e}")
        
        return generated_sections

def save_comprehensive_notes(notes_sections: List[str], url: str, metadata: Dict[str, Dict]) -> None:
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
            # Write header with metadata
            f.write(f"# 📚 Comprehensive Study Guide\n\n")
            f.write(f"**Source:** {url}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Pages Processed:** {len(metadata)}\n")
            f.write(f"**Document Sections:** {len(notes_sections)}\n\n")
            
            # Write content type summary
            content_types = {}
            for meta in metadata.values():
                content_type = meta.get('analysis', {}).get('type', 'general')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            f.write("**Content Overview:**\n")
            for content_type, count in sorted(content_types.items()):
                f.write(f"- {content_type.replace('_', ' ').title()}: {count} pages\n")
            
            f.write(f"\n{'='*100}\n\n")
            
            # Write all sections
            for i, section in enumerate(notes_sections):
                if section.strip():
                    f.write(section)
                    if i < len(notes_sections) - 1:
                        f.write(f"\n\n{'='*100}\n\n")
        
        # Also save metadata as JSON for reference
        metadata_filename = f"metadata_{safe_domain}_{timestamp}.json"
        with open(metadata_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Success! Comprehensive notes saved to: {filename}")
        print(f"📊 Metadata saved to: {metadata_filename}")
        print(f"📄 Total sections: {len(notes_sections)}")
        print(f"📈 Pages processed: {len(metadata)}")
        
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")
        print(f"\n❌ Error: Could not write notes to file.")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced documentation scraper with comprehensive note generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://docs.example.com --limit 50
  %(prog)s https://api.example.com/docs --limit 0  # No limit
  %(prog)s https://tutorial.example.com --limit 10
        """
    )
    parser.add_argument("url", type=str, help="The starting URL of the documentation to process")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=30, 
        help="Maximum number of pages to scrape. Set to 0 for no limit. Default is 30."
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

    api_key = APIKeyManager.get_api_key(API_KEY_FILE)
    if not api_key:
        print(f"❌ FATAL: Could not load API key from '{API_KEY_FILE}'.")
        print("Please ensure the file exists and contains a valid Google AI API key.")
        return

    limit_text = f"page limit: {args.limit}" if args.limit > 0 else "page limit: UNLIMITED ⚠️"
    print(f"\n{'='*80}")
    print(f"🚀 ENHANCED DOCUMENTATION SCRAPER")
    print(f"{'='*80}")
    print(f"📄 Target: {args.url}")
    print(f"⚙️  Configuration: {limit_text}")
    print(f"🤖 AI Model: {MODEL_NAME}")
    print(f"{'='*80}\n")

    # Scraping phase
    print("🕷️  Starting intelligent web scraping...")
    scraper = WebsiteScraper(args.url, max_pages=args.limit)
    scraped_data, metadata = scraper.crawl()

    if not scraped_data:
        print("\n❌ Could not find any content to process.")
        print("This might be due to:")
        print("- Website blocking automated access")
        print("- No readable content found")
        print("- Network connectivity issues")
        return

    # Analysis phase
    print(f"\n📊 Content Analysis:")
    content_types = {}
    total_words = 0
    for meta in metadata.values():
        analysis = meta.get('analysis', {})
        content_type = analysis.get('type', 'general')
        content_types[content_type] = content_types.get(content_type, 0) + 1
        total_words += analysis.get('word_count', 0)
    
    for content_type, count in sorted(content_types.items()):
        print(f"  📑 {content_type.replace('_', ' ').title()}: {count} pages")
    print(f"  📝 Total words: {total_words:,}")

    # Generation phase
    print(f"\n{'='*80}")
    print("🧠 Generating comprehensive study materials...")
    print("This may take several minutes for large documentation sets.")
    print(f"{'='*80}")
    
    generator = EnhancedNoteGenerator(api_key)
    notes_sections = generator.generate_comprehensive_notes(scraped_data, metadata, args.url)

    # Save results
    save_comprehensive_notes(notes_sections, args.url, metadata)
    
    print(f"\n🎉 Process completed successfully!")
    print("Your comprehensive study guide includes:")
    print("  📚 Multi-level explanations (ELI5 to Expert)")
    print("  💡 Practical examples and tutorials")  
    print("  📝 Quick reference materials and cheat sheets")
    print("  🎯 Self-assessment questions and exercises")
    print("  🗺️  Learning progression paths")

if __name__ == "__main__":
    main()
