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
REQUEST_TIMEOUT = 12
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_CONTENT_CHARS = 400000  # Optimized for better prompt space
RATE_LIMIT_DELAY = 0.5  # More respectful crawling

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
    """Enhanced content analysis with better categorization"""
    
    @staticmethod
    def categorize_content(url: str, content: str) -> Dict[str, any]:
        """Advanced content categorization with semantic analysis"""
        categories = {
            'type': 'general',
            'priority': 'medium',
            'contains_code': False,
            'contains_examples': False,
            'is_api_reference': False,
            'is_tutorial': False,
            'is_concept': False,
            'is_troubleshooting': False,
            'is_installation': False,
            'complexity_level': 'intermediate',
            'word_count': len(content.split()),
            'code_blocks': 0,
            'headers_count': 0
        }
        
        url_lower = url.lower()
        content_lower = content.lower()
        
        # Enhanced URL pattern matching
        api_patterns = ['api', 'reference', '/api/', 'endpoints', 'swagger', 'openapi']
        tutorial_patterns = ['tutorial', 'guide', 'getting-started', 'quickstart', 'walkthrough', 'how-to']
        concept_patterns = ['concept', 'overview', 'introduction', 'fundamentals', 'basics', 'theory']
        example_patterns = ['example', 'sample', 'demo', 'playground', 'snippet']
        troubleshooting_patterns = ['troubleshoot', 'debug', 'error', 'fix', 'problem', 'issue']
        install_patterns = ['install', 'setup', 'configuration', 'deployment', 'getting-started']
        
        # Categorize by URL patterns with weighted scoring
        if any(pattern in url_lower for pattern in api_patterns):
            categories.update({
                'type': 'api_reference',
                'is_api_reference': True,
                'priority': 'high',
                'complexity_level': 'advanced'
            })
        elif any(pattern in url_lower for pattern in tutorial_patterns):
            categories.update({
                'type': 'tutorial',
                'is_tutorial': True,
                'priority': 'high',
                'complexity_level': 'beginner'
            })
        elif any(pattern in url_lower for pattern in concept_patterns):
            categories.update({
                'type': 'concept',
                'is_concept': True,
                'priority': 'high',
                'complexity_level': 'beginner'
            })
        elif any(pattern in url_lower for pattern in example_patterns):
            categories.update({
                'type': 'example',
                'contains_examples': True,
                'priority': 'medium',
                'complexity_level': 'intermediate'
            })
        elif any(pattern in url_lower for pattern in troubleshooting_patterns):
            categories.update({
                'type': 'troubleshooting',
                'is_troubleshooting': True,
                'priority': 'high',
                'complexity_level': 'intermediate'
            })
        elif any(pattern in url_lower for pattern in install_patterns):
            categories.update({
                'type': 'installation',
                'is_installation': True,
                'priority': 'high',
                'complexity_level': 'beginner'
            })
        
        # Enhanced content analysis
        code_indicators = ['```', '`', 'function', 'class', 'import', 'def ', 'const ', 'var ', 'let ', '<code>']
        categories['code_blocks'] = sum(content.count(indicator) for indicator in ['```', '<code>'])
        categories['contains_code'] = any(indicator in content_lower for indicator in code_indicators)
        
        example_indicators = ['example', 'for instance', 'here\'s how', 'demo', 'sample code', 'try this']
        categories['contains_examples'] = any(indicator in content_lower for indicator in example_indicators)
        
        # Count headers for structure analysis
        categories['headers_count'] = content.count('#') + content.count('##') + content.count('###')
        
        # Complexity assessment based on content
        advanced_terms = ['algorithm', 'optimization', 'architecture', 'scalability', 'performance', 'security']
        beginner_terms = ['introduction', 'basic', 'simple', 'easy', 'quick', 'start']
        
        if any(term in content_lower for term in advanced_terms):
            categories['complexity_level'] = 'advanced'
        elif any(term in content_lower for term in beginner_terms):
            categories['complexity_level'] = 'beginner'
            
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
        self.failed_urls = set()
        
        # Enhanced HTML to Markdown converter
        self.text_maker = html2text.HTML2Text()
        self.text_maker.body_width = 0
        self.text_maker.ignore_links = False
        self.text_maker.ignore_images = True
        self.text_maker.ignore_emphasis = False
        self.text_maker.skip_internal_links = False

    def _get_page_content(self, url: str) -> Optional[str]:
        """Enhanced content fetching with better error handling"""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                return response.text
            else:
                logger.debug(f"Skipping non-HTML content: {url} ({content_type})")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url}")
            self.failed_urls.add(url)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            self.failed_urls.add(url)
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            self.failed_urls.add(url)
            
        return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Enhanced title extraction with fallbacks"""
        # Try multiple title sources in order of preference
        title_sources = [
            lambda s: s.find('title'),
            lambda s: s.find('h1'),
            lambda s: s.find('h2'),
            lambda s: s.find('meta', {'property': 'og:title'}),
            lambda s: s.find('meta', {'name': 'title'})
        ]
        
        for source in title_sources:
            element = source(soup)
            if element:
                title = element.get('content') if element.name == 'meta' else element.get_text()
                if title and title.strip():
                    # Clean up title
                    title = re.sub(r'\s+', ' ', title.strip())
                    return title[:100]  # Limit length
        
        return "Untitled Page"

    def _parse_content_and_links(self, html: str, page_url: str) -> Tuple[str, List[str], str]:
        """Enhanced content parsing with better structure preservation"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = self._extract_title(soup)
        
        # Remove unwanted elements more selectively
        unwanted_selectors = [
            'script', 'style', 'noscript',
            'nav', 'footer', '.footer',
            'header', '.header',
            'aside', '.sidebar',
            '.advertisement', '.ad',
            '.cookie-banner', '.popup',
            'iframe[src*="youtube"]', 'iframe[src*="vimeo"]'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Enhanced main content detection
        main_content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.main-content',
            '.content',
            '.documentation',
            '.docs-content',
            '.markdown-body',
            '#content',
            '.post-content',
            '.entry-content'
        ]
        
        main_content = None
        for selector in main_content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body or soup
        
        # Convert to markdown with better formatting
        if main_content:
            markdown_content = self.text_maker.handle(str(main_content))
            # Clean up excessive whitespace
            markdown_content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
            markdown_content = markdown_content.strip()
        else:
            markdown_content = ""
        
        # Enhanced link extraction
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(page_url, href)
            parsed_url = urlparse(full_url)
            
            # More comprehensive filtering
            if (parsed_url.netloc == self.domain and 
                not any(ext in parsed_url.path.lower() for ext in 
                       ['.pdf', '.jpg', '.png', '.gif', '.zip', '.doc', '.docx', 
                        '.xls', '.xlsx', '.ppt', '.pptx', '.mp4', '.mp3', '.avi']) and
                not parsed_url.path.endswith('/') or len(parsed_url.path) > 1):
                
                clean_url = full_url.split('#')[0].split('?')[0]
                # Avoid obvious non-content URLs
                if not any(skip in clean_url.lower() for skip in 
                          ['login', 'register', 'admin', 'download', 'print']):
                    links.add(clean_url)
        
        return markdown_content, list(links), title

    def crawl(self) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """Enhanced crawling with better prioritization"""
        urls_to_visit = {self.base_url}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            while urls_to_visit:
                if self.max_pages > 0 and len(self.visited_urls) >= self.max_pages:
                    logger.info(f"Reached scraping limit of {self.max_pages} pages. Stopping crawl.")
                    break

                url = urls_to_visit.pop()
                if url in self.visited_urls or url in self.failed_urls:
                    continue
                    
                self.visited_urls.add(url)
                
                progress_prefix = f"[{len(self.visited_urls)}/{self.max_pages}]" if self.max_pages > 0 else f"[{len(self.visited_urls)}]"
                logger.info(f"{progress_prefix} Scraping: {url}")
                
                html_content = self._get_page_content(url)
                if not html_content:
                    continue
                
                text, new_links, title = self._parse_content_and_links(html_content, url)
                
                # More selective content filtering
                if text and len(text.strip()) > 200 and len(text.split()) > 20:
                    self.scraped_content[url] = text
                    
                    # Enhanced metadata with performance metrics
                    self.content_metadata[url] = {
                        'title': title,
                        'analysis': ContentAnalyzer.categorize_content(url, text),
                        'scraped_at': datetime.now().isoformat(),
                        'char_count': len(text),
                        'estimated_read_time': max(1, len(text.split()) // 200)  # words per minute
                    }
                else:
                    logger.debug(f"Skipping low-content page: {url}")
                
                # Add new links with priority-based ordering
                for link in new_links:
                    if link not in self.visited_urls and link not in self.failed_urls:
                        urls_to_visit.add(link)
                
                sleep(RATE_LIMIT_DELAY)
        
        success_rate = len(self.scraped_content) / max(1, len(self.visited_urls)) * 100
        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages with content. Success rate: {success_rate:.1f}%")
        return self.scraped_content, self.content_metadata

class EnhancedNoteGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            google_api_key=api_key, 
            temperature=0.4,  # Balanced creativity and consistency
            max_output_tokens=8192
        )
        self.output_parser = StrOutputParser()

    def _create_comprehensive_prompt(self) -> ChatPromptTemplate:
        """Significantly improved prompt with better structure and engagement"""
        return ChatPromptTemplate.from_template("""
You are an elite technical educator and documentation specialist with expertise in creating world-class learning materials. Your mission is to transform raw documentation into an engaging, comprehensive study guide that makes complex topics accessible and memorable.

**CONTEXT:**
- Website: {website_url}
- Raw Documentation: {content}

**YOUR MISSION:**
Create a masterful study guide that's both comprehensive and genuinely engaging. Think Netflix-quality educational content meets comprehensive technical reference.

---

# 🎯 **[DYNAMIC TITLE BASED ON CONTENT]**

*🔥 The Ultimate Study Guide That Actually Makes Sense*

## 📚 **Smart Table of Contents**
[Create an intelligent, nested TOC with estimated reading times]

---

## ⚡ **The 60-Second Elevator Pitch**

**🎪 What is this?** [One compelling sentence that hooks the reader]  
**🎯 Perfect for:** [Specific user personas - be creative and relatable]  
**💡 Why you'll love it:** [3-4 benefit-focused bullets that create excitement]  
**⏰ Time investment:** [Realistic learning timeline with milestones]  
**🚀 What you'll build:** [Specific, exciting outcomes they'll achieve]

> 💬 **"Think of this as your GPS for [topic] - no more wandering in documentation hell!"**

---

## 🧠 **The "Aha!" Moment Explained**

### 🎈 **ELI5: The Story Version**
[Use storytelling, characters, and vivid analogies. Make it memorable and fun. Think Pixar-level explanation.]

### 🏗️ **Under the Hood: How It Really Works**
[Technical deep-dive with ASCII diagrams, flowcharts, and system thinking]

```
[ASCII DIAGRAM showing core concepts and relationships]
```

### 🌟 **The Mental Model That Changes Everything**
[Provide a breakthrough way of thinking about the concept that makes everything click]

---

## 🎮 **Interactive Learning Adventure**

### 🎯 **Mission 1: Zero to Hero (15 minutes)**
```
[Complete, copy-paste ready tutorial with:
- Clear objectives
- Step-by-step with expected outputs
- Common pitfalls called out
- Victory conditions]
```

### 🏆 **Mission 2: Real-World Challenge**
**Scenario:** [Engaging, practical scenario]
**Your Mission:** [Clear objective]
**Tools:** [What they'll use]
**Expected Outcome:** [What success looks like]

```
[Complete working example with detailed explanations]
```

### 🎪 **Mission 3: Advanced Mastery**
[Complex, production-ready scenario that showcases expertise]

---

## 📖 **The Complete Playbook**

### 🔧 **Configuration Mastery**
| Setting | Purpose | Pro Tip | When to Use |
|---------|---------|---------|-------------|
| [Fill with comprehensive table] ||||

### 🎛️ **All the Knobs and Switches**
[Complete API/feature reference organized by user journey, not alphabetically]

#### **For Getting Started** 🌱
[Most essential functions with beginner-friendly explanations]

#### **For Daily Work** ⚡
[Most commonly used features with efficiency tips]

#### **For Advanced Scenarios** 🚀
[Power-user features with use cases and examples]

---

## 🎨 **Pattern Library & Best Practices**

### ✨ **The Golden Patterns**
[5-7 proven patterns with when/why to use each]

```javascript
// Pattern 1: The Foundation
// Why: [Explanation]
// When: [Use cases]
[Code example with detailed comments]
```

### 🚫 **Pitfall Prevention**
| ❌ **Don't Do This** | ✅ **Do This Instead** | 🤔 **Why It Matters** |
|---------------------|------------------------|------------------------|
| [Anti-patterns] | [Correct patterns] | [Impact explanation] |

---

## 🎯 **Mastery Checkpoints**

### 🎮 **Level 1: Apprentice** (30 minutes)
**Skills Check:**
- [ ] Can explain [concept] to a colleague
- [ ] Built your first [basic implementation]
- [ ] Debugged common error scenarios

**Challenge:** [Specific task to prove competency]

### 🎮 **Level 2: Practitioner** (2 hours)
**Skills Check:**
- [ ] Implemented [intermediate concepts]
- [ ] Optimized for [specific scenarios]
- [ ] Integrated with [related technologies]

**Challenge:** [More complex integration task]

### 🎮 **Level 3: Expert** (8+ hours)
**Skills Check:**
- [ ] Designed scalable architectures
- [ ] Contributed to community solutions
- [ ] Mentored others successfully

**Challenge:** [Open-ended, creative problem]

---

## 🔥 **The Power-User Toolkit**

### ⚡ **Cheat Sheet Supreme**
```bash
# The commands you'll use 80% of the time
[Most essential commands with flags and examples]
```

### 🎪 **Workflow Automation**
[Scripts, shortcuts, and productivity hacks]

### 🔍 **Debugging Like a Detective**
**The 5-Step Debug Process:**
1. [Systematic approach to problem-solving]
2. [Tools and techniques]
3. [Common solutions]

---

## 🌐 **Beyond the Basics**

### 🤝 **Ecosystem Connections**
[How this fits with popular tools, frameworks, and workflows]

### 📈 **Level Up Your Skills**
**Next Learning Steps:**
- 📚 **Books:** [Curated recommendations]
- 🎥 **Videos:** [Quality tutorials and talks]
- 🛠️ **Tools:** [Essential utilities and extensions]
- 👥 **Communities:** [Where to get help and contribute]

### 🔮 **Future-Proofing**
[Emerging trends, upcoming changes, and how to stay current]

---

## 🎉 **Your Success Toolkit**

### 📋 **Quick Reference Cards**
[Downloadable, printable reference materials]

### 🎯 **Practice Exercises** 
[Progressive challenges with solutions]

### 💡 **Real-World Project Ideas**
[Specific projects to build portfolio pieces]

---

**🎪 TONE & STYLE REQUIREMENTS:**
- Write like you're the most enthusiastic, knowledgeable mentor ever
- Use analogies that make people go "OH! Now I get it!"
- Include personality and humor without being unprofessional
- Make complex things feel achievable and exciting
- Use emojis strategically for visual scanning and engagement
- Vary sentence length for rhythm and readability
- Include encouraging callouts and motivational elements

**🎯 ENGAGEMENT PRINCIPLES:**
- Start with wins (quick success moments)
- Build confidence progressively
- Provide multiple explanation styles for different learning preferences
- Include "why" context for every "what" and "how"
- Make it scannable but comprehensive
- Include interactive elements and challenges
- End sections with clear next actions

**✅ QUALITY CHECKLIST:**
- Would a complete beginner feel confident starting?
- Would an expert find valuable shortcuts and insights?
- Are there enough examples to cover different use cases?
- Is the progression logical and well-paced?
- Would someone bookmark this over the original docs?
- Does it solve real problems people actually face?

Generate an exceptional study guide that follows this structure and principles:
""")

    def _chunk_content_intelligently(self, scraped_data: Dict[str, str], metadata: Dict[str, Dict]) -> List[Dict]:
        """Enhanced intelligent chunking with content-aware grouping"""
        chunks = []
        
        # Enhanced content grouping by type and relationships
        content_by_type = {
            'foundation': [],    # Installation, setup, concepts
            'tutorials': [],     # Tutorials, getting started
            'reference': [],     # API docs, references
            'examples': [],      # Examples, samples
            'advanced': [],      # Advanced topics, troubleshooting
            'general': []        # Everything else
        }
        
        # Categorize content more intelligently
        for url, content in scraped_data.items():
            meta = metadata.get(url, {})
            analysis = meta.get('analysis', {})
            
            content_item = {
                'url': url,
                'content': content,
                'title': meta.get('title', 'Untitled'),
                'analysis': analysis,
                'priority_score': self._calculate_priority_score(analysis),
                'char_count': len(content),
                'complexity': analysis.get('complexity_level', 'intermediate')
            }
            
            # Smart categorization
            if analysis.get('is_installation') or analysis.get('is_concept'):
                content_by_type['foundation'].append(content_item)
            elif analysis.get('is_tutorial'):
                content_by_type['tutorials'].append(content_item)
            elif analysis.get('is_api_reference'):
                content_by_type['reference'].append(content_item)
            elif analysis.get('contains_examples'):
                content_by_type['examples'].append(content_item)
            elif analysis.get('complexity_level') == 'advanced' or analysis.get('is_troubleshooting'):
                content_by_type['advanced'].append(content_item)
            else:
                content_by_type['general'].append(content_item)
        
        # Create balanced chunks with mixed content types
        content_order = ['foundation', 'tutorials', 'examples', 'reference', 'general', 'advanced']
        
        current_chunk = {
            'content': '',
            'urls': [],
            'types': set(),
            'char_count': 0,
            'complexity_mix': {'beginner': 0, 'intermediate': 0, 'advanced': 0}
        }
        
        for content_type in content_order:
            items = sorted(content_by_type[content_type], 
                         key=lambda x: x['priority_score'], reverse=True)
            
            for item in items:
                content_with_header = f"\n\n--- {item['title']} ({item['url']}) ---\n{item['analysis']['type'].upper()} | {item['complexity'].upper()}\n\n{item['content']}"
                
                if current_chunk['char_count'] + len(content_with_header) > MAX_CONTENT_CHARS:
                    if current_chunk['content']:
                        chunks.append(current_chunk)
                    
                    current_chunk = {
                        'content': content_with_header,
                        'urls': [item['url']],
                        'types': {content_type},
                        'char_count': len(content_with_header),
                        'complexity_mix': {item['complexity']: 1}
                    }
                else:
                    current_chunk['content'] += content_with_header
                    current_chunk['urls'].append(item['url'])
                    current_chunk['types'].add(content_type)
                    current_chunk['char_count'] += len(content_with_header)
                    current_chunk['complexity_mix'][item['complexity']] = current_chunk['complexity_mix'].get(item['complexity'], 0) + 1
        
        if current_chunk['content']:
            chunks.append(current_chunk)
        
        return chunks

    def _calculate_priority_score(self, analysis: Dict) -> int:
        """Calculate content priority based on multiple factors"""
        score = 0
        
        # Base priority
        priority_weights = {'high': 10, 'medium': 5, 'low': 2}
        score += priority_weights.get(analysis.get('priority', 'medium'), 5)
        
        # Content type bonuses
        if analysis.get('is_tutorial') or analysis.get('is_concept'):
            score += 8
        elif analysis.get('is_api_reference'):
            score += 6
        elif analysis.get('contains_examples'):
            score += 4
        
        # Content quality indicators
        if analysis.get('contains_code'):
            score += 3
        if analysis.get('word_count', 0) > 500:
            score += 2
        if analysis.get('headers_count', 0) > 3:
            score += 1
        
        return score

    def generate_comprehensive_notes(self, scraped_data: Dict[str, str], metadata: Dict[str, Dict], website_url: str) -> List[str]:
        """Generate enhanced comprehensive notes with better content organization"""
        if not scraped_data:
            return ["No content was scraped to generate notes."]
        
        chunks = self._chunk_content_intelligently(scraped_data, metadata)
        prompt = self._create_comprehensive_prompt()
        chain = prompt | self.llm | self.output_parser
        
        generated_sections = []
        
        for i, chunk in enumerate(chunks):
            content_types = ', '.join(chunk['types'])
            complexity_summary = ', '.join([f"{k}: {v}" for k, v in chunk['complexity_mix'].items() if v > 0])
            
            logger.info(f"Generating notes for section {i+1}/{len(chunks)}")
            logger.info(f"  📄 Pages: {len(chunk['urls'])}")
            logger.info(f"  📊 Types: {content_types}")
            logger.info(f"  🎯 Complexity: {complexity_summary}")
            logger.info(f"  📝 Size: {chunk['char_count']:,} chars")
            
            try:
                section_notes = chain.invoke({
                    "content": chunk['content'], 
                    "website_url": website_url
                })
                
                if len(chunks) > 1:
                    section_header = f"\n\n{'='*100}\n# 📖 **Study Guide Section {i+1} of {len(chunks)}**\n*Focus: {content_types.title()}*\n{'='*100}\n\n"
                    section_notes = section_header + section_notes
                
                generated_sections.append(section_notes)
                
            except Exception as e:
                logger.error(f"Error generating notes for section {i+1}: {e}")
                error_section = f"\n\n# ❌ Error in Section {i+1}\n\nFailed to generate notes for this section. Error: {str(e)}\n"
                generated_sections.append(error_section)
        
        return generated_sections

def save_comprehensive_notes(notes_sections: List[str], url: str, metadata: Dict[str, Dict]) -> None:
    """Enhanced note saving with better organization and metadata"""
    if not notes_sections or not any(notes_sections):
        print("❌ No notes were generated to save.")
        return
    
    domain = urlparse(url).netloc
    safe_domain = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"study_guide_{safe_domain}_{timestamp}.md"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            # Enhanced header with study guide branding
            f.write(f"# 🎓 Comprehensive Study Guide\n")
            f.write(f"*Generated by Enhanced Documentation Scraper*\n\n")
            f.write(f"📚 **Source:** {url}\n")
            f.write(f"🕒 **Generated:** {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}\n")
            f.write(f"📄 **Pages Processed:** {len(metadata)}\n")
            f.write(f"📖 **Study Sections:** {len(notes_sections)}\n")
            
            # Enhanced content analysis
            content_stats = {
                'types': {},
                'complexity': {'beginner': 0, 'intermediate': 0, 'advanced': 0},
                'total_words': 0,
                'total_read_time': 0
            }
            
            for meta in metadata.values():
                analysis = meta.get('analysis', {})
                content_type = analysis.get('type', 'general')
                complexity = analysis.get('complexity_level', 'intermediate')
                
                content_stats['types'][content_type] = content_stats['types'].get(content_type, 0) + 1
                content_stats['complexity'][complexity] += 1
                content_stats['total_words'] += analysis.get('word_count', 0)
                content_stats['total_read_time'] += meta.get('estimated_read_time', 0)
            
            f.write(f"⏱️  **Estimated Study Time:** {content_stats['total_read_time']} minutes\n\n")
            
            f.write("## 📊 **Content Breakdown**\n\n")
            f.write("### 📚 **Content Types**\n")
            for content_type, count in sorted(content_stats['types'].items()):
                emoji_map = {
                    'tutorial': '🎯', 'concept': '🧠', 'api_reference': '📚',
                    'example': '💡', 'installation': '⚙️', 'troubleshooting': '🔧',
                    'general': '📄'
                }
                emoji = emoji_map.get(content_type, '📄')
                f.write(f"- {emoji} **{content_type.replace('_', ' ').title()}:** {count} pages\n")
            
            f.write(f"\n### 🎓 **Difficulty Distribution**\n")
            for level, count in content_stats['complexity'].items():
                if count > 0:
                    level_emoji = {'beginner': '🌱', 'intermediate': '🚀', 'advanced': '🎯'}
                    f.write(f"- {level_emoji[level]} **{level.title()}:** {count} pages\n")
            
            f.write(f"\n### 📈 **Study Metrics**\n")
            f.write(f"- 📝 **Total Words:** {content_stats['total_words']:,}\n")
            f.write(f"- ⏰ **Reading Time:** ~{content_stats['total_read_time']} minutes\n")
            f.write(f"- 🎯 **Completion Goal:** Study in {max(1, content_stats['total_read_time'] // 60)} session(s)\n")
            
            f.write(f"\n{'='*100}\n\n")
            f.write("# 🚀 **Ready to Learn? Let's Go!**\n\n")
            f.write("> 💡 **Pro Tip:** This guide is designed for active learning. Have your development environment ready and try the examples as you read!\n\n")
            f.write(f"{'='*100}\n\n")
            
            # Write all sections with enhanced formatting
            for i, section in enumerate(notes_sections):
                if section.strip():
                    f.write(section)
                    if i < len(notes_sections) - 1:
                        f.write(f"\n\n{'='*100}\n")
                        f.write(f"# 🎯 **Continue Learning...**\n")
                        f.write(f"{'='*100}\n\n")
        
        # Enhanced metadata file with study analytics
        metadata_filename = f"study_analytics_{safe_domain}_{timestamp}.json"
        enhanced_metadata = {
            'generation_info': {
                'source_url': url,
                'generated_at': datetime.now().isoformat(),
                'total_pages': len(metadata),
                'total_sections': len(notes_sections),
                'scraper_version': '2.0_enhanced'
            },
            'content_analytics': content_stats,
            'page_details': metadata,
            'study_recommendations': {
                'suggested_order': ['installation', 'concept', 'tutorial', 'example', 'api_reference', 'troubleshooting'],
                'time_per_session': 30,
                'total_sessions': max(1, content_stats['total_read_time'] // 30),
                'difficulty_progression': 'Start with beginner content, then intermediate, finish with advanced'
            }
        }
        
        with open(metadata_filename, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
        
        # Success message with actionable next steps
        print(f"\n🎉 **SUCCESS! Your study guide is ready!**")
        print(f"{'='*60}")
        print(f"📖 **Study Guide:** {filename}")
        print(f"📊 **Analytics:** {metadata_filename}")
        print(f"📄 **Pages Processed:** {len(metadata)}")
        print(f"📚 **Study Sections:** {len(notes_sections)}")
        print(f"⏱️  **Estimated Study Time:** {content_stats['total_read_time']} minutes")
        print(f"🎯 **Recommended Sessions:** {max(1, content_stats['total_read_time'] // 30)}")
        
        print(f"\n🚀 **Next Steps:**")
        print(f"1. 📖 Open {filename} in your favorite markdown viewer")
        print(f"2. 🎯 Start with the Table of Contents for overview")
        print(f"3. 💻 Have your development environment ready")
        print(f"4. ✅ Follow the progressive learning path")
        print(f"5. 🎪 Try all the interactive examples!")
        
        # Quick study tips
        print(f"\n💡 **Study Tips:**")
        print(f"- 🔥 Focus on one section at a time")
        print(f"- 💻 Practice every code example")
        print(f"- 📝 Take notes on key insights")
        print(f"- 🎯 Complete the checkpoint challenges")
        print(f"- 🤝 Share your learnings with others")
        
    except IOError as e:
        logger.error(f"Failed to write to file {filename}: {e}")
        print(f"\n❌ **Error:** Could not save study guide to file.")
        print(f"💡 **Suggestion:** Check file permissions and disk space.")

def main():
    # Tell the function to use the global variable from the start
    global RATE_LIMIT_DELAY

    parser = argparse.ArgumentParser(
        description="🚀 Enhanced Documentation Scraper - Transform docs into engaging study guides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 **Examples:**
  %(prog)s https://docs.example.com --limit 50
    ↳ Scrape up to 50 pages for comprehensive coverage

  %(prog)s https://api.example.com/docs --limit 0
    ↳ Unlimited scraping (use with caution!)

  %(prog)s https://tutorial.example.com --limit 10
    ↳ Perfect for focused learning guides

💡 **Pro Tips:**
- Start with --limit 20 for testing
- API docs work great with higher limits
- Tutorial sites are perfect with lower limits
- Check robots.txt before scraping large sites

🎪 **What You'll Get:**
✅ Engaging, multi-level explanations (ELI5 to Expert)
✅ Practical examples and hands-on tutorials
✅ Quick reference materials and cheat sheets
✅ Self-assessment questions and exercises
✅ Structured learning progression paths
✅ Production-ready code examples
        """
    )
    parser.add_argument("url", type=str,
                       help="🌐 The starting URL of the documentation to transform")
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="📄 Maximum pages to scrape (0 = unlimited). Default: 25. Recommended: 10-50."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=RATE_LIMIT_DELAY, # Now this works correctly
        help=f"⏱️  Delay between requests in seconds. Default: {RATE_LIMIT_DELAY}. Be respectful!"
    )

    args = parser.parse_args()

    # Enhanced URL validation
    try:
        parsed = urlparse(args.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Only HTTP/HTTPS URLs supported")
    except Exception:
        print("❌ **Error:** Please provide a valid URL")
        print("💡 **Format:** https://example.com/docs")
        print("🚫 **Avoid:** File URLs, FTP, or malformed URLs")
        return

    # Load API key with helpful error messages
    api_key = APIKeyManager.get_api_key(API_KEY_FILE)
    if not api_key:
        print(f"\n❌ **SETUP REQUIRED:** API key not found!")
        print(f"{'='*60}")
        print(f"📝 **Create file:** {API_KEY_FILE}")
        print(f"🔑 **Add your key:** Get it from https://makersuite.google.com/app/apikey")
        print(f"💾 **Save the file** in the same directory as this script")
        print(f"\n💡 **The file should contain only your API key, nothing else.**")
        return

    # Update rate limit if specified
    # The 'global' declaration is already at the top
    RATE_LIMIT_DELAY = args.delay

    # Enhanced startup banner
    limit_text = f"📄 Pages: {args.limit}" if args.limit > 0 else "📄 Pages: UNLIMITED ⚠️"
    print(f"\n{'='*80}")
    print(f"🚀 **ENHANCED DOCUMENTATION SCRAPER v2.0**")
    print(f"*Transform boring docs into engaging study guides*")
    print(f"{'='*80}")
    print(f"🎯 **Target:** {args.url}")
    print(f"⚙️  **Config:** {limit_text} | Delay: {args.delay}s")
    print(f"🤖 **AI Model:** {MODEL_NAME}")
    print(f"🎪 **Output:** Interactive study guide with examples")
    print(f"{'='*80}\n")

    # Warning for unlimited scraping
    if args.limit == 0:
        print("⚠️  **WARNING:** Unlimited scraping enabled!")
        print("🤝 **Please be respectful** of the target website")
        print("💡 **Consider starting with --limit 50** for testing\n")

    # Scraping phase with progress
    print("🕷️  **Phase 1:** Intelligent web scraping in progress...")
    print("📊 **Features:** Smart content detection, priority-based crawling")

    scraper = WebsiteScraper(args.url, max_pages=args.limit)
    scraped_data, metadata = scraper.crawl()

    if not scraped_data:
        print("\n❌ **No content found!** This could be due to:")
        print("🚫 Website blocking automated access (check robots.txt)")
        print("📄 No readable content found (JavaScript-heavy site?)")
        print("🌐 Network connectivity issues")
        print("🔐 Authentication required")
        print("\n💡 **Try:**")
        print("- Check if the site loads in your browser")
        print("- Try a different starting URL")
        print("- Check your internet connection")
        return

    # Enhanced analysis phase
    print(f"\n{'='*80}")
    print(f"📊 **Phase 2:** Content analysis complete!")
    print(f"{'='*80}")

    content_types = {}
    complexity_levels = {'beginner': 0, 'intermediate': 0, 'advanced': 0}
    total_words = 0
    total_read_time = 0

    for meta in metadata.values():
        analysis = meta.get('analysis', {})
        content_type = analysis.get('type', 'general')
        complexity = analysis.get('complexity_level', 'intermediate')

        content_types[content_type] = content_types.get(content_type, 0) + 1
        complexity_levels[complexity] += 1
        total_words += analysis.get('word_count', 0)
        total_read_time += meta.get('estimated_read_time', 0)

    print(f"📚 **Content Types Found:**")
    for content_type, count in sorted(content_types.items()):
        emoji_map = {
            'tutorial': '🎯', 'concept': '🧠', 'api_reference': '📚',
            'example': '💡', 'installation': '⚙️', 'troubleshooting': '🔧',
            'general': '📄'
        }
        emoji = emoji_map.get(content_type, '📄')
        print(f"  {emoji} {content_type.replace('_', ' ').title()}: {count} pages")

    print(f"\n🎓 **Difficulty Levels:**")
    for level, count in complexity_levels.items():
        if count > 0:
            level_emoji = {'beginner': '🌱', 'intermediate': '🚀', 'advanced': '🎯'}
            print(f"  {level_emoji[level]} {level.title()}: {count} pages")

    print(f"\n📈 **Study Metrics:**")
    print(f"  📝 Words: {total_words:,}")
    print(f"  ⏰ Read time: ~{total_read_time} minutes")
    print(f"  🎯 Sessions: {max(1, total_read_time // 30)} recommended")

    # Generation phase with excitement
    print(f"\n{'='*80}")
    print("🧠 **Phase 3:** AI-powered study guide generation...")
    print("✨ **Creating:** Multi-level explanations, examples, and exercises")
    print("🎪 **This may take a few minutes for comprehensive guides**")
    print(f"{'='*80}")

    generator = EnhancedNoteGenerator(api_key)
    notes_sections = generator.generate_comprehensive_notes(scraped_data, metadata, args.url)

    # Save results with celebration
    save_comprehensive_notes(notes_sections, args.url, metadata)

    print(f"\n🎊 **MISSION ACCOMPLISHED!**")
    print(f"{'='*80}")
    print("🎓 **Your comprehensive study guide includes:**")
    print("  ✨ Multi-level explanations (ELI5 → Expert)")
    print("  💻 Copy-paste ready code examples")
    print("  📝 Quick reference cheat sheets")
    print("  🎯 Interactive learning challenges")
    print("  🗺️  Progressive skill-building paths")
    print("  🎪 Engaging analogies and explanations")
    print("  🔧 Troubleshooting and best practices")
    print(f"\n🚀 **Happy learning! You've got this!** 🎉")

if __name__ == "__main__":
    main()
                
