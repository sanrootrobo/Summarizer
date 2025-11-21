# Fixed: Summary_On_Sterioids2.py
import os
import requests
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote, unquote
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
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_CONTENT_CHARS = 500000
MAX_SEARCH_RESULTS = 8  # Maximum results per search query
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
            'is_research': False,
            'word_count': len(content.split()),
            'main_topics': ContentAnalyzer._extract_topics(content),
            'technologies': ContentAnalyzer._extract_technologies(content, url)
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
            categories['priority'] = 'high'
        
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
        if any(pattern in content_lower for pattern in ['```', 'code', 'function', 'class', 'import', 'def ', 'const ', 'var ', 'let ']):
            categories['contains_code'] = True
        
        if any(pattern in content_lower for pattern in ['example', 'for instance', 'here\'s how', 'demo', 'usage']):
            categories['contains_examples'] = True
            
        return categories
    
    @staticmethod
    def _extract_topics(content: str) -> List[str]:
        """Extract main topics from content using keyword analysis"""
        # Simple topic extraction based on common technical terms
        content_lower = content.lower()
        topics = []
        
        # Common programming/technical topics
        topic_patterns = [
            r'\b(authentication|auth)\b',
            r'\b(api|rest|graphql)\b',
            r'\b(database|db|sql|nosql)\b',
            r'\b(javascript|python|java|react|vue|angular)\b',
            r'\b(docker|kubernetes|aws|azure|gcp)\b',
            r'\b(security|encryption|ssl|https)\b',
            r'\b(testing|unit test|integration)\b',
            r'\b(deployment|ci/cd|devops)\b',
            r'\b(performance|optimization|caching)\b',
            r'\b(configuration|config|setup)\b'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, content_lower)
            topics.extend(set(matches))
        
        return list(set(topics))[:10]  # Limit to top 10 topics
    
    @staticmethod
    def _extract_technologies(content: str, url: str) -> List[str]:
        """Extract technology names from content and URL"""
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Common technologies
        tech_patterns = [
            'react', 'vue', 'angular', 'svelte', 'next.js', 'nuxt',
            'node.js', 'express', 'fastify', 'django', 'flask', 'rails',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform',
            'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
            'typescript', 'javascript', 'python', 'java', 'go', 'rust',
            'webpack', 'vite', 'babel', 'eslint', 'prettier',
            'jest', 'cypress', 'selenium', 'playwright'
        ]
        
        found_techs = []
        for tech in tech_patterns:
            if tech in content_lower or tech in url_lower:
                found_techs.append(tech)
        
        return found_techs[:8]  # Limit to top 8 technologies

class ResearchQueryGenerator:
    """Generate intelligent research queries using Gemini Flash with deep content analysis"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model=SEARCH_MODEL_NAME,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048
        )
        self.output_parser = StrOutputParser()
    
    def generate_research_queries(self, scraped_content: Dict[str, str], metadata: Dict[str, Dict], main_url: str) -> List[str]:
        """Generate intelligent research queries based on deep content analysis"""
        
        # First, deeply analyze the content to understand what it's about
        content_insights = self._extract_content_insights(scraped_content, metadata, main_url)
        
        if not content_insights['main_topics']:
            logger.warning("Could not extract meaningful content insights - using basic fallback")
            return self._generate_basic_fallback_queries(main_url)
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert research analyst. You have been given documentation content to analyze, and your job is to generate SPECIFIC, CONTEXT-AWARE search queries that will find complementary resources.

**CONTENT ANALYSIS:**
Website: {main_url}
Main Technology/Product: {main_technology}
Key Topics Covered: {main_topics}
Technologies Mentioned: {technologies}
Content Types Found: {content_types}
Specific Features/APIs: {features}
Use Cases Mentioned: {use_cases}
Integration Points: {integrations}
Current Gaps/Weaknesses: {content_gaps}

**CONTENT SAMPLE ANALYSIS:**
{content_summary}

**YOUR MISSION:**
Generate 10-12 SPECIFIC search queries that will find content to fill gaps and enhance understanding. 

**CRITICAL REQUIREMENTS:**
1. Use EXACT technology names and specific terms from the content
2. Generate queries for MISSING content types or underexplored areas
3. Focus on practical, actionable resources
4. Target high-quality sources (Stack Overflow, GitHub, tutorials, etc.)
5. Include specific feature names, API endpoints, or functionality mentioned in docs

**QUERY CATEGORIES TO INCLUDE:**
- Specific tutorial queries: "[technology] [specific feature] tutorial"
- Problem-solution queries: "[technology] [common issue] solution"
- Comparison queries: "[technology] vs [alternative] [specific aspect]"
- Integration queries: "[technology] integration with [other tech]"
- Best practice queries: "[technology] [specific area] best practices"
- Real-world example queries: "[technology] [use case] example implementation"

**AVOID:**
- Generic terms like "docs", "documentation", "guide" without specifics
- Queries that would return the original documentation site
- Overly broad queries that lack context

**OUTPUT FORMAT:**
Return ONLY a JSON array of specific, contextual search queries:
["specific query 1", "specific query 2", ...]

**EXAMPLE GOOD QUERIES (for a React docs site):**
["React hooks useState tutorial examples", "React context API best practices", "React performance optimization techniques", "React testing with Jest examples", "React TypeScript integration guide"]

Generate specific, contextual research queries:
""")
        
        chain = prompt | self.llm | self.output_parser
        
        try:
            logger.info("🧠 Analyzing content deeply to generate context-aware research queries...")
            result = chain.invoke(content_insights)
            
            # Parse JSON response
            try:
                queries = json.loads(result.strip())
                if isinstance(queries, list) and queries:
                    # Validate queries are specific and contextual
                    validated_queries = self._validate_contextual_queries(queries, content_insights)
                    logger.info(f"✅ Generated {len(validated_queries)} context-aware research queries")
                    return validated_queries[:12]
                else:
                    logger.warning("No valid queries returned - using intelligent fallback")
                    return self._generate_intelligent_fallback_queries(content_insights)
            except json.JSONDecodeError:
                logger.warning("Failed to parse generated queries - using intelligent fallback")
                return self._generate_intelligent_fallback_queries(content_insights)
                
        except Exception as e:
            logger.error(f"Error generating research queries: {e}")
            return self._generate_intelligent_fallback_queries(content_insights)
    
    def _extract_content_insights(self, scraped_content: Dict[str, str], metadata: Dict[str, Dict], main_url: str) -> Dict[str, str]:
        """Extract deep insights from scraped content to understand what it's actually about"""
        
        # Combine all content for analysis
        all_text = ""
        url_paths = set()
        page_titles = []
        
        for url, content in scraped_content.items():
            all_text += content + " "
            parsed_url = urlparse(url)
            url_paths.update(parsed_url.path.split('/'))
            
            meta = metadata.get(url, {})
            title = meta.get('title', '')
            if title and title != 'Untitled Page':
                page_titles.append(title.lower())
        
        # Extract main technology/product name
        main_tech = self._extract_main_technology(main_url, all_text, page_titles)
        
        # Extract specific topics and features
        topics = self._extract_specific_topics(all_text, url_paths)
        technologies = self._extract_technologies_advanced(all_text)
        features = self._extract_features_and_apis(all_text)
        use_cases = self._extract_use_cases(all_text)
        integrations = self._extract_integration_points(all_text)
        
        # Analyze content types and gaps
        content_types = {}
        for meta in metadata.values():
            analysis = meta.get('analysis', {})
            content_type = analysis.get('type', 'general')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        content_gaps = self._identify_content_gaps(content_types, topics, features)
        
        # Create content summary
        content_summary = self._create_focused_content_summary(all_text[:2000])  # First 2000 chars
        
        return {
            'main_url': main_url,
            'main_technology': main_tech,
            'main_topics': ', '.join(topics[:10]),
            'technologies': ', '.join(technologies[:8]),
            'features': ', '.join(features[:10]),
            'use_cases': ', '.join(use_cases[:8]),
            'integrations': ', '.join(integrations[:6]),
            'content_types': ', '.join([f"{k}({v})" for k, v in content_types.items()]),
            'content_gaps': content_gaps,
            'content_summary': content_summary
        }
    
    def _extract_main_technology(self, main_url: str, content: str, titles: List[str]) -> str:
        """Extract the main technology/product name from URL and content"""
        domain = urlparse(main_url).netloc.replace('www.', '').lower()
        
        # Try to extract from domain
        domain_parts = domain.split('.')
        if len(domain_parts) >= 2:
            potential_tech = domain_parts[0]
            # Skip generic terms
            if potential_tech not in ['docs', 'api', 'developer', 'help', 'support']:
                return potential_tech
        
        # Try to extract from content and titles
        content_lower = content.lower()
        
        # Common tech patterns
        tech_patterns = [
            r'\b(react|vue|angular|svelte|next\.?js|nuxt)\b',
            r'\b(node\.?js|express|fastify|django|flask|rails)\b',
            r'\b(docker|kubernetes|aws|azure|gcp|terraform)\b',
            r'\b(mongodb|postgresql|mysql|redis|elasticsearch)\b',
            r'\b(typescript|javascript|python|java|golang|rust)\b',
            r'\b(webpack|vite|babel|eslint|prettier)\b',
            r'\b(graphql|rest|api|sdk)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                return matches[0].replace('.', '')
        
        # Fallback to domain
        return domain_parts[0] if domain_parts else 'unknown'
    
    def _extract_specific_topics(self, content: str, url_paths: set) -> List[str]:
        """Extract specific topics, not just generic categories"""
        content_lower = content.lower()
        topics = set()
        
        # Extract from URL paths
        meaningful_paths = [p for p in url_paths if len(p) > 2 and p not in ['', 'docs', 'api', 'v1', 'v2']]
        topics.update(meaningful_paths[:5])
        
        # Technical topic patterns
        topic_patterns = {
            r'\b(authentication|auth|oauth|jwt|login|session)\b': 'authentication',
            r'\b(api|rest|graphql|endpoint|request|response)\b': 'api',
            r'\b(database|db|sql|query|orm|migration)\b': 'database',
            r'\b(deployment|deploy|ci/cd|pipeline|build)\b': 'deployment',
            r'\b(testing|test|unit test|integration|e2e)\b': 'testing',
            r'\b(security|ssl|https|encryption|vulnerability)\b': 'security',
            r'\b(performance|optimization|caching|speed)\b': 'performance',
            r'\b(configuration|config|setup|installation)\b': 'configuration',
            r'\b(middleware|plugin|extension|addon)\b': 'middleware',
            r'\b(routing|routes|navigation|url)\b': 'routing',
            r'\b(state|store|redux|context|props)\b': 'state-management',
            r'\b(component|widget|element|ui)\b': 'components',
            r'\b(hooks|lifecycle|event|callback)\b': 'lifecycle',
            r'\b(validation|form|input|schema)\b': 'validation',
            r'\b(error|exception|handling|debug)\b': 'error-handling'
        }
        
        for pattern, topic in topic_patterns.items():
            if re.search(pattern, content_lower):
                topics.add(topic)
        
        return list(topics)[:15]
    
    def _extract_technologies_advanced(self, content: str) -> List[str]:
        """Extract specific technologies mentioned in content"""
        content_lower = content.lower()
        technologies = set()
        
        # Framework and library patterns
        tech_patterns = [
            r'\b(react|reactjs|react\.js)\b',
            r'\b(vue|vuejs|vue\.js)\b',
            r'\b(angular|angularjs)\b',
            r'\b(svelte|sveltekit)\b',
            r'\b(next\.?js|nextjs)\b',
            r'\b(nuxt\.?js|nuxtjs)\b',
            r'\b(node\.?js|nodejs)\b',
            r'\b(express\.?js|express)\b',
            r'\b(fastify)\b',
            r'\b(django)\b',
            r'\b(flask)\b',
            r'\b(rails|ruby on rails)\b',
            r'\b(docker)\b',
            r'\b(kubernetes|k8s)\b',
            r'\b(mongodb|mongo)\b',
            r'\b(postgresql|postgres)\b',
            r'\b(mysql)\b',
            r'\b(redis)\b',
            r'\b(elasticsearch|elastic)\b',
            r'\b(typescript|ts)\b',
            r'\b(javascript|js)\b',
            r'\b(python)\b',
            r'\b(java)\b',
            r'\b(golang|go)\b',
            r'\b(rust)\b',
            r'\b(webpack)\b',
            r'\b(vite)\b',
            r'\b(babel)\b',
            r'\b(jest)\b',
            r'\b(cypress)\b',
            r'\b(graphql)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                # Normalize the match
                tech_name = matches[0].replace('.js', '').replace('.', '')
                technologies.add(tech_name)
        
        return list(technologies)[:10]
    
    def _extract_features_and_apis(self, content: str) -> List[str]:
        """Extract specific features, API endpoints, or functionality"""
        features = set()
        
        # Look for API endpoints
        api_patterns = [
            r'/api/v?\d*/([a-zA-Z]+)',
            r'\.([a-zA-Z]+)\(',  # Method calls
            r'<([A-Z][a-zA-Z]+)>',  # Component names
            r'@([a-zA-Z]+)',  # Decorators/annotations
            r'--([a-z\-]+)',  # CLI flags
        ]
        
        for pattern in api_patterns:
            matches = re.findall(pattern, content)
            features.update([m for m in matches if len(m) > 2])
        
        # Look for specific feature keywords
        feature_keywords = [
            'useState', 'useEffect', 'useContext', 'useReducer',  # React hooks
            'middleware', 'router', 'controller', 'model',  # Backend patterns
            'component', 'directive', 'service', 'pipe',  # Angular
            'computed', 'watch', 'mounted', 'created',  # Vue
            'async', 'await', 'promise', 'callback',  # Async patterns
        ]
        
        content_lower = content.lower()
        for keyword in feature_keywords:
            if keyword.lower() in content_lower:
                features.add(keyword)
        
        return list(features)[:15]
    
    def _extract_use_cases(self, content: str) -> List[str]:
        """Extract specific use cases or application scenarios"""
        use_cases = set()
        content_lower = content.lower()
        
        # Use case patterns
        use_case_patterns = {
            r'\b(e-?commerce|shopping|store|cart)\b': 'ecommerce',
            r'\b(blog|cms|content management)\b': 'cms',
            r'\b(dashboard|admin|analytics)\b': 'dashboard',
            r'\b(chat|messaging|real-?time)\b': 'realtime-app',
            r'\b(mobile|responsive|pwa)\b': 'mobile-app',
            r'\b(authentication|login|user management)\b': 'user-auth',
            r'\b(payment|billing|subscription)\b': 'payments',
            r'\b(social|media|sharing)\b': 'social-media',
            r'\b(search|filter|query)\b': 'search',
            r'\b(file upload|storage|media)\b': 'file-handling',
            r'\b(notification|email|sms)\b': 'notifications',
            r'\b(microservice|api|backend)\b': 'backend-service'
        }
        
        for pattern, use_case in use_case_patterns.items():
            if re.search(pattern, content_lower):
                use_cases.add(use_case)
        
        return list(use_cases)[:10]
    
    def _extract_integration_points(self, content: str) -> List[str]:
        """Extract integration points with other technologies"""
        integrations = set()
        content_lower = content.lower()
        
        integration_patterns = [
            r'\bintegrat\w* with ([a-zA-Z]+)',
            r'\bconnect\w* to ([a-zA-Z]+)',
            r'\busing ([a-zA-Z]+) with',
            r'\b([a-zA-Z]+) integration',
            r'\b([a-zA-Z]+) plugin',
            r'\b([a-zA-Z]+) adapter'
        ]
        
        for pattern in integration_patterns:
            matches = re.findall(pattern, content_lower)
            integrations.update([m for m in matches if len(m) > 2])
        
        return list(integrations)[:8]
    
    def _identify_content_gaps(self, content_types: Dict[str, int], topics: List[str], features: List[str]) -> str:
        """Identify what types of content are missing or underrepresented"""
        gaps = []
        
        # Check for missing content types
        expected_types = ['tutorial', 'example', 'api_reference', 'concept']
        for expected_type in expected_types:
            if content_types.get(expected_type, 0) < 2:
                gaps.append(f"needs more {expected_type} content")
        
        # Check for specific gaps based on features
        if features and content_types.get('example', 0) < 3:
            gaps.append("needs practical examples")
        
        if topics and content_types.get('tutorial', 0) < 2:
            gaps.append("needs step-by-step tutorials")
        
        return ', '.join(gaps) if gaps else 'comprehensive coverage'
    
    def _create_focused_content_summary(self, content_sample: str) -> str:
        """Create a focused summary of what the content is actually about"""
        # Extract key sentences that contain specific technical information
        sentences = content_sample.split('.')
        key_sentences = []
        
        for sentence in sentences[:10]:  # First 10 sentences
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(word in sentence.lower() for word in ['api', 'function', 'method', 'component', 'service', 'library', 'framework'])):
                key_sentences.append(sentence)
        
        return '. '.join(key_sentences[:3]) if key_sentences else content_sample[:500]
    
    def _validate_contextual_queries(self, queries: List[str], content_insights: Dict[str, str]) -> List[str]:
        """Validate that queries are specific and contextual, not generic"""
        main_tech = content_insights.get('main_technology', '').lower()
        topics = content_insights.get('main_topics', '').lower().split(', ')
        
        validated_queries = []
        
        for query in queries:
            query = query.strip()
            query_lower = query.lower()
            
            # Skip if too generic
            generic_terms = ['docs', 'documentation', 'guide', 'help', 'manual']
            if any(term in query_lower for term in generic_terms) and main_tech not in query_lower:
                continue
            
            # Skip if too short or too long
            if len(query) < 15 or len(query) > 80:
                continue
            
            # Prefer queries that contain specific technology or topic names
            if (main_tech and main_tech in query_lower) or any(topic in query_lower for topic in topics if topic):
                validated_queries.append(query)
            elif len(validated_queries) < 6:  # Accept some less specific queries if we need more
                validated_queries.append(query)
        
        return validated_queries
    
    def _generate_intelligent_fallback_queries(self, content_insights: Dict[str, str]) -> List[str]:
        """Generate intelligent fallback queries based on extracted insights"""
        main_tech = content_insights.get('main_technology', 'programming')
        topics = [t.strip() for t in content_insights.get('main_topics', '').split(',') if t.strip()][:5]
        technologies = [t.strip() for t in content_insights.get('technologies', '').split(',') if t.strip()][:3]
        features = [f.strip() for f in content_insights.get('features', '').split(',') if f.strip()][:5]
        
        queries = []
        
        # Technology-specific queries
        if main_tech and main_tech != 'unknown':
            queries.extend([
                f"{main_tech} tutorial examples",
                f"{main_tech} best practices guide",
                f"{main_tech} common problems solutions",
                f"how to use {main_tech} beginners"
            ])
        
        # Topic-specific queries
        for topic in topics:
            if topic and len(topic) > 2:
                queries.extend([
                    f"{main_tech} {topic} tutorial",
                    f"{topic} implementation guide"
                ])
        
        # Technology integration queries
        for tech in technologies:
            if tech and tech != main_tech:
                queries.extend([
                    f"{main_tech} {tech} integration",
                    f"{main_tech} with {tech} example"
                ])
        
        # Feature-specific queries
        for feature in features:
            if feature and len(feature) > 3:
                queries.append(f"{main_tech} {feature} tutorial")
        
        # Remove duplicates and limit
        return list(dict.fromkeys(queries))[:12]
    
    def _generate_basic_fallback_queries(self, main_url: str) -> List[str]:
        """Basic fallback when content analysis fails"""
        domain = urlparse(main_url).netloc.replace('www.', '')
        base_term = domain.split('.')[0]
        
        return [
            f"{base_term} tutorial examples",
            f"{base_term} getting started guide",
            f"{base_term} best practices",
            f"how to use {base_term}",
            f"{base_term} common issues solutions",
            f"{base_term} integration examples"
        ]

class GoogleSearchResearcher:
    """Perform Google searches and extract useful URLs with improved DuckDuckGo fallback"""
    
    def __init__(self, api_key: str = None, cx_id: str = None):
        self.api_key = api_key
        self.cx_id = cx_id
        self.google_base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        # --- FIX START ---
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        # Track API usage to respect rate limits
        self.google_requests_made = 0
        self.max_google_requests_per_session = 90  # Stay well under 100/day limit
        self.last_google_request_time = 0
        # --- FIX END ---
    
    def search_and_extract_urls(self, queries: List[str], exclude_domain: str = None) -> List[str]:
        """Perform searches using Google first with rate limiting, fallback to improved DuckDuckGo"""
        all_urls = set()
        google_available = bool(self.api_key and self.cx_id)
        
        for i, query in enumerate(queries):
            logger.info(f"🔍 Searching [{i+1}/{len(queries)}]: {query}")
            
            urls_found = False
            
            # Try Google Search first if credentials available and under rate limit
            if (google_available and 
                self.google_requests_made < self.max_google_requests_per_session):
                try:
                    google_urls = self._search_google_with_rate_limit(query, exclude_domain)
                    if google_urls:
                        all_urls.update(google_urls)
                        urls_found = True
                        logger.info(f"✅ Google found {len(google_urls)} URLs for: {query}")
                    else:
                        logger.info(f"⚠️ Google returned no results for: {query}")
                except Exception as e:
                    logger.warning(f"❌ Google search failed for '{query}': {e}")
                    # If we hit rate limits, disable Google for remaining queries
                    if "429" in str(e) or "quota" in str(e).lower():
                        logger.warning("🚫 Google API rate limit reached - switching to DuckDuckGo only")
                        google_available = False
            
            # Fallback to improved DuckDuckGo if Google failed or not available
            if not urls_found:
                try:
                    logger.info(f"🦆 Using DuckDuckGo fallback for: {query}")
                    ddg_urls = self._search_duckduckgo_improved(query, exclude_domain)
                    if ddg_urls:
                        all_urls.update(ddg_urls)
                        logger.info(f"✅ DuckDuckGo found {len(ddg_urls)} URLs for: {query}")
                    else:
                        logger.info(f"⚠️ DuckDuckGo returned no results for: {query}")
                except Exception as e:
                    logger.warning(f"❌ DuckDuckGo search also failed for '{query}': {e}")
            
            # Be polite to search engines
            sleep(1.0 if google_available else 0.8)
        
        filtered_urls = list(all_urls)[:MAX_RESEARCH_PAGES]
        logger.info(f"🎯 Total research URLs collected: {len(filtered_urls)} from {len(queries)} searches")
        logger.info(f"📊 Google requests used: {self.google_requests_made}/{self.max_google_requests_per_session}")
        return filtered_urls
    
    def _search_google_with_rate_limit(self, query: str, exclude_domain: str = None) -> List[str]:
        """Search using Google Custom Search API with proper rate limiting"""
        import time
        
        # Implement rate limiting (max 1 request per second)
        current_time = time.time()
        time_since_last_request = current_time - self.last_google_request_time
        if time_since_last_request < 1.0:
            sleep(1.0 - time_since_last_request)
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.cx_id,
                'q': query,
                'num': min(MAX_SEARCH_RESULTS, 10)
            }
            
            response = self.session.get(self.google_base_url, params=params, timeout=REQUEST_TIMEOUT)
            self.last_google_request_time = time.time()
            self.google_requests_made += 1
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("Google API rate limit hit - will use DuckDuckGo for remaining queries")
                raise Exception("Google API rate limit exceeded")
            
            response.raise_for_status()
            
            results = response.json()
            urls = []
            
            if 'items' in results:
                for item in results['items']:
                    url = item.get('link', '')
                    if self._is_useful_url(url, exclude_domain):
                        urls.append(url)
            elif 'error' in results:
                error = results['error']
                if error.get('code') == 429:
                    raise Exception("Google API quota exceeded")
                else:
                    raise Exception(f"Google API error: {error.get('message', 'Unknown error')}")
            
            return urls
            
        except requests.RequestException as e:
            if "429" in str(e):
                raise Exception("Google API rate limit exceeded")
            else:
                raise Exception(f"Google API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from Google: {e}")
    
    def _search_google(self, query: str, exclude_domain: str = None) -> List[str]:
        """Legacy method - now redirects to rate-limited version"""
        return self._search_google_with_rate_limit(query, exclude_domain)
    
    def _search_duckduckgo_improved(self, query: str, exclude_domain: str = None) -> List[str]:
        """Improved DuckDuckGo search with better parsing and multiple strategies"""
        urls = []
        
        # Strategy 1: Try DuckDuckGo HTML search
        try:
            urls.extend(self._ddg_html_search(query, exclude_domain))
        except Exception as e:
            logger.debug(f"DDG HTML search failed: {e}")
        
        # Strategy 2: If not enough results, try lite version
        if len(urls) < 3:
            try:
                lite_urls = self._ddg_lite_search(query, exclude_domain)
                urls.extend([url for url in lite_urls if url not in urls])
            except Exception as e:
                logger.debug(f"DDG Lite search failed: {e}")
        
        return urls[:MAX_SEARCH_RESULTS]
    
    def _ddg_html_search(self, query: str, exclude_domain: str = None) -> List[str]:
        """DuckDuckGo HTML search with improved parsing"""
        search_url = "https://html.duckduckgo.com/html/"
        
        # Enhance query with site operators for better results
        enhanced_query = f"{query} (site:stackoverflow.com OR site:github.com OR site:medium.com OR site:dev.to)"
        
        params = {
            'q': enhanced_query,
            'b': '',  # Start index
            'kl': 'us-en',  # Language
            'df': '',  # Date filter
            's': '0',  # Start position
            'v': 'l',  # Layout
            'o': 'json',  # Output (though we parse HTML)
            'dc': '1'  # More results
        }
        
        headers = self.session.headers.copy()
        headers.update({
            'Referer': 'https://duckduckgo.com/',
            'DNT': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin'
        })
        
        response = self.session.get(search_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []
        
        # Try multiple selectors for result links
        link_selectors = [
            'a.result__a',  # Standard results
            '.result__body a[href^="http"]',  # Alternative selector
            '.results_links_deep a[href^="http"]',  # Deep links
            'a[href*="uddg="]'  # DuckDuckGo wrapped links
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            
            for link in links:
                href = link.get('href', '')
                
                # Handle DuckDuckGo wrapped URLs
                if 'uddg=' in href:
                    try:
                        # Extract the real URL from DuckDuckGo wrapper
                        import urllib.parse
                        if href.startswith('/l/?uddg='):
                            encoded_url = href.split('uddg=')[1].split('&')[0]
                            real_url = unquote(encoded_url)
                        else:
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                            real_url = parsed.get('uddg', [''])[0]
                        
                        if real_url and self._is_useful_url(real_url, exclude_domain):
                            urls.append(real_url)
                    except Exception:
                        continue
                
                # Handle direct URLs
                elif href.startswith('http') and self._is_useful_url(href, exclude_domain):
                    urls.append(href)
            
            if urls:  # If we found results with this selector, stop trying others
                break
        
        # If enhanced query didn't work, try simple query
        if not urls and enhanced_query != query:
            params['q'] = query
            response = self.session.get(search_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    
                    if 'uddg=' in href:
                        try:
                            if href.startswith('/l/?uddg='):
                                encoded_url = href.split('uddg=')[1].split('&')[0]
                                real_url = unquote(encoded_url)
                            else:
                                import urllib.parse
                                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                                real_url = parsed.get('uddg', [''])[0]
                            
                            if real_url and self._is_useful_url(real_url, exclude_domain):
                                urls.append(real_url)
                        except Exception:
                            continue
                    elif href.startswith('http') and self._is_useful_url(href, exclude_domain):
                        urls.append(href)
                
                if urls:
                    break
        
        return list(dict.fromkeys(urls))  # Remove duplicates while preserving order
    
    def _ddg_lite_search(self, query: str, exclude_domain: str = None) -> List[str]:
        """DuckDuckGo Lite search as fallback"""
        search_url = "https://lite.duckduckgo.com/lite/"
        
        params = {
            'q': query,
            'kl': 'us-en'
        }
        
        headers = self.session.headers.copy()
        headers.update({
            'Referer': 'https://lite.duckduckgo.com/',
        })
        
        response = self.session.get(search_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []
        
        # Parse lite results
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip navigation and internal links
            if href.startswith('/') or 'duckduckgo.com' in href:
                continue
            
            if href.startswith('http') and self._is_useful_url(href, exclude_domain):
                urls.append(href)
        
        return urls
    
    def _is_useful_url(self, url: str, exclude_domain: str = None) -> bool:
        """Enhanced URL filtering for better quality results"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Exclude the original domain if specified
            if exclude_domain and exclude_domain.lower() in domain:
                return False
            
            # Exclude certain file types
            excluded_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.zip', '.tar', '.gz', '.mp3', '.mp4', '.avi', '.mov']
            if any(path.endswith(ext) for ext in excluded_extensions):
                return False
            
            # Exclude low-quality domains
            excluded_domains = [
                'youtube.com', 'youtu.be', 'twitter.com', 'x.com', 'facebook.com', 'linkedin.com',
                'pinterest.com', 'instagram.com', 'tiktok.com',  # Social media
                'amazon.com', 'ebay.com', 'shopping.google.com', 'aliexpress.com',  # Shopping
                'ads.google.com', 'doubleclick.net', 'googleadservices.com',  # Ads
                'wikipedia.org',  # Sometimes too general
                'reddit.com',  # Can be useful but often low quality for technical docs
                'quora.com'   # Similar to reddit
            ]
            
            if any(excluded in domain for excluded in excluded_domains):
                return False
            
            # Strongly prefer high-quality technical domains
            high_quality_domains = [
                'stackoverflow.com', 'stackexchange.com',
                'github.com', 'gitlab.com',
                'medium.com', 'dev.to', 'hashnode.com',
                'hackernoon.com', 'freecodecamp.org',
                'css-tricks.com', 'smashingmagazine.com',
                'tutorialspoint.com', 'geeksforgeeks.org',
                'digitalocean.com', 'aws.amazon.com',
                'docs.microsoft.com', 'developer.mozilla.org',
                'nodejs.org', 'reactjs.org', 'vuejs.org',
                'atlassian.com', 'jetbrains.com'
            ]
            
            # Give high priority to quality domains
            if any(quality_domain in domain for quality_domain in high_quality_domains):
                return True
            
            # Check for quality path indicators
            quality_path_indicators = [
                'tutorial', 'guide', 'docs', 'documentation', 'wiki',
                'blog', 'learn', 'course', 'example', 'howto', 
                'getting-started', 'best-practices', 'patterns'
            ]
            
            if any(indicator in path for indicator in quality_path_indicators):
                return True
            
            # General filtering - must be HTTP/HTTPS and reasonable length
            if parsed.scheme not in ['http', 'https']:
                return False
            
            if len(url) > 500:  # Extremely long URLs are usually not useful
                return False
            
            # Exclude URLs with too many parameters (often tracking/ads)
            if len(parsed.query) > 200:
                return False
                
            return True
            
        except Exception:
            return False

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
