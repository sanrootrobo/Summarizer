# 📚 Enhanced Research & Study Guide Generator v2.2

A powerful desktop application that automatically generates comprehensive study guides from multiple sources, including websites, YouTube videos, local documents, and AI-powered web research.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER (Tkinter)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Source     │  │   Research   │  │   AI Model   │  │    Prompt    │   │
│  │   Config     │  │   Settings   │  │   Settings   │  │   Editor     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐   │
│  │   Study Guide Preview (MD)     │  │      Process Logs (Live)       │   │
│  └────────────────────────────────┘  └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                   │
│                     (AdvancedScraperApp Controller)                          │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  Input Mode Router                                              │         │
│  │  • Web Scraping Flow                                            │         │
│  │  • YouTube Video Flow (Multimodal)                              │         │
│  │  • Local Document Flow                                          │         │
│  └────────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                    │                    │
        ┌───────────┴──────────┬─────────┴─────────┬─────────┴──────────┐
        ▼                      ▼                   ▼                    ▼
┌──────────────┐      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Web Content │      │   YouTube    │    │    Local     │    │  AI Research │
│  Extraction  │      │   Analysis   │    │  Documents   │    │   Module     │
└──────────────┘      └──────────────┘    └──────────────┘    └──────────────┘
        │                      │                   │                    │
        ▼                      ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA PROCESSING LAYER                                 │
│                                                                               │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│  │  WebsiteScraper    │  │ YouTubeResearcher  │  │ LocalDocumentLoader│   │
│  │                    │  │                    │  │                    │   │
│  │  • BeautifulSoup   │  │  • yt-dlp          │  │  • PyMuPDF (PDF)   │   │
│  │  • html2text       │  │  • Transcript Ext. │  │  • python-docx     │   │
│  │  • Link Following  │  │  • Video Quality   │  │  • Text Files      │   │
│  │  • Rate Limiting   │  │    Filtering       │  │                    │   │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              RESEARCH ENHANCEMENT MODULE (Optional)                   │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────┐  ┌───────────────────────────────┐    │  │
│  │  │ EnhancedResearchQuery    │  │  Web Research Backends:       │    │  │
│  │  │ Generator                │  │  • GoogleSearchResearcher     │    │  │
│  │  │                          │  │  • PlaywrightResearcher       │    │  │
│  │  │ • Topic Extraction (AI)  │  │  • DuckDuckGo Fallback        │    │  │
│  │  │ • Query Generation (AI)  │  │                               │    │  │
│  │  │ • Diversity Optimization │  │  ┌─────────────────────────┐  │    │  │
│  │  └──────────────────────────┘  │  │ EnhancedYouTubeResearcher│ │    │  │
│  │                                 │  │ • Video Search          │ │    │  │
│  │                                 │  │ • Quality Filtering     │ │    │  │
│  │                                 │  │ • Parallel Downloads    │ │    │  │
│  │                                 │  └─────────────────────────┘  │    │  │
│  │                                 └───────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    CONTENT AGGREGATION        │
                    │  • Merge all sources          │
                    │  • Organize by type           │
                    │  • Metadata tracking          │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI GENERATION LAYER                                  │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Google Gemini AI                                   │  │
│  │                                                                        │  │
│  │  ┌───────────────────────────┐    ┌───────────────────────────────┐ │  │
│  │  │  Text-Based Generation    │    │  Multimodal Video Analysis    │ │  │
│  │  │  (via LangChain)          │    │  (Direct API)                 │ │  │
│  │  │                           │    │                               │ │  │
│  │  │  • Gemini 2.5 Flash/Pro   │    │  • Gemini 2.5 Pro             │ │  │
│  │  │  • Custom Prompts         │    │  • Video Upload & Processing  │ │  │
│  │  │  • Structured Output      │    │  • Visual + Audio Analysis    │ │  │
│  │  │  • Multiple Sources       │    │  • Transcript-free            │ │  │
│  │  └───────────────────────────┘    └───────────────────────────────┘ │  │
│  │                                                                        │  │
│  │              EnhancedNoteGenerator / Direct API Calls                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                        │
│                                                                               │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│  │  Markdown Preview  │  │  HTML Export       │  │  File Save         │   │
│  │  • Live Rendering  │  │  • Styled Output   │  │  • .md / .txt      │   │
│  │  • Syntax Colors   │  │  • Browser Open    │  │  • Timestamped     │   │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

                              EXTERNAL DEPENDENCIES
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  Core: google-generativeai, langchain, beautifulsoup4, PyMuPDF, python-docx │
│  Optional: playwright, yt-dlp, ffmpeg                                        │
│  UI: tkinter (built-in), markdown                                            │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Diagrams

### Flow 1: Web Scraping + Research Mode
```
User Input (URL) 
    │
    ▼
WebsiteScraper ──────┐
    │                │
    ▼                │
Initial Content      │
    │                │
    ▼                │
AI Topic Analysis    │
    │                │
    ▼                │
Query Generation     │
    │                │
    ├────────────────┼──────────┬──────────────┐
    ▼                ▼          ▼              ▼
Web Search    Scrape URLs   YouTube      Content Merge
Results                     Search            │
    │                │          │              │
    └────────────────┴──────────┴──────────────┤
                                                ▼
                                        AI Generation
                                                │
                                                ▼
                                        Study Guide Output
```

### Flow 2: YouTube Video (Multimodal) Mode
```
User Input (YouTube URL)
    │
    ▼
yt-dlp Download ─────> Video File (MP4)
    │
    ▼
Upload to Gemini API
    │
    ▼
Server Processing (Video + Audio)
    │
    ▼
Multimodal AI Analysis
    │
    ▼
Study Guide Output
    │
    ▼
Cleanup (Delete Local + Remote Files)
```

### Flow 3: Local Documents Mode
```
User Selects Files (PDF, DOCX, TXT)
    │
    ├────────┬────────┬────────┐
    ▼        ▼        ▼        ▼
  PyMuPDF  docx   text     (more...)
    │        │        │        │
    └────────┴────────┴────────┘
              │
              ▼
        Text Extraction
              │
              ▼
     Optional: AI Research
              │
              ▼
        AI Generation
              │
              ▼
    Study Guide Output
```

## ✨ Features

### 🎯 Multiple Input Modes
- **🌐 Web Scraping**: Crawl and extract content from websites with intelligent link following
- **🎬 YouTube Video Analysis**: Direct multimodal analysis of YouTube videos using Google's Gemini AI
- **📁 Local Documents**: Process PDF, DOCX, and TXT files

### 🔬 AI-Powered Research (Beta)
- **Intelligent Query Generation**: AI analyzes your content and generates targeted search queries
- **Web Research**: Automated web search and content extraction using:
  - Google Custom Search API (fast)
  - DuckDuckGo (free fallback)
  - Playwright browser automation (most robust)
- **YouTube Research**: Automatic video discovery and transcript extraction based on your topic
- **Smart Content Filtering**: Prioritizes high-quality, relevant sources

### 🤖 Advanced AI Generation
- **Google Gemini Integration**: Powered by Gemini 2.5 Pro/Flash models
- **Multimodal Support**: Direct video analysis without transcript extraction
- **Customizable Prompts**: Load or edit generation prompts to suit your needs
- **Comprehensive Output**: Structured study guides with summaries, key topics, examples, and resources

### 🎨 Modern Interface
- **Dark Theme**: Easy on the eyes with a sleek, professional design
- **Live Preview**: Real-time markdown rendering with syntax highlighting
- **Process Logs**: Detailed logging to track generation progress
- **Export Options**: Save as Markdown or HTML, open in browser

## 📋 Requirements

### Core Dependencies
```bash
pip install google-generativeai
pip install langchain-google-genai
pip install langchain-core
pip install beautifulsoup4
pip install requests
pip install PyYAML
pip install html2text
pip install PyMuPDF
pip install python-docx
pip install markdown
```

### Optional Dependencies
```bash
# For advanced web research
pip install playwright
playwright install

# For YouTube research
pip install yt-dlp

# Note: ffmpeg is also required for YouTube video download
# Install via your system package manager (apt, brew, choco, etc.)
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd research-study-guide-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API credentials**
   
   Create a file named `gemini_api.key` with your Google Gemini API key:
   ```
   your-api-key-here
   ```
   
   Get your API key from: https://makersuite.google.com/app/apikey

4. **Configure settings (optional)**
   
   Edit `config.yml` to customize default settings:
   ```yaml
   api:
     key_file: "gemini_api.key"
     google_search:
       key_file: "google_api.key"  # Optional
       cx_file: "google_cx.key"     # Optional
   
   llm:
     model_name: "gemini-2.5-pro"
     parameters:
       temperature: 0.5
       max_output_tokens: 8192
   
   scraper:
     rate_limit_delay: 0.5
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## 📖 Usage Guide

### Basic Workflow

1. **Choose Input Mode**
   - 🌐 Web Scraper: Enter a URL and set crawl limit
   - 🎬 YouTube Video: Paste a YouTube URL for direct analysis
   - 📁 Local Documents: Add PDF, DOCX, or TXT files

2. **Configure Research (Optional)**
   - Enable AI-powered research for deeper content
   - Choose web search method (API vs. Playwright)
   - Enable YouTube transcript analysis
   - Set number of queries and pages to research

3. **Set AI Model Parameters**
   - Verify API key file path
   - Adjust temperature (0.0-1.0) for creativity
   - Set max output tokens

4. **Customize Prompt (Optional)**
   - Load custom prompt from file
   - Edit in-app to guide AI generation style

5. **Generate & Save**
   - Click "Start Generation"
   - Monitor progress in logs tab
   - Preview in markdown tab
   - Export as Markdown or HTML

### YouTube Video Analysis Mode

The multimodal video analysis feature allows direct processing of YouTube videos:

1. Select "🎬 YouTube Video (Multimodal)" mode
2. Paste YouTube URL
3. Ensure `yt-dlp` and `ffmpeg` are installed
4. Uses Gemini 2.5 Pro for visual + audio analysis
5. No transcript required - analyzes video content directly

**Requirements for Video Mode:**
- `yt-dlp` command-line tool
- `ffmpeg` for video processing
- Gemini 2.5 Pro or compatible multimodal model

## 🔧 Configuration

### API Keys

**Gemini API (Required)**
- File: `gemini_api.key`
- Get from: https://makersuite.google.com/app/apikey

**Google Custom Search (Optional)**
- Files: `google_api.key`, `google_cx.key`
- Used for faster web research
- Get from: https://developers.google.com/custom-search

### Prompt Customization

Create a `prompt.md` file with your custom prompt template. Variables available:
- `{content}`: The collected source content
- `{website_url}`: Primary source name/URL
- `{source_count}`: Number of sources processed
- `{metadata}`: Generation metadata

Example:
```markdown
You are an expert tutor creating study materials for students.

Analyze the following content and create a comprehensive study guide with:
1. Executive Summary
2. Key Concepts (with definitions)
3. Important Examples
4. Practice Questions
5. Additional Resources

Content:
{content}

Source: {website_url}
```

## 🎯 Use Cases

- **Students**: Generate study guides from course websites, video lectures, and textbooks
- **Researchers**: Compile comprehensive overviews from multiple sources
- **Content Creators**: Research topics thoroughly with AI assistance
- **Educators**: Create teaching materials from diverse content sources
- **Self-Learners**: Build structured learning resources on any topic

## 🐛 Troubleshooting

### Common Issues

**"Dependencies missing" warning**
- Install all required packages: `pip install -r requirements.txt`

**Playwright not available**
- Install: `pip install playwright && playwright install`
- Use Google API or DuckDuckGo fallback

**YouTube research not working**
- Install: `pip install yt-dlp`
- Ensure videos have English captions/transcripts

**Video download fails**
- Install `ffmpeg` via system package manager
- Check internet connection
- Verify YouTube URL is valid

**API rate limits**
- Add delays between requests in `config.yml`
- Use Google Custom Search API for better limits

## 📊 Features Comparison

| Feature | Web Scraper | YouTube Video | Local Docs |
|---------|-------------|---------------|------------|
| Single Source | ✅ | ✅ | ✅ |
| Multiple Sources | ✅ | ❌ | ✅ |
| AI Research | ✅ | ❌ | ✅ |
| Web Research | ✅ | ❌ | ✅ |
| YouTube Research | ✅ | ❌ | ✅ |
| Multimodal Analysis | ❌ | ✅ | ❌ |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Google Gemini AI](https://deepmind.google/technologies/gemini/)
- Uses [LangChain](https://python.langchain.com/) for AI orchestration
- Web scraping powered by [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- Browser automation via [Playwright](https://playwright.dev/)
- Video processing with [yt-dlp](https://github.com/yt-dlp/yt-dlp)

