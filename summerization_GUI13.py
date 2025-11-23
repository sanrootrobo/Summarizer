# (Keep all other classes like DarkThemeManager, APIKeyManager, etc., the same)
import collections # Import collections for the deque in the scraper

# ... (Previous classes remain unchanged) ...

# REPLACE THE ENTIRE WebsiteScraper CLASS WITH THE NEW VERSION BELOW
class WebsiteScraper:
    """ An intelligent web scraper that uses an LLM to select relevant links when crawling, with support for deep crawling. """
    def __init__(self, base_url: str, max_pages: int, user_agent: str, request_timeout: int, rate_limit_delay: float, gemini_api_key: Optional[str] = None, deep_crawl: bool = False, crawl_depth: int = 0):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.request_timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay
        self.scraped_content = {}
        self.visited_urls = set()
        self.gemini_api_key = gemini_api_key
        self.llm = None
        self.crawl_topic = ""
        # --- NEW: Parameters for deep crawling ---
        self.deep_crawl = deep_crawl
        self.crawl_depth = crawl_depth if crawl_depth > 0 else float('inf') # Treat 0 as infinite

        if self.gemini_api_key and DEPENDENCIES_AVAILABLE:
            try:
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.gemini_api_key, temperature=0.0)
                logging.info("🧠 WebsiteScraper initialized with LLM for intelligent link selection.")
            except Exception as e:
                logging.warning(f"⚠️ Could not initialize LLM for scraper, will use basic selection. Error: {e}")
                self.llm = None

    def _get_crawl_topic(self, initial_content: str) -> str:
        if not self.llm: return "general information"
        try:
            prompt = ChatPromptTemplate.from_template("Summarize the main topic of the following text in a short, concise phrase. Examples: 'asynchronous programming in Python', 'React state management libraries'.\n\nTEXT CONTENT:\n---\n{content}\n---\n\nMain Topic:")
            chain = prompt | self.llm | StrOutputParser()
            topic = chain.invoke({"content": initial_content})
            return topic.strip()
        except Exception as e:
            logging.error(f"❌ Could not determine crawl topic with LLM: {e}")
            return "general information"

    def _select_relevant_urls_with_llm(self, links_with_text: List[Tuple[str, str]], limit: int) -> List[str]:
        if not self.llm or not self.crawl_topic: return [link[0] for link in links_with_text[:limit]]
        formatted_links = "\n".join([f"- URL: {url}\n  Link Text: '{text}'" for url, text in links_with_text])
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an intelligent web crawler. Based on the **Main Topic**, select the **{limit}** most relevant URLs from the list. Prioritize tutorials, guides, and core features. Avoid 'contact', 'about', 'pricing', 'login'.\n\n"
                "**Main Topic:** {topic}\n\n"
                "**Candidate Links:**\n{links}\n\n"
                "Respond ONLY with a JSON list of URL strings. Example: [\"https://.../url1\", \"https://.../url2\"]"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"topic": self.crawl_topic, "links": formatted_links, "limit": limit})
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                selected_urls = json.loads(json_match.group(0))
                if isinstance(selected_urls, list) and all(isinstance(i, str) for i in selected_urls):
                    return selected_urls[:limit]
            logging.warning("⚠️ LLM response for URL selection was not valid JSON. Falling back.")
        except Exception as e:
            logging.error(f"❌ LLM URL selection failed: {e}. Falling back.")
        return [link[0] for link in links_with_text[:limit]]

    def _get_page_content(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''): return response.text
        except requests.exceptions.RequestException as e: logging.warning(f"Request failed for {url}: {e}")
        return None

    def _parse_content_and_links(self, html: str, page_url: str) -> tuple[str, list[tuple[str, str]]]:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup.select('script, style, nav, footer, header, aside, form, .ad, .advertisement'): element.decompose()
        main_content_area = (soup.select_one('main, article, [role="main"], .main-content, .content, .post-content') or soup.body)
        text_maker = html2text.HTML2Text(); text_maker.body_width = 0; text_maker.ignore_links = True; text_maker.ignore_images = True
        markdown_content = text_maker.handle(str(main_content_area)) if main_content_area else ""
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content).strip()
        markdown_content = re.sub(r'[ \t]+', ' ', markdown_content)
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith(('#', 'mailto:', 'tel:')): continue
            full_url = urljoin(page_url, href).split('#')[0]
            if urlparse(full_url).scheme in ['http', 'https'] and urlparse(full_url).netloc == self.domain:
                anchor_text = a.get_text(strip=True)
                if anchor_text and len(anchor_text) > 2:
                    links.add((full_url, anchor_text))
        return markdown_content, list(links)

    def crawl(self, additional_urls: Optional[List[str]] = None) -> dict[str, str]:
        # Mode 1: Flat crawl of external URLs (for research) - This logic remains unchanged.
        if additional_urls:
            logging.info(f"Starting flat crawl of {len(additional_urls)} external research URLs...")
            for url in additional_urls:
                if len(self.scraped_content) >= self.max_pages: break
                if url in self.visited_urls: continue
                self.visited_urls.add(url)
                logging.info(f"[{len(self.scraped_content) + 1}/{self.max_pages}] Research page: {url}")
                html = self._get_page_content(url)
                if html:
                    text, _ = self._parse_content_and_links(html, url)
                    if text and len(text.strip()) > 150: self.scraped_content[url] = text
                sleep(self.rate_limit_delay)
            logging.info(f"Research crawling complete. Scraped {len(self.scraped_content)} pages.")
            return self.scraped_content

        # --- Mode 2: Primary scraping from base_url with depth control ---
        if self.deep_crawl:
             logging.info(f"🚀 Starting DEEP crawl from {self.base_url} (Depth: {self.crawl_depth if self.crawl_depth != float('inf') else 'Infinite'})")
        else:
             logging.info(f"🚀 Starting ONE-LEVEL crawl from {self.base_url}")

        # Use a deque for efficient breadth-first search, storing (url, depth)
        urls_to_visit = collections.deque([(self.base_url, 0)])

        while urls_to_visit and len(self.scraped_content) < self.max_pages:
            url, current_depth = urls_to_visit.popleft()

            if url in self.visited_urls: continue
            self.visited_urls.add(url)

            logging.info(f"[{len(self.scraped_content) + 1}/{self.max_pages}][Depth:{current_depth}] Scraping: {url}")

            html = self._get_page_content(url)
            if not html: continue

            text, new_links_with_text = self._parse_content_and_links(html, url)
            if text and len(text.strip()) > 150:
                self.scraped_content[url] = text

            # --- Crawling Logic ---
            # 1. Establish crawl topic on the first page
            if not self.crawl_topic and self.llm and text:
                self.crawl_topic = self._get_crawl_topic(text[:4000])
                logging.info(f"🧠 Determined crawl context: '{self.crawl_topic}'")

            # 2. Decide whether to follow links from the current page
            should_add_links = self.deep_crawl or current_depth == 0

            # 3. Check if we've reached the maximum depth
            if current_depth >= self.crawl_depth:
                should_add_links = False

            if should_add_links and new_links_with_text:
                remaining_capacity = self.max_pages - len(self.scraped_content)
                unvisited_links = [(u, t) for u, t in new_links_with_text if u not in self.visited_urls and u not in [item[0] for item in urls_to_visit]]

                if not unvisited_links: continue

                # Use LLM to select links if we have more options than capacity
                if self.llm and self.crawl_topic and len(unvisited_links) > remaining_capacity:
                    logging.info(f"🧠 Found {len(unvisited_links)} links, exceeds capacity. Using LLM to select...")
                    selected_urls = self._select_relevant_urls_with_llm(unvisited_links, limit=remaining_capacity)
                    logging.info(f"  ✅ LLM selected {len(selected_urls)} URLs to visit next.")
                else: # Otherwise, take them in order
                    selected_urls = [u for u, t in unvisited_links[:remaining_capacity]]

                for new_url in selected_urls:
                    urls_to_visit.append((new_url, current_depth + 1))
            
            sleep(self.rate_limit_delay)

        logging.info(f"Crawl complete. Scraped {len(self.scraped_content)} total pages.")
        return self.scraped_content


# REPLACE THE ENTIRE AdvancedScraperApp CLASS WITH THIS NEW VERSION
class AdvancedScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Research & Study Guide Generator v2.3 (Deep Crawl)")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)
        DarkThemeManager.configure_dark_theme(self.root)

        self.input_mode_var = tk.StringVar(value="scrape")
        self.url_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=10) # Max pages total
        # --- NEW: tk variables for deep crawl ---
        self.deep_crawl_var = tk.BooleanVar(value=False)
        self.crawl_depth_var = tk.IntVar(value=2) # Default depth when enabled
        
        self.research_enabled_var = tk.BooleanVar(value=False)
        self.web_research_enabled_var = tk.BooleanVar(value=True)
        self.yt_research_enabled_var = tk.BooleanVar(value=True)
        self.web_research_method_var = tk.StringVar(value="google_api")
        self.research_pages_var = tk.IntVar(value=5)
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

        self.create_widgets()
        self.load_initial_settings()
        self.toggle_input_mode() # This will also call the new toggle_depth_setting
        self.toggle_research_panel()
        self.setup_logging()

    def create_widgets(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
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
        ttk.Radiobutton(mode_frame, text="🎬 YouTube Video", variable=self.input_mode_var, value="youtube_video", command=self.toggle_input_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="📁 Local Documents", variable=self.input_mode_var, value="upload", command=self.toggle_input_mode).pack(anchor=tk.W)

        self.url_input_frame = ttk.LabelFrame(parent, text="🌐 URL Input", padding=10)
        self.url_input_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(self.url_input_frame, text="Target URL:").pack(fill=tk.X, anchor='w')
        ttk.Entry(self.url_input_frame, textvariable=self.url_var, font=("Segoe UI", 9)).pack(fill=tk.X, pady=(2, 8))

        self.scraper_options_frame = ttk.Frame(self.url_input_frame)
        self.scraper_options_frame.pack(fill=tk.X, pady=(0, 0))
        ttk.Label(self.scraper_options_frame, text="Max total pages to scrape:").pack(fill=tk.X, anchor='w')
        ttk.Spinbox(self.scraper_options_frame, from_=1, to=1000, textvariable=self.limit_var, width=10).pack(fill=tk.X, pady=(2, 10))
        
        # --- NEW: Deep Crawl Controls ---
        self.deep_crawl_check = ttk.Checkbutton(self.scraper_options_frame, text="Enable Deep Crawl (Follows links recursively)", variable=self.deep_crawl_var, command=self.toggle_depth_setting)
        self.deep_crawl_check.pack(anchor=tk.W)
        
        self.depth_frame = ttk.Frame(self.scraper_options_frame)
        self.depth_frame.pack(fill=tk.X, padx=(20, 0), pady=(2, 0)) # Indent the depth setting
        self.depth_label = ttk.Label(self.depth_frame, text="Crawl Depth (0 for infinite):")
        self.depth_label.pack(side=tk.LEFT, padx=(0, 5))
        self.depth_spinbox = ttk.Spinbox(self.depth_frame, from_=0, to=50, textvariable=self.crawl_depth_var, width=8)
        self.depth_spinbox.pack(side=tk.LEFT)

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

    # --- NEW: Method to toggle the depth setting's state ---
    def toggle_depth_setting(self):
        """Enables or disables the crawl depth spinbox based on the checkbox."""
        is_enabled = self.deep_crawl_var.get()
        new_state = tk.NORMAL if is_enabled else tk.DISABLED
        self.depth_label.config(state=new_state)
        self.depth_spinbox.config(state=new_state)

    def run_full_process(self):
        """Main orchestrator for text-based generation (scrape/upload)."""
        try:
            if not DEPENDENCIES_AVAILABLE: self.show_demo_content(); return

            self.progress_var.set("Validating inputs..."); logging.info("🚀 Starting Text-Based Study Guide Generation...")
            api_key = APIKeyManager.get_gemini_api_key(self.api_key_file_var.get())
            if not api_key:
                messagebox.showerror("API Key Error", "The Gemini API key is missing or invalid."); return

            source_data, source_name, mode = {}, "", self.input_mode_var.get()
            delay = self.config.get('scraper', {}).get('rate_limit_delay', 0.5)
            user_agent = random.choice(USER_AGENTS)

            self.progress_var.set("Collecting initial content...")
            if mode == "scrape":
                logging.info("📄 Mode: Web Scraping")
                url = self.url_var.get().strip()
                if not url: messagebox.showerror("Input Error", "URL cannot be empty."); return
                if not url.startswith(('http://', 'https://')): url = 'https://' + url; self.url_var.set(url)
                
                # --- MODIFICATION: Get deep crawl settings from GUI ---
                limit = self.limit_var.get()
                deep_crawl = self.deep_crawl_var.get()
                crawl_depth = self.crawl_depth_var.get()
                
                logging.info(f"🎯 Target: {url}"); logging.info(f"📊 Max Pages: {limit} | Deep Crawl: {deep_crawl} | Depth: {crawl_depth}")
                
                scraper = WebsiteScraper(
                    url, limit, user_agent, 15, delay, 
                    gemini_api_key=api_key, 
                    deep_crawl=deep_crawl, 
                    crawl_depth=crawl_depth
                )
                source_data = scraper.crawl(); source_name = urlparse(url).netloc
            else: # mode == "upload"
                logging.info("📁 Mode: Local Document Processing")
                file_paths = list(self.file_listbox.get(0, tk.END))
                if not file_paths: messagebox.showerror("Input Error", "Please add at least one document."); return
                logging.info(f"📚 Processing {len(file_paths)} local documents")
                loader = LocalDocumentLoader(file_paths)
                source_data = loader.load_and_extract_text(); source_name = f"{len(file_paths)}_local_documents"

            # ... (The rest of run_full_process remains the same)
            if not source_data:
                logging.warning("⚠️ No content extracted from initial source. Cannot proceed.")
                messagebox.showwarning("No Content", "No text content could be extracted from the source."); return
            logging.info(f"✅ Initial content collection complete: {len(source_data)} sources")

            if self.research_enabled_var.get():
                self.progress_var.set("Conducting AI research...")
                logging.info("\n🔬 Starting AI Research Phase"); logging.info("-" * 40)
                query_generator = EnhancedResearchQueryGenerator(api_key)
                topic_data = query_generator.extract_main_topic_and_subtopics(source_data)
                research_queries = query_generator.generate_diverse_research_queries(topic_data, self.research_queries_var.get())
                logging.info(f"🎯 Generated {len(research_queries)} research queries")
                for i, query in enumerate(research_queries, 1): logging.info(f"  {i}. {query}")

                if self.web_research_enabled_var.get():
                    self.progress_var.set("Researching web sources..."); logging.info("\n🌐 Starting Web Research")
                    exclude_domain = source_name if mode == "scrape" else ""
                    if self.web_research_method_var.get() == "playwright":
                        if PLAYWRIGHT_AVAILABLE: researcher = EnhancedPlaywrightResearcher()
                        else:
                            logging.warning("⚠️ Playwright not available, falling back to API method")
                            google_api_key, google_cx = APIKeyManager.get_google_search_credentials(self.google_api_key_file_var.get(), self.google_cx_file_var.get())
                            researcher = EnhancedGoogleSearchResearcher(google_api_key, google_cx)
                    else:
                        google_api_key, google_cx = APIKeyManager.get_google_search_credentials(self.google_api_key_file_var.get(), self.google_cx_file_var.get())
                        researcher = EnhancedGoogleSearchResearcher(google_api_key, google_cx)
                    research_urls = researcher.search_and_extract_urls(research_queries, exclude_domain)
                    if research_urls:
                        research_page_limit = self.research_pages_var.get()
                        urls_to_scrape = research_urls[:research_page_limit]
                        logging.info(f"🕷️ Scraping top {len(urls_to_scrape)} research URLs (limit: {research_page_limit})")

                        research_scraper = WebsiteScraper("http://research.local", len(urls_to_scrape), user_agent, 15, delay)
                        research_data = research_scraper.crawl(additional_urls=urls_to_scrape)
                        if research_data: source_data.update(research_data); logging.info(f"✅ Added {len(research_data)} web research sources")
                    else: logging.warning("⚠️ No research URLs found from web search")

                if self.yt_research_enabled_var.get():
                    self.progress_var.set("Analyzing YouTube videos..."); logging.info("\n📺 Starting YouTube Research")
                    if YT_DLP_AVAILABLE:
                        try:
                            yt_researcher = EnhancedYouTubeResearcher()
                            video_transcripts = yt_researcher.get_transcripts_for_queries(research_queries, self.yt_videos_per_query_var.get())
                            if video_transcripts: source_data.update(video_transcripts); logging.info(f"✅ Added {len(video_transcripts)} YouTube transcript sources")
                            else: logging.warning("⚠️ No suitable YouTube videos found")
                        except Exception as e: logging.error(f"❌ YouTube research failed: {e}")
                    else: logging.warning("⚠️ YouTube research skipped - yt-dlp not available")

            self.progress_var.set("Generating study guide..."); logging.info(f"\n📝 Starting Study Guide Generation"); logging.info("-" * 40); logging.info(f"📊 Total sources: {len(source_data)}")
            llm_config = {'model_name': self.model_name_var.get(), 'parameters': {'temperature': self.temperature_var.get(), 'max_output_tokens': self.max_tokens_var.get()}}
            generator = EnhancedNoteGenerator(api_key, llm_config, self.prompt_text.get("1.0", tk.END))
            self.final_notes_content = generator.generate_comprehensive_notes(source_data, source_name)

            self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
            self.progress_var.set("Generation Complete!"); logging.info("\n🎉 Study Guide Generation Complete!")
            self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("Success!", f"Study guide generated successfully!\n\nSources processed: {len(source_data)}"))
        except Exception as e:
            logging.error(f"❌ Unexpected error in main process: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"))
            if not self.final_notes_content: self.root.after(0, lambda: self.progress_var.set("Ready"))
            else: self.root.after(0, lambda: self.progress_var.set("Complete - Ready to Save"))


    def toggle_input_mode(self):
        mode = self.input_mode_var.get()
        is_scrape_mode = (mode == "scrape")

        # Enable/disable the entire URL input frame and its children
        url_frame_state = tk.NORMAL if (is_scrape_mode or mode == "youtube_video") else tk.DISABLED
        self.url_input_frame.config(state=url_frame_state)
        for widget in self.url_input_frame.winfo_children():
            # This is a bit of a trick to enable/disable the frame's contents
            widget.master.config(state=url_frame_state)

        # Specifically manage the visibility and state of scraper options
        if is_scrape_mode:
            self.url_input_frame.config(text="🕷️ Web Scraper Input")
            self.scraper_options_frame.pack(fill=tk.X, pady=(0, 0)) # Show scraper options
            self.research_checkbutton.config(state=tk.NORMAL)
            self.toggle_depth_setting() # Set the initial state of the depth spinbox
        else:
            self.scraper_options_frame.pack_forget() # Hide scraper options
            if mode == "youtube_video":
                self.url_input_frame.config(text="🎬 YouTube Video Input")
                self.research_enabled_var.set(False)
                self.research_checkbutton.config(state=tk.DISABLED)
            
        # Handle Upload frame visibility
        upload_frame_state = tk.NORMAL if (mode == "upload") else tk.DISABLED
        self._set_child_widgets_state(self.upload_frame, upload_frame_state)
        if mode == "upload":
            self.research_checkbutton.config(state=tk.NORMAL)
        
        self.toggle_research_panel()

    # (Keep the rest of AdvancedScraperApp the same, no other changes needed)
    # ... from run_youtube_video_analysis onward ...
    def create_research_tab(self, parent):
        master_frame = ttk.LabelFrame(parent, text="🎛️ Master Control", padding=10)
        master_frame.pack(fill=tk.X, pady=(0, 10))
        self.research_checkbutton = ttk.Checkbutton(master_frame, text="🔬 Enable AI-Powered Research (Beta)", variable=self.research_enabled_var, command=self.toggle_research_panel)
        self.research_checkbutton.pack(anchor=tk.W)

        settings_frame = ttk.LabelFrame(parent, text="⚙️ Research Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(settings_frame, text="AI search queries to generate:").pack(anchor=tk.W)
        ttk.Spinbox(settings_frame, from_=3, to=15, textvariable=self.research_queries_var, width=10).pack(fill=tk.X, pady=(2, 10))

        self.web_research_panel = ttk.LabelFrame(parent, text="🌐 Web Research", padding=10)
        self.web_research_panel.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(self.web_research_panel, text="Enable Web Search Research", variable=self.web_research_enabled_var).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(self.web_research_panel, text="Search Method:").pack(anchor=tk.W)
        self.g_api_radio = ttk.Radiobutton(self.web_research_panel, text="🚀 Google API / DuckDuckGo (Fast)", variable=self.web_research_method_var, value="google_api")
        self.g_api_radio.pack(anchor=tk.W, padx=20)
        self.playwright_radio = ttk.Radiobutton(self.web_research_panel, text="🎭 Playwright Browser (Robust)", variable=self.web_research_method_var, value="playwright")
        self.playwright_radio.pack(anchor=tk.W, padx=20, pady=(0, 5))
        ttk.Label(self.web_research_panel, text="Max external pages to scrape from search:").pack(anchor=tk.W)
        ttk.Spinbox(self.web_research_panel, from_=1, to=20, textvariable=self.research_pages_var, width=10).pack(fill=tk.X, pady=(2, 0))

        self.yt_research_panel = ttk.LabelFrame(parent, text="📺 YouTube Research", padding=10)
        self.yt_research_panel.pack(fill=tk.X)
        self.yt_checkbutton = ttk.Checkbutton(self.yt_research_panel, text="Enable Video Transcript Analysis", variable=self.yt_research_enabled_var)
        self.yt_checkbutton.pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(self.yt_research_panel, text="Videos to analyze per search query:").pack(anchor=tk.W)
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
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_name_var, font=("Segoe UI", 9))
        self.model_entry.pack(fill=tk.X, pady=(2, 8))
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
    
    def run_youtube_video_analysis(self):
        """New method to handle the direct multimodal video analysis."""
        logging.info("🚀 Starting Multimodal YouTube Video Analysis...")
        self.progress_var.set("Validating inputs...")

        # --- 1. Get user inputs from GUI ---
        youtube_url = self.url_var.get().strip()
        api_key_path = self.api_key_file_var.get()
        user_prompt = self.prompt_text.get("1.0", tk.END).strip()
        downloaded_video_filename = "temp_video_for_analysis.mp4"

        if not all([youtube_url, api_key_path, user_prompt]):
            messagebox.showerror("Input Error", "YouTube URL, API Key file, and a prompt are all required for this mode.")
            return

        if not shutil.which('yt-dlp') or not shutil.which('ffmpeg'):
            logging.error("❌ 'yt-dlp' or 'ffmpeg' command not found.")
            messagebox.showerror("Missing Tools", "This feature requires 'yt-dlp' and 'ffmpeg' to be installed and in your system's PATH.")
            return

        # --- 2. Read API Key and Configure Gemini ---
        self.progress_var.set("Configuring API...")
        api_key = APIKeyManager.get_gemini_api_key(api_key_path)
        if not api_key:
            messagebox.showerror("API Key Error", f"Could not read a valid API key from '{api_key_path}'."); return
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logging.error(f"❌ Failed to configure Gemini API: {e}")
            messagebox.showerror("API Error", f"Failed to configure the Gemini API: {e}"); return
            
        video_file = None
        try:
            # --- 3. Download and Convert Video ---
            self.progress_var.set("Downloading video...")
            logging.info(f"Downloading and converting video: {youtube_url}")
            yt_dlp_command = [
                'yt-dlp', '-f', 'bestvideo[height<=480]+bestaudio/best[height<=480]',
                '--recode-video', 'mp4', '-o', downloaded_video_filename,
                '--force-overwrite', youtube_url
            ]
            
            try:
                # Using subprocess.run to capture output for better error logging
                process = subprocess.run(yt_dlp_command, check=True, capture_output=True, text=True)
                logging.info("yt-dlp command finished successfully.")
            except subprocess.CalledProcessError as e:
                logging.error("\n--- yt-dlp Error ---")
                logging.error(f"yt-dlp failed. It might be an issue with the URL, format selection, or ffmpeg.")
                logging.error(f"Stderr: {e.stderr}")
                messagebox.showerror("Download Error", f"yt-dlp failed to download the video.\n\nError: {e.stderr}")
                return

            if not os.path.exists(downloaded_video_filename):
                logging.error(f"--- Critical Error --- Download process finished, but the file '{downloaded_video_filename}' was not created.")
                messagebox.showerror("File Error", "Video download failed. The output file was not created.")
                return

            logging.info(f"✅ Successfully created '{downloaded_video_filename}'.")

            # --- 4. Upload, Analyze ---
            self.progress_var.set("Uploading video to API...")
            logging.info(f"Uploading file '{downloaded_video_filename}' to the Gemini API...")
            video_file = genai.upload_file(path=downloaded_video_filename)
            logging.info(f"File uploaded. Name: {video_file.name}. Waiting for processing...")
            
            self.progress_var.set("Processing video...")
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(name=video_file.name)
                logging.info(f"Current file state: {video_file.state.name}")
                self.progress_var.set(f"Processing... ({video_file.state.name})")

            if video_file.state.name == "FAILED":
                logging.error("Video processing failed on the server.")
                messagebox.showerror("API Error", "The Gemini API failed to process the uploaded video.")
                return

            logging.info("✅ Video processing complete.")
            self.progress_var.set("Generating analysis...")
            logging.info("Sending video and prompt to the model...")
            
            # Use the model name specified in the AI tab, defaulting to gemini-1.5-pro
            model_name = self.model_name_var.get() or "gemini-1.5-pro"
            logging.info(f"Using model: {model_name}")
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content([user_prompt, video_file])

            logging.info("✅ Model response received.")
            self.final_notes_content = response.text
            
            self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
            self.progress_var.set("Generation Complete!"); logging.info("\n🎉 Multimodal Analysis Complete!")
            self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("Success!", "Multimodal video analysis completed successfully!"))

        except Exception as e:
            logging.error(f"❌ An unexpected error occurred during video analysis: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
        finally:
            # --- 5. Clean up resources ---
            logging.info("Cleaning up resources...")
            if video_file:
                try:
                    genai.delete_file(name=video_file.name)
                    logging.info(f"Deleted remote file: {video_file.name}")
                except Exception as e:
                    logging.warning(f"Could not delete remote file {video_file.name}: {e}")

            if os.path.exists(downloaded_video_filename):
                os.remove(downloaded_video_filename)
                logging.info(f"Deleted local file: {downloaded_video_filename}")

            self.is_processing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"))
            if not self.final_notes_content: self.root.after(0, lambda: self.progress_var.set("Ready"))
            else: self.root.after(0, lambda: self.progress_var.set("Complete - Ready to Save"))


    # --- Helper & UI Methods ---
    def update_temperature_label(self, value): self.temp_label.configure(text=f"Temperature: {float(value):.1f}")
    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            if isinstance(widget, (ttk.Frame, ttk.LabelFrame)): self._set_child_widgets_state(widget, state)
            else:
                try: widget.configure(state=state)
                except tk.TclError: pass
    
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
        # Master control for research panel based on checkbox
        research_active = self.research_enabled_var.get()
        panel_state = tk.NORMAL if research_active else tk.DISABLED

        # Set state for all child widgets in the research panels
        self._set_child_widgets_state(self.web_research_panel, panel_state)
        self._set_child_widgets_state(self.yt_research_panel, panel_state)

        # If research is disabled, we don't need to do the dependency checks
        if not research_active: return

        # Re-enable based on dependency availability
        if not PLAYWRIGHT_AVAILABLE: self.playwright_radio.config(state=tk.DISABLED)
        if not YT_DLP_AVAILABLE: self.yt_checkbutton.config(state=tk.DISABLED)

    def load_initial_settings(self):
        self.config = self._load_config_file()
        prompt_template = self._load_prompt_file()
        if self.config:
            logging.info("📝 Loading settings from config.yml...")
            api_settings = self.config.get('api', {}); llm_settings = self.config.get('llm', {}); llm_params = llm_settings.get('parameters', {})
            self.api_key_file_var.set(api_settings.get('key_file', 'gemini_api.key'))
            # Recommend gemini-1.5-pro for video, but allow user override
            self.model_name_var.set(llm_settings.get('model_name', 'gemini-1.5-pro'))
            self.temperature_var.set(llm_params.get('temperature', 0.5))
            self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
            google_search_settings = api_settings.get('google_search', {})
            self.google_api_key_file_var.set(google_search_settings.get('key_file', 'google_api.key'))
            self.google_cx_file_var.set(google_search_settings.get('cx_file', 'google_cx.key'))
            logging.info("✅ Configuration loaded successfully")
        else: logging.warning("⚠️ Could not load config.yml. Using defaults.")

        if prompt_template: self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, prompt_template)
        else: self.reset_prompt_to_default()
        self.update_temperature_label(self.temperature_var.get()); self._check_dependencies()

    def _check_dependencies(self):
        if not DEPENDENCIES_AVAILABLE: messagebox.showwarning("Missing Dependencies", "Core AI/Document libraries missing. App will run in DEMO mode.")
        if not PLAYWRIGHT_AVAILABLE:
            if hasattr(self, 'playwright_radio'): self.playwright_radio.config(state=tk.DISABLED)
            if self.web_research_method_var.get() == 'playwright': self.web_research_method_var.set('google_api')
            logging.warning("⚠️ Playwright not available. Install with: pip install playwright && playwright install")
        if not YT_DLP_AVAILABLE:
            if hasattr(self, 'yt_checkbutton'): self.yt_checkbutton.config(state=tk.DISABLED)
            self.yt_research_enabled_var.set(False)
            logging.warning("⚠️ yt-dlp not available. Install with: pip install yt-dlp")

    def reset_prompt_to_default(self):
        default_prompt = self._load_prompt_file("prompt.md") or "You are an expert educational content creator. Generate a comprehensive study guide in Markdown based on the provided content.\n\nStructure your response with:\n1. **Executive Summary**\n2. **Main Topics** (use ## and ###)\n3. **Key Points & Definitions**\n4. **Practical Examples**\n5. **Further Learning Resources**\n\nContent to analyze:\n{content}\n\nSource: {website_url}"
        self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, default_prompt)

    def save_notes(self):
        if not self.final_notes_content: messagebox.showwarning("No Content", "No study guide to save."); return
        source_name_raw = "video" if self.input_mode_var.get() == "youtube_video" else (urlparse(self.url_var.get()).netloc or "website" if self.input_mode_var.get() == "scrape" else "local_docs")
        source_name = re.sub(r'[^a-zA-Z0-9_-]', '', source_name_raw) # Sanitize filename
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

        # Determine which function to run based on the selected mode
        mode = self.input_mode_var.get()
        if mode == "youtube_video":
            target_function = self.run_youtube_video_analysis
        else:
            target_function = self.run_full_process

        threading.Thread(target=target_function, daemon=True).start()

    def show_demo_content(self):
        demo_content = f"# 🚀 Demo Mode\n\nThis is a demonstration because one or more critical dependencies (like LangChain, PyMuPDF, etc.) are not installed. Please install all required packages to enable full functionality.\n\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.root.after(0, lambda: self.markdown_preview.update_preview(demo_content))
        self.final_notes_content = demo_content
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

# --- Helper & UI Methods (Continued from AdvancedScraperApp) ---
# NOTE: The methods below are part of the AdvancedScraperApp class.
# They are included here for completeness if you are rebuilding the class.

    def run_youtube_video_analysis(self):
        """New method to handle the direct multimodal video analysis."""
        logging.info("🚀 Starting Multimodal YouTube Video Analysis...")
        self.progress_var.set("Validating inputs...")

        # --- 1. Get user inputs from GUI ---
        youtube_url = self.url_var.get().strip()
        api_key_path = self.api_key_file_var.get()
        user_prompt = self.prompt_text.get("1.0", tk.END).strip()
        downloaded_video_filename = "temp_video_for_analysis.mp4"

        if not all([youtube_url, api_key_path, user_prompt]):
            messagebox.showerror("Input Error", "YouTube URL, API Key file, and a prompt are all required for this mode.")
            self.is_processing = False; self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"); return

        if not shutil.which('yt-dlp') or not shutil.which('ffmpeg'):
            logging.error("❌ 'yt-dlp' or 'ffmpeg' command not found.")
            messagebox.showerror("Missing Tools", "This feature requires 'yt-dlp' and 'ffmpeg' to be installed and in your system's PATH.")
            self.is_processing = False; self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"); return

        # --- 2. Read API Key and Configure Gemini ---
        self.progress_var.set("Configuring API...")
        api_key = APIKeyManager.get_gemini_api_key(api_key_path)
        if not api_key:
            messagebox.showerror("API Key Error", f"Could not read a valid API key from '{api_key_path}'.");
            self.is_processing = False; self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"); return
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logging.error(f"❌ Failed to configure Gemini API: {e}")
            messagebox.showerror("API Error", f"Failed to configure the Gemini API: {e}");
            self.is_processing = False; self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"); return

        video_file = None
        try:
            # --- 3. Download and Convert Video ---
            self.progress_var.set("Downloading video...")
            logging.info(f"Downloading and converting video: {youtube_url}")
            yt_dlp_command = [
                'yt-dlp', '-f', 'bestvideo[height<=480]+bestaudio/best[height<=480]',
                '--recode-video', 'mp4', '-o', downloaded_video_filename,
                '--force-overwrite', youtube_url
            ]

            try:
                process = subprocess.run(yt_dlp_command, check=True, capture_output=True, text=True)
                logging.info("yt-dlp command finished successfully.")
            except subprocess.CalledProcessError as e:
                logging.error(f"yt-dlp failed. Stderr: {e.stderr}")
                messagebox.showerror("Download Error", f"yt-dlp failed to download the video.\n\nError: {e.stderr}")
                return

            if not os.path.exists(downloaded_video_filename):
                logging.error(f"Download process finished, but the file '{downloaded_video_filename}' was not created.")
                messagebox.showerror("File Error", "Video download failed. The output file was not created.")
                return

            logging.info(f"✅ Successfully created '{downloaded_video_filename}'.")

            # --- 4. Upload, Analyze ---
            self.progress_var.set("Uploading video to API...")
            logging.info(f"Uploading file '{downloaded_video_filename}' to the Gemini API...")
            video_file = genai.upload_file(path=downloaded_video_filename)
            logging.info(f"File uploaded. Name: {video_file.name}. Waiting for processing...")

            self.progress_var.set("Processing video...")
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(name=video_file.name)
                logging.info(f"Current file state: {video_file.state.name}")
                self.progress_var.set(f"Processing... ({video_file.state.name})")

            if video_file.state.name == "FAILED":
                logging.error("Video processing failed on the server.")
                messagebox.showerror("API Error", "The Gemini API failed to process the uploaded video.")
                return

            logging.info("✅ Video processing complete.")
            self.progress_var.set("Generating analysis...")
            logging.info("Sending video and prompt to the model...")

            model_name = self.model_name_var.get() or "gemini-1.5-pro"
            logging.info(f"Using model: {model_name}")
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content([user_prompt, video_file])

            logging.info("✅ Model response received.")
            self.final_notes_content = response.text

            self.root.after(0, lambda: self.markdown_preview.update_preview(self.final_notes_content))
            self.progress_var.set("Generation Complete!"); logging.info("\n🎉 Multimodal Analysis Complete!")
            self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("Success!", "Multimodal video analysis completed successfully!"))

        except Exception as e:
            logging.error(f"❌ An unexpected error occurred during video analysis: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
        finally:
            # --- 5. Clean up resources ---
            logging.info("Cleaning up resources...")
            if video_file:
                try:
                    genai.delete_file(name=video_file.name)
                    logging.info(f"Deleted remote file: {video_file.name}")
                except Exception as e:
                    logging.warning(f"Could not delete remote file {video_file.name}: {e}")

            if os.path.exists(downloaded_video_filename):
                os.remove(downloaded_video_filename)
                logging.info(f"Deleted local file: {downloaded_video_filename}")

            self.is_processing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL, text="🚀 Start Generation"))
            if not self.final_notes_content: self.root.after(0, lambda: self.progress_var.set("Ready"))
            else: self.root.after(0, lambda: self.progress_var.set("Complete - Ready to Save"))

    def update_temperature_label(self, value): self.temp_label.configure(text=f"Temperature: {float(value):.1f}")

    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            if isinstance(widget, (ttk.Frame, ttk.LabelFrame)): self._set_child_widgets_state(widget, state)
            else:
                try: widget.configure(state=state)
                except tk.TclError: pass

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
        research_active = self.research_enabled_var.get()
        panel_state = tk.NORMAL if research_active else tk.DISABLED
        self._set_child_widgets_state(self.web_research_panel, panel_state)
        self._set_child_widgets_state(self.yt_research_panel, panel_state)
        if not research_active: return
        if not PLAYWRIGHT_AVAILABLE: self.playwright_radio.config(state=tk.DISABLED)
        if not YT_DLP_AVAILABLE: self.yt_checkbutton.config(state=tk.DISABLED)

    def load_initial_settings(self):
        self.config = self._load_config_file()
        prompt_template = self._load_prompt_file()
        if self.config:
            logging.info("📝 Loading settings from config.yml...")
            api_settings = self.config.get('api', {}); llm_settings = self.config.get('llm', {}); llm_params = llm_settings.get('parameters', {})
            self.api_key_file_var.set(api_settings.get('key_file', 'gemini_api.key'))
            self.model_name_var.set(llm_settings.get('model_name', 'gemini-1.5-pro'))
            self.temperature_var.set(llm_params.get('temperature', 0.5))
            self.max_tokens_var.set(llm_params.get('max_output_tokens', 8192))
            google_search_settings = api_settings.get('google_search', {})
            self.google_api_key_file_var.set(google_search_settings.get('key_file', 'google_api.key'))
            self.google_cx_file_var.set(google_search_settings.get('cx_file', 'google_cx.key'))
            logging.info("✅ Configuration loaded successfully")
        else: logging.warning("⚠️ Could not load config.yml. Using defaults.")

        if prompt_template: self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, prompt_template)
        else: self.reset_prompt_to_default()
        self.update_temperature_label(self.temperature_var.get()); self._check_dependencies()

    def _check_dependencies(self):
        if not DEPENDENCIES_AVAILABLE: messagebox.showwarning("Missing Dependencies", "Core AI/Document libraries missing. App will run in DEMO mode.")
        if not PLAYWRIGHT_AVAILABLE:
            if hasattr(self, 'playwright_radio'): self.playwright_radio.config(state=tk.DISABLED)
            if self.web_research_method_var.get() == 'playwright': self.web_research_method_var.set('google_api')
            logging.warning("⚠️ Playwright not available. Install with: pip install playwright && playwright install")
        if not YT_DLP_AVAILABLE:
            if hasattr(self, 'yt_checkbutton'): self.yt_checkbutton.config(state=tk.DISABLED)
            self.yt_research_enabled_var.set(False)
            logging.warning("⚠️ yt-dlp not available. Install with: pip install yt-dlp")

    def reset_prompt_to_default(self):
        default_prompt = self._load_prompt_file("prompt.md") or "You are an expert educational content creator. Generate a comprehensive study guide in Markdown based on the provided content.\n\nStructure your response with:\n1. **Executive Summary**\n2. **Main Topics** (use ## and ###)\n3. **Key Points & Definitions**\n4. **Practical Examples**\n5. **Further Learning Resources**\n\nContent to analyze:\n{content}\n\nSource: {website_url}"
        self.prompt_text.delete("1.0", tk.END); self.prompt_text.insert(tk.END, default_prompt)

    def save_notes(self):
        if not self.final_notes_content: messagebox.showwarning("No Content", "No study guide to save."); return
        source_name_raw = "video" if self.input_mode_var.get() == "youtube_video" else (urlparse(self.url_var.get()).netloc or "website" if self.input_mode_var.get() == "scrape" else "local_docs")
        source_name = re.sub(r'[^a-zA-Z0-9_-]', '', source_name_raw)
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

        mode = self.input_mode_var.get()
        if mode == "youtube_video":
            target_function = self.run_youtube_video_analysis
        else:
            target_function = self.run_full_process

        threading.Thread(target=target_function, daemon=True).start()

    def show_demo_content(self):
        demo_content = f"# 🚀 Demo Mode\n\nThis is a demonstration because one or more critical dependencies are not installed.\n\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.root.after(0, lambda: self.markdown_preview.update_preview(demo_content))
        self.final_notes_content = demo_content
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
    """Main function to initialize and run the Tkinter application."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    root = tk.Tk()
    app = AdvancedScraperApp(root)

    def on_closing():
        if app.is_processing and messagebox.askokcancel("Quit", "A processing task is still running. Are you sure you want to quit?"):
            root.destroy()
        elif not app.is_processing:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
