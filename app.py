# app.py

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import subprocess
import webbrowser
import sys

import config
from youtube import YouTubeService
from agent import AssistantAgent

class YouTubeSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title(config.APP_TITLE)
        self.root.geometry(config.APP_GEOMETRY)
        self.root.configure(bg=config.THEME["bg"])

        self.current_videos = []
        self.youtube_service = None
        self.agent = None

        # --- FIX: Reordered method calls ---
        # 1. Create the GUI first, which defines self.status_var
        self._create_gui()
        
        # 2. Configure styles
        self._configure_styles()
        
        # 3. Now, initialize the backend, which can safely access self.status_var
        self._initialize_backend()
        
        self.search_entry.focus()

    def _initialize_backend(self):
        """Initialize YouTube service and AI agent in a thread to keep UI responsive."""
        self.search_button.config(state='disabled')
        
        def init():
            try:
                self.status_var.set("Initializing YouTube service...")
                self.youtube_service = YouTubeService()
                
                self.status_var.set("Initializing AI Assistant (this may take a moment)...")
                self.agent = AssistantAgent(self.youtube_service)
                
                self.status_var.set("Ready. Ask the AI to find YouTube videos.")
                self.search_button.config(state='normal')
            except Exception as e:
                messagebox.showerror("Initialization Error", str(e))
                self.root.quit()

        threading.Thread(target=init, daemon=True).start()

    def _configure_styles(self):
        """Configure ttk styles for the application's theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # General widget styles
        style.configure('TFrame', background=config.THEME["bg"])
        style.configure('TLabel', background=config.THEME["bg"], foreground=config.THEME["fg"])
        style.configure('TButton', background=config.THEME["bg_dark"], foreground=config.THEME["fg"])
        style.map('TButton', background=[('active', config.THEME["bg_light"])])
        style.configure('TEntry', fieldbackground=config.THEME["bg_dark"], foreground=config.THEME["fg"])
        style.configure('TLabelframe', background=config.THEME["bg"], foreground=config.THEME["fg"])
        style.configure('TLabelframe.Label', background=config.THEME["bg"], foreground=config.THEME["fg_secondary"])
        
        # Treeview styles
        style.configure('Treeview', background=config.THEME["bg_light"], foreground=config.THEME["fg"], fieldbackground=config.THEME["bg_light"])
        style.configure('Treeview.Heading', background=config.THEME["bg_dark"], foreground=config.THEME["fg"])
        style.map('Treeview', background=[('selected', config.THEME["accent"])])

    def _create_gui(self):
        """Create the main graphical user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1) # Results frame will expand

        # --- Search Bar ---
        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="Ask AI:", font=('Arial', 12)).grid(row=0, column=0, padx=(0, 10))
        self.search_entry = ttk.Entry(search_frame, font=('Arial', 12))
        self.search_entry.grid(row=0, column=1, sticky="ew")
        self.search_entry.bind('<Return>', lambda e: self.process_ai_prompt())
        self.search_button = ttk.Button(search_frame, text="Search", command=self.process_ai_prompt)
        self.search_button.grid(row=0, column=2, padx=(10, 0))

        # --- AI Response Area ---
        ai_frame = ttk.LabelFrame(main_frame, text="AI Thought Process", padding="5")
        ai_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ai_frame.columnconfigure(0, weight=1)
        self.ai_response_text = scrolledtext.ScrolledText(ai_frame, height=6, wrap=tk.WORD, bg=config.THEME["bg_light"], fg=config.THEME["fg_secondary"])
        self.ai_response_text.grid(row=0, column=0, sticky="ew")
        self.ai_response_text.config(state=tk.DISABLED)

        # --- Results Treeview ---
        results_frame = ttk.LabelFrame(main_frame, text="Video Results", padding="5")
        results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        columns = ('Title', 'Channel', 'Duration', 'Views', 'Published')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        for col in columns: self.tree.heading(col, text=col)
        self.tree.column('Title', width=500)
        self.tree.column('Channel', width=180)
        self.tree.column('Duration', width=80, anchor='center')
        self.tree.column('Views', width=100, anchor='center')
        self.tree.column('Published', width=120, anchor='center')

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.bind('<Double-1>', self.play_selected_video)

        # --- Control Buttons ---
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.play_button = ttk.Button(control_frame, text="Play Selected", command=self.play_selected_video, state='disabled')
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        self.copy_url_button = ttk.Button(control_frame, text="Copy URL", command=self.copy_selected_url, state='disabled')
        self.copy_url_button.pack(side=tk.LEFT)
        
        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Initializing GUI...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.grid(row=1, column=0, sticky="ew")

    def process_ai_prompt(self):
        prompt = self.search_entry.get().strip()
        if not prompt or not self.agent: return
        
        self.search_button.config(state='disabled')
        self.clear_results()
        self.update_ai_response("🤔 AI is thinking...", clear=True)
        self.status_var.set("Processing with AI agent...")
        
        threading.Thread(target=self._ai_process_thread, args=(prompt,), daemon=True).start()

    def _ai_process_thread(self, prompt):
        try:
            response = self.agent.invoke(prompt)
            self.root.after(0, self._handle_ai_response, response)
        except Exception as e:
            self.root.after(0, self._handle_ai_error, str(e))

    def _handle_ai_response(self, response):
        thought_process = ""
        if steps := response.get('intermediate_steps'):
            for action, observation in steps:
                thought_process += f"Action: {action.tool}\nInput: {action.tool_input}\nObservation: {observation}\n\n"
        self.update_ai_response(thought_process, clear=True)
        
        ai_output = response.get('output', 'AI gave no final answer.')
        self.update_ai_response(f"🤖 **Final Answer:**\n{ai_output}")
        
        self._update_results_display(self.agent.last_search_results)
        self.search_button.config(state='normal')

    def _handle_ai_error(self, error_message):
        self.update_ai_response(f"❌ Error: {error_message}")
        self.status_var.set("An error occurred during AI processing.")
        self.search_button.config(state='normal')

    def update_ai_response(self, text, clear=False):
        self.ai_response_text.config(state=tk.NORMAL)
        if clear: self.ai_response_text.delete(1.0, tk.END)
        self.ai_response_text.insert(tk.END, text + "\n")
        self.ai_response_text.config(state=tk.DISABLED)
        self.ai_response_text.see(tk.END) # Auto-scroll

    def _update_results_display(self, videos):
        self.clear_results(clear_ui_only=True)
        self.current_videos = videos
        
        for video in videos:
            self.tree.insert('', tk.END, values=(
                video['title'], video['channel'], video['duration'], video['views'], video['published']
            ))
        
        has_results = bool(videos)
        self.play_button.config(state='normal' if has_results else 'disabled')
        self.copy_url_button.config(state='normal' if has_results else 'disabled')
        self.status_var.set(f"Found {len(videos)} videos.")

    def play_selected_video(self, event=None):
        if not (selection := self.tree.selection()): return
        video_index = self.tree.index(selection[0])
        video = self.current_videos[video_index]

        self.status_var.set(f"Attempting to play: {video['title'][:50]}...")
        try:
            # Try to use mpv player
            subprocess.Popen(['mpv', video['url']], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, OSError):
            # Fallback to web browser
            webbrowser.open(video['url'])
            self.status_var.set(f"Opened in browser: {video['title'][:50]}...")

    def copy_selected_url(self):
        if not (selection := self.tree.selection()): return
        video_index = self.tree.index(selection[0])
        video_url = self.current_videos[video_index]['url']
        
        self.root.clipboard_clear()
        self.root.clipboard_append(video_url)
        self.status_var.set(f"Copied URL to clipboard: {video_url}")

    def clear_results(self, clear_ui_only=False):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if not clear_ui_only:
            self.current_videos = []
            self.update_ai_response("", clear=True)
            self.status_var.set("Results cleared.")
        self.play_button.config(state='disabled')
        self.copy_url_button.config(state='disabled')

def main():
    root = tk.Tk()
    app = YouTubeSearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
