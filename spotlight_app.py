# spotlight_app.py

import tkinter as tk
from tkinter import ttk
import threading
import subprocess
import webbrowser
import sys

import config
from youtube import YouTubeService
from agent import AssistantAgent

class SpotlightApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw() # Hide the main window initially

        # --- Window Style ---
        self.root.configure(bg=config.THEME["bg"])
        self.root.overrideredirect(True) # Create a borderless, chromeless window

        # --- Window Centering ---
        self.center_window()

        # --- Widgets ---
        self._create_widgets()
        self._configure_styles()
        
        # --- Bindings ---
        self.root.bind("<Escape>", self.hide_window)
        self.root.bind("<FocusOut>", self.hide_window) # Hide when user clicks away
        self.search_entry.bind("<Return>", self.start_search)

        # --- Backend Initialization ---
        self.youtube_service = None
        self.agent = None
        self._initialize_backend()

        # --- Show Window and Run ---
        self.root.deiconify() # Show the configured window
        self.search_entry.focus_set() # Immediately focus the search bar
        self.root.mainloop()

    def center_window(self):
        """Calculates screen dimensions and centers the window."""
        window_width = 800
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_coord = (screen_width / 2) - (window_width / 2)
        # Position it in the top third of the screen, like Spotlight
        y_coord = (screen_height / 3) - (window_height / 2)
        self.root.geometry(f"{window_width}x{window_height}+{int(x_coord)}+{int(y_coord)}")

    def _configure_styles(self):
        """Configure ttk styles for the spotlight theme."""
        style = ttk.Style()
        style.configure('Spotlight.TFrame', background=config.THEME["bg_light"])
        style.configure('Spotlight.TLabel', background=config.THEME["bg_light"], foreground=config.THEME["fg"])
        style.configure('Result.TButton', background=config.THEME["bg_light"], foreground=config.THEME["fg_secondary"], borderwidth=0)
        style.map('Result.TButton', 
                  background=[('active', config.THEME["bg_dark"])],
                  foreground=[('active', config.THEME["fg"])])

    def _create_widgets(self):
        """Create the widgets for the spotlight interface."""
        # A main frame with a slightly different background
        self.main_frame = ttk.Frame(self.root, style='Spotlight.TFrame', padding=2)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Search Entry
        self.search_entry = tk.Entry(
            self.main_frame,
            font=('Arial', 24),
            bg=config.THEME["bg_light"],
            fg=config.THEME["fg"],
            insertbackground=config.THEME["accent"], # Cursor color
            bd=0, # No border
            highlightthickness=0 # No focus border
        )
        self.search_entry.pack(fill=tk.X, padx=15, pady=10)
        
        # Separator
        separator = ttk.Separator(self.main_frame, orient='horizontal')
        separator.pack(fill=tk.X, padx=10, pady=5)

        # Results Frame
        self.results_frame = ttk.Frame(self.main_frame, style='Spotlight.TFrame')
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status Label (initially hidden)
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(self.results_frame, textvariable=self.status_var, style='Spotlight.TLabel', font=('Arial', 12))

    def _initialize_backend(self):
        """Initialize backend services in a separate thread."""
        def init():
            try:
                self.youtube_service = YouTubeService()
                self.agent = AssistantAgent(self.youtube_service)
                self.status_var.set("Ask the AI to find YouTube videos...")
                self.status_label.pack(pady=20) # Show the label now
            except Exception as e:
                self.status_var.set(f"Error: {e}")
                self.status_label.pack(pady=20)
        
        threading.Thread(target=init, daemon=True).start()

    def start_search(self, event=None):
        """Initiates the search process in a thread."""
        prompt = self.search_entry.get().strip()
        if not prompt or not self.agent:
            return

        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        self.status_var.set(f"🤔 Searching for: {prompt}")
        self.status_label = ttk.Label(self.results_frame, textvariable=self.status_var, style='Spotlight.TLabel', font=('Arial', 12))
        self.status_label.pack(pady=20)

        threading.Thread(target=self._search_thread, args=(prompt,), daemon=True).start()

    def _search_thread(self, prompt):
        """The actual AI search logic that runs in a background thread."""
        try:
            response = self.agent.invoke(prompt)
            videos = self.agent.last_search_results
            self.root.after(0, self.display_results, videos)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def display_results(self, videos):
        """Updates the GUI with the video results."""
        # Clear "Searching..." label
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not videos:
            self.status_var.set("No videos found.")
            self.status_label = ttk.Label(self.results_frame, textvariable=self.status_var, style='Spotlight.TLabel')
            self.status_label.pack(pady=20)
            return

        for video in videos:
            # Use a lambda to capture the correct url for each button
            btn = ttk.Button(
                self.results_frame,
                text=f"▶  {video['title']}  ({video['channel']})",
                style='Result.TButton',
                command=lambda v=video: self.play_video(v)
            )
            btn.pack(fill=tk.X, pady=2, ipady=5)
            
    def display_error(self, error_message):
        """Displays an error message in the results frame."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.status_var.set(f"❌ Error: {error_message}")
        self.status_label = ttk.Label(self.results_frame, textvariable=self.status_var, style='Spotlight.TLabel')
        self.status_label.pack(pady=20)

    def play_video(self, video):
        """Plays the selected video and hides the window."""
        url = video['url']
        try:
            subprocess.Popen(['mpv', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, OSError):
            webbrowser.open(url)
        self.hide_window()

    def hide_window(self, event=None):
        """Hides and closes the spotlight window."""
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    SpotlightApp()
