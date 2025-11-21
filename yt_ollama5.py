import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import subprocess
import os
import sys
from datetime import datetime
import re

# Import LangChain components
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool

# Import YouTube functionality
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pickle

# YouTube API OAuth2 setup
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

class YouTubeSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Search & Play with AI Assistant")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Store video data
        self.current_videos = []
        self.youtube_service = None
        self.llm = None
        self.agent_executor = None
        
        # Initialize components
        self.init_youtube_service()
        self.init_llm_agent()
        
        # Create GUI
        self.create_gui()
        
        # Style configuration
        self.configure_styles()
    
    def init_youtube_service(self):
        """Initialize YouTube service with OAuth2"""
        try:
            self.youtube_service, error = self.authenticate_youtube_oauth2()
            if error:
                messagebox.showerror("Authentication Error", f"Failed to authenticate with YouTube API:\n{error}")
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize YouTube service:\n{str(e)}")
    
    def init_llm_agent(self):
        """Initialize LLM and agent for natural language processing"""
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(model="llama3.2")
            
            # Create YouTube tool
            youtube_tool = Tool(
                name="youtube_tool",
                description="""Universal YouTube tool that handles various YouTube operations based on natural language prompts. 
                
                Supported operations:
                - General search: "search for python tutorials", "find videos about machine learning"
                - Channel search: "recent videos from Lex Fridman channel", "latest videos by TechCrunch"
                - Video info: "get info about https://youtube.com/watch?v=VIDEO_ID"
                - Trending: "trending videos about AI", "popular machine learning videos"
                - Specify results: "show 5 videos about programming", "first 3 videos from channel"
                
                Input should be a natural language prompt describing what you want to find on YouTube.""",
                func=self.youtube_universal_tool
            )
            
            # Get ReAct prompt template
            prompt_template = hub.pull("hwchase17/react")
            
            # Create agent
            agent = create_react_agent(
                llm=self.llm,
                tools=[youtube_tool],
                prompt=prompt_template,
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=[youtube_tool],
                verbose=False,  # Disable verbose for GUI
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate",
                return_intermediate_steps=True
            )
            
        except Exception as e:
            messagebox.showerror("LLM Initialization Error", f"Failed to initialize LLM agent:\n{str(e)}")
    
    def authenticate_youtube_oauth2(self):
        """Authenticate with YouTube Data API using OAuth2 credentials"""
        creds = None
        
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(CREDENTIALS_FILE):
                    return None, f"OAuth2 credentials file '{CREDENTIALS_FILE}' not found."
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    return None, f"Error during OAuth2 flow: {e}"
            
            try:
                with open(TOKEN_FILE, 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                print(f"Warning: Could not save token: {e}")
        
        try:
            youtube_service = build('youtube', 'v3', credentials=creds)
            return youtube_service, None
        except Exception as e:
            return None, f"Error building YouTube service: {e}"
    
    def configure_styles(self):
        """Configure ttk styles for dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for dark theme
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('TButton', background='#404040', foreground='#ffffff')
        style.map('TButton', background=[('active', '#505050')])
        style.configure('TEntry', fieldbackground='#404040', foreground='#ffffff')
        style.configure('Treeview', background='#353535', foreground='#ffffff', fieldbackground='#353535')
        style.configure('Treeview.Heading', background='#404040', foreground='#ffffff')
        style.map('Treeview', background=[('selected', '#0078d4')])
    
    def create_gui(self):
        """Create the main GUI interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="YouTube Search & Play with AI Assistant", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        # Search label and entry
        ttk.Label(search_frame, text="AI Prompt:").grid(row=0, column=0, padx=(0, 10))
        
        self.search_entry = ttk.Entry(search_frame, font=('Arial', 12))
        self.search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.search_entry.bind('<Return>', self.on_search_enter)
        
        # Search button
        self.search_button = ttk.Button(search_frame, text="Ask AI", command=self.process_ai_prompt)
        self.search_button.grid(row=0, column=2)
        
        # AI Response frame
        ai_frame = ttk.LabelFrame(main_frame, text="AI Response", padding="5")
        ai_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 10))
        ai_frame.columnconfigure(0, weight=1)
        
        # AI response text area
        self.ai_response = scrolledtext.ScrolledText(ai_frame, height=4, wrap=tk.WORD, 
                                                    bg='#353535', fg='#ffffff', 
                                                    font=('Arial', 10))
        self.ai_response.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.ai_response.config(state=tk.DISABLED)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Video Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create Treeview for video results
        columns = ('Title', 'Channel', 'Duration', 'Views', 'Published')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=12)
        
        # Configure column headings and widths
        self.tree.heading('Title', text='Title')
        self.tree.heading('Channel', text='Channel')
        self.tree.heading('Duration', text='Duration')
        self.tree.heading('Views', text='Views')
        self.tree.heading('Published', text='Published')
        
        self.tree.column('Title', width=500)
        self.tree.column('Channel', width=180)
        self.tree.column('Duration', width=80)
        self.tree.column('Views', width=100)
        self.tree.column('Published', width=120)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid treeview and scrollbar
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind double-click event
        self.tree.bind('<Double-1>', self.play_selected_video)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0))
        
        # Control buttons
        self.play_button = ttk.Button(control_frame, text="▶ Play Selected", 
                                     command=self.play_selected_video, state='disabled')
        self.play_button.grid(row=0, column=0, padx=(0, 10))
        
        self.copy_url_button = ttk.Button(control_frame, text="📋 Copy URL", 
                                         command=self.copy_selected_url, state='disabled')
        self.copy_url_button.grid(row=0, column=1, padx=(0, 10))
        
        self.clear_button = ttk.Button(control_frame, text="🗑 Clear Results", 
                                      command=self.clear_results)
        self.clear_button.grid(row=0, column=2, padx=(0, 10))
        
        # Example prompts
        examples_frame = ttk.LabelFrame(main_frame, text="Example Prompts", padding="5")
        examples_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        example_prompts = [
            "Show me 5 recent videos from Lex Fridman channel",
            "Find trending Python programming tutorials",
            "Search for latest AI and machine learning videos",
            "Get recent videos about web development"
        ]
        
        for i, prompt in enumerate(example_prompts):
            btn = ttk.Button(examples_frame, text=prompt, 
                           command=lambda p=prompt: self.set_example_prompt(p))
            btn.grid(row=i//2, column=i%2, padx=5, pady=2, sticky=(tk.W))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Enter a natural language prompt to search for YouTube videos")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Focus on search entry
        self.search_entry.focus()
    
    def set_example_prompt(self, prompt):
        """Set example prompt in the search entry"""
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, prompt)
        self.search_entry.focus()
    
    def on_search_enter(self, event):
        """Handle Enter key press in search entry"""
        self.process_ai_prompt()
    
    def process_ai_prompt(self):
        """Process user prompt using AI agent"""
        prompt = self.search_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a search prompt")
            return
        
        if not self.agent_executor:
            messagebox.showerror("AI Error", "AI agent not initialized")
            return
        
        # Disable search button and show loading status
        self.search_button.configure(state='disabled')
        self.status_var.set("AI is processing your request...")
        self.update_ai_response("🤔 AI is thinking...")
        self.root.update()
        
        # Run AI processing in a separate thread
        thread = threading.Thread(target=self._ai_process_thread, args=(prompt,))
        thread.daemon = True
        thread.start()
    
    def _ai_process_thread(self, prompt):
        """AI processing thread to avoid blocking GUI"""
        try:
            # Process with AI agent
            response = self.agent_executor.invoke({"input": prompt})
            
            # Update GUI in main thread
            self.root.after(0, self._handle_ai_response, response, prompt)
            
        except Exception as e:
            self.root.after(0, self._handle_ai_error, f"AI processing failed: {str(e)}")
    
    def _handle_ai_response(self, response, original_prompt):
        """Handle AI response and extract video data"""
        try:
            ai_output = response.get('output', 'No response from AI')
            self.update_ai_response(f"🤖 AI Response:\n{ai_output}")
            
            # Parse the AI response to extract videos if available
            if hasattr(self, 'last_search_results') and self.last_search_results:
                self._update_results(self.last_search_results)
            else:
                self.status_var.set("AI processed request - no video results to display")
            
        except Exception as e:
            self._handle_ai_error(f"Error handling AI response: {str(e)}")
        
        finally:
            self.search_button.configure(state='normal')
    
    def _handle_ai_error(self, error_message):
        """Handle AI processing errors"""
        self.update_ai_response(f"❌ Error: {error_message}")
        self.status_var.set("AI processing failed")
        self.search_button.configure(state='normal')
    
    def update_ai_response(self, text):
        """Update AI response text area"""
        self.ai_response.config(state=tk.NORMAL)
        self.ai_response.delete(1.0, tk.END)
        self.ai_response.insert(1.0, text)
        self.ai_response.config(state=tk.DISABLED)
        self.root.update()
    
    def youtube_universal_tool(self, prompt):
        """Universal YouTube tool that integrates with the GUI"""
        try:
            # Parse the prompt to determine operation
            operation_type, query, max_results, additional_params = self.parse_youtube_request(prompt)
            
            # Perform the search
            if operation_type == "channel_search":
                videos = self.search_channel_videos(query, max_results)
            else:
                videos = self.search_general_videos(query, max_results, additional_params)
            
            # Store results for GUI update
            self.last_search_results = videos
            
            # Return formatted response for AI
            if videos:
                result = f"Found {len(videos)} videos:\n\n"
                for i, video in enumerate(videos, 1):
                    result += f"{i}. {video['title']}\n"
                    result += f"   Channel: {video['channel']}\n"
                    result += f"   Duration: {video['duration']} | Views: {video['views']}\n"
                    result += f"   URL: {video['url']}\n\n"
                result += "[SEARCH COMPLETED SUCCESSFULLY]"
                return result
            else:
                return f"No videos found for query: {query}"
        
        except Exception as e:
            return f"Error searching YouTube: {str(e)}"
    
    def parse_youtube_request(self, prompt):
        """Parse YouTube request from prompt"""
        prompt_lower = prompt.lower()
        
        # Default values
        operation_type = "search"
        query = prompt
        max_results = 5  # Default to 5 for GUI
        additional_params = {}
        
        # Extract max results if specified
        result_patterns = [r'(\d+)\s*(?:results?|videos?)', r'top\s*(\d+)', r'first\s*(\d+)', r'show\s*(\d+)']
        for pattern in result_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                max_results = min(int(match.group(1)), 10)  # Cap at 10 results
                break
        
        # Determine operation type
        if any(keyword in prompt_lower for keyword in ['channel', 'from channel', 'by channel']):
            operation_type = "channel_search"
            # Extract channel name
            channel_patterns = [
                r"from\s+(?:the\s+)?['\"]?([^'\"]+?)['\"]?\s+(?:channel|youtube)",
                r"by\s+['\"]?([^'\"]+?)['\"]?(?:\s+channel)?",
                r"channel\s+['\"]?([^'\"]+?)['\"]?",
                r"['\"]([^'\"]+)['\"]?\s+(?:channel|videos)"
            ]
            
            for pattern in channel_patterns:
                match = re.search(pattern, prompt_lower)
                if match:
                    query = match.group(1).strip()
                    break
        
        elif any(keyword in prompt_lower for keyword in ['recent', 'latest', 'newest']):
            additional_params['order'] = 'date'
        
        elif any(keyword in prompt_lower for keyword in ['trending', 'popular', 'most viewed']):
            additional_params['order'] = 'viewCount'
        
        return operation_type, query, max_results, additional_params
    
    def search_general_videos(self, query, max_results=5, additional_params=None):
        """Search for general videos"""
        if not self.youtube_service:
            return []
        
        search_params = {
            'q': query,
            'part': 'id,snippet',
            'maxResults': max_results,
            'type': 'video'
        }
        
        if additional_params and 'order' in additional_params:
            search_params['order'] = additional_params['order']
        else:
            search_params['order'] = 'relevance'
        
        search_response = self.youtube_service.search().list(**search_params).execute()
        return self.process_search_results(search_response)
    
    def search_channel_videos(self, channel_name, max_results=5):
        """Search for videos from a specific channel"""
        if not self.youtube_service:
            return []
        
        # First, find the channel
        channel_search = self.youtube_service.search().list(
            q=channel_name,
            part='id,snippet',
            maxResults=1,
            type='channel'
        ).execute()
        
        if not channel_search.get('items'):
            # Fallback to general search
            return self.search_general_videos(f"{channel_name} channel", max_results)
        
        channel_id = channel_search['items'][0]['id']['channelId']
        
        # Get videos from the channel
        videos_response = self.youtube_service.search().list(
            channelId=channel_id,
            part='id,snippet',
            maxResults=max_results,
            order='date',
            type='video'
        ).execute()
        
        return self.process_search_results(videos_response)
    
    def process_search_results(self, search_response):
        """Process search results and get additional video details"""
        if not search_response.get('items'):
            return []
        
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        
        # Get detailed video information
        videos_response = self.youtube_service.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(video_ids)
        ).execute()
        
        videos = []
        for video in videos_response['items']:
            try:
                snippet = video['snippet']
                stats = video.get('statistics', {})
                content_details = video.get('contentDetails', {})
                
                # Parse duration
                duration = content_details.get('duration', 'PT0S')
                duration_formatted = self.parse_duration(duration)
                
                # Format view count
                view_count = int(stats.get('viewCount', 0))
                views_formatted = self.format_view_count(view_count)
                
                # Format publish date
                published_date = snippet.get('publishedAt', '')
                published_formatted = self.format_date(published_date)
                
                video_data = {
                    'id': video['id'],
                    'title': snippet['title'],
                    'channel': snippet['channelTitle'],
                    'duration': duration_formatted,
                    'views': views_formatted,
                    'published': published_formatted,
                    'url': f"https://www.youtube.com/watch?v={video['id']}",
                    'description': snippet.get('description', '')[:200] + "..."
                }
                
                videos.append(video_data)
                
            except Exception as e:
                print(f"Error processing video: {e}")
                continue
        
        return videos
    
    def parse_duration(self, duration_str):
        """Parse ISO 8601 duration to readable format"""
        try:
            duration_str = duration_str.replace('PT', '')
            
            hours = 0
            minutes = 0
            seconds = 0
            
            if 'H' in duration_str:
                hours = int(duration_str.split('H')[0])
                duration_str = duration_str.split('H')[1]
            
            if 'M' in duration_str:
                minutes = int(duration_str.split('M')[0])
                duration_str = duration_str.split('M')[1]
            
            if 'S' in duration_str:
                seconds = int(duration_str.split('S')[0])
            
            if hours > 0:
                return f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                return f"{minutes}:{seconds:02d}"
                
        except:
            return "Unknown"
    
    def format_view_count(self, count):
        """Format view count to readable format"""
        if count >= 1000000:
            return f"{count/1000000:.1f}M"
        elif count >= 1000:
            return f"{count/1000:.1f}K"
        else:
            return str(count)
    
    def format_date(self, date_str):
        """Format date to readable format"""
        try:
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime('%Y-%m-%d')
        except:
            return "Unknown"
    
    def _update_results(self, videos):
        """Update the results display"""
        # Clear existing results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Store videos data
        self.current_videos = videos
        
        # Populate treeview
        for video in videos:
            self.tree.insert('', tk.END, values=(
                video['title'][:70] + "..." if len(video['title']) > 70 else video['title'],
                video['channel'],
                video['duration'],
                video['views'],
                video['published']
            ))
        
        # Enable buttons
        self.play_button.configure(state='normal' if videos else 'disabled')
        self.copy_url_button.configure(state='normal' if videos else 'disabled')
        
        # Update status
        self.status_var.set(f"Found {len(videos)} videos")
    
    def play_selected_video(self, event=None):
        """Play the selected video using mpv"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to play")
            return
        
        # Get selected video index
        item = selection[0]
        video_index = self.tree.index(item)
        
        if video_index >= len(self.current_videos):
            messagebox.showerror("Error", "Invalid video selection")
            return
        
        video = self.current_videos[video_index]
        self.play_video_with_mpv(video['url'], video['title'])
    
    def play_video_with_mpv(self, url, title):
        """Play video using mpv"""
        try:
            self.status_var.set(f"Playing: {title[:50]}...")
            
            # Try to start mpv
            if os.name == 'nt':  # Windows
                subprocess.Popen(['mpv', url], creationflags=subprocess.CREATE_NO_WINDOW)
            else:  # Linux/Mac
                subprocess.Popen(['mpv', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            messagebox.showinfo("Playing", f"Started playing:\n{title}")
            
        except FileNotFoundError:
            messagebox.showerror("MPV Not Found", 
                               "MPV player not found. Please install MPV:\n\n"
                               "Windows: Download from https://mpv.io/\n"
                               "Ubuntu: sudo apt install mpv\n"
                               "macOS: brew install mpv")
        except Exception as e:
            messagebox.showerror("Playback Error", f"Failed to play video:\n{str(e)}")
    
    def copy_selected_url(self):
        """Copy selected video URL to clipboard"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to copy URL")
            return
        
        item = selection[0]
        video_index = self.tree.index(item)
        
        if video_index >= len(self.current_videos):
            messagebox.showerror("Error", "Invalid video selection")
            return
        
        video = self.current_videos[video_index]
        
        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(video['url'])
        self.root.update()
        
        messagebox.showinfo("Copied", f"URL copied to clipboard:\n{video['url']}")
    
    def clear_results(self):
        """Clear all results"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.current_videos = []
        self.play_button.configure(state='disabled')
        self.copy_url_button.configure(state='disabled')
        self.update_ai_response("")
        self.status_var.set("Results cleared")

def main():
    """Main function to run the application"""
    try:
        root = tk.Tk()
        app = YouTubeSearchApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")

if __name__ == "__main__":
    main()
