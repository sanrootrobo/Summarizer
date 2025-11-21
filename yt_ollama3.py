import os
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
import yt_dlp
import json
from datetime import datetime

def search_youtube_ytdlp(query, max_results=3):
    """
    Search YouTube videos using yt-dlp
    Returns a formatted string with video titles, URLs, descriptions, and metadata
    """
    try:
        # Configure yt-dlp options for search
        ydl_opts = {
            'quiet': True,  # Suppress output
            'no_warnings': True,
            'extract_flat': True,  # Don't download, just extract metadata
            'default_search': 'ytsearch',  # Use YouTube search
            'ignoreerrors': True,
        }
        
        # Construct search query - limit results
        search_query = f"ytsearch{max_results}:{query}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info without downloading
            search_results = ydl.extract_info(search_query, download=False)
            
            if not search_results or 'entries' not in search_results:
                return f"No videos found for query: {query}"
            
            results = []
            for entry in search_results['entries']:
                if entry is None:  # Skip None entries
                    continue
                    
                title = entry.get('title', 'No title')
                url = entry.get('url', entry.get('webpage_url', 'No URL'))
                uploader = entry.get('uploader', entry.get('channel', 'Unknown'))
                duration = entry.get('duration', 0)
                view_count = entry.get('view_count', 0)
                upload_date = entry.get('upload_date', '')
                
                # Format duration
                if duration and isinstance(duration, (int, float)):
                    duration = int(duration)  # Convert to int to avoid float formatting issues
                    duration_str = f"{duration//60}:{duration%60:02d}" if duration < 3600 else f"{duration//3600}:{(duration%3600)//60:02d}:{duration%60:02d}"
                else:
                    duration_str = "Unknown"
                
                # Format upload date
                if upload_date:
                    try:
                        date_obj = datetime.strptime(upload_date, '%Y%m%d')
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                    except:
                        formatted_date = upload_date
                else:
                    formatted_date = "Unknown"
                
                # Format view count
                if view_count and isinstance(view_count, (int, float)):
                    view_count = int(view_count)  # Ensure it's an integer
                    if view_count >= 1000000:
                        views_str = f"{view_count/1000000:.1f}M views"
                    elif view_count >= 1000:
                        views_str = f"{view_count/1000:.1f}K views"
                    else:
                        views_str = f"{view_count} views"
                else:
                    views_str = "Unknown views"
                
                result_text = f"""
Title: {title}
Channel: {uploader}
Duration: {duration_str}
Views: {views_str}
Upload Date: {formatted_date}
URL: {url}
{'='*60}"""
                results.append(result_text)
            
            if not results:
                return f"No valid video results found for query: {query}"
            
            return f"YouTube Search Results for '{query}':\n" + "\n".join(results)
    
    except Exception as e:
        return f"Error searching YouTube with yt-dlp: {str(e)}"

def search_channel_ytdlp(channel_name, max_results=3):
    """
    Search for recent videos from a specific YouTube channel using yt-dlp
    """
    try:
        # Try different channel search formats
        search_queries = [
            f"ytsearch{max_results}:{channel_name} channel",
            f"ytsearch{max_results}:@{channel_name.replace(' ', '')}",
            f"ytsearch{max_results}:{channel_name} recent videos"
        ]
        
        for search_query in search_queries:
            try:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'ignoreerrors': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    search_results = ydl.extract_info(search_query, download=False)
                    
                    if search_results and 'entries' in search_results:
                        # Filter results to try to get videos from the specific channel
                        channel_videos = []
                        for entry in search_results['entries']:
                            if entry is None:
                                continue
                            
                            uploader = entry.get('uploader', entry.get('channel', '')).lower()
                            if channel_name.lower() in uploader:
                                channel_videos.append(entry)
                        
                        # If we found channel-specific videos, use those; otherwise use all results
                        videos_to_show = channel_videos if channel_videos else search_results['entries'][:max_results]
                        
                        if videos_to_show:
                            results = []
                            for entry in videos_to_show[:max_results]:
                                if entry is None:
                                    continue
                                
                                title = entry.get('title', 'No title')
                                url = entry.get('url', entry.get('webpage_url', 'No URL'))
                                uploader = entry.get('uploader', entry.get('channel', 'Unknown'))
                                duration = entry.get('duration', 0)
                                view_count = entry.get('view_count', 0)
                                upload_date = entry.get('upload_date', '')
                                
                                # Format duration
                                if duration and isinstance(duration, (int, float)):
                                    duration = int(duration)  # Convert to int
                                    duration_str = f"{duration//60}:{duration%60:02d}" if duration < 3600 else f"{duration//3600}:{(duration%3600)//60:02d}:{duration%60:02d}"
                                else:
                                    duration_str = "Unknown"
                                
                                # Format upload date
                                if upload_date:
                                    try:
                                        date_obj = datetime.strptime(upload_date, '%Y%m%d')
                                        formatted_date = date_obj.strftime('%Y-%m-%d')
                                    except:
                                        formatted_date = upload_date
                                else:
                                    formatted_date = "Unknown"
                                
                                # Format view count
                                if view_count and isinstance(view_count, (int, float)):
                                    view_count = int(view_count)  # Ensure it's an integer
                                    if view_count >= 1000000:
                                        views_str = f"{view_count/1000000:.1f}M views"
                                    elif view_count >= 1000:
                                        views_str = f"{view_count/1000:.1f}K views"
                                    else:
                                        views_str = f"{view_count} views"
                                else:
                                    views_str = "Unknown views"
                                
                                result_text = f"""
Title: {title}
Channel: {uploader}
Duration: {duration_str}
Views: {views_str}
Upload Date: {formatted_date}
URL: {url}
{'='*60}"""
                                results.append(result_text)
                            
                            return f"Recent videos from '{channel_name}':\n" + "\n".join(results)
            
            except Exception as e:
                continue  # Try next search query format
        
        return f"Could not find recent videos for channel: {channel_name}"
    
    except Exception as e:
        return f"Error searching channel with yt-dlp: {str(e)}"

def get_video_info_ytdlp(url):
    """
    Get detailed information about a specific YouTube video
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if not info:
                return f"Could not extract information from URL: {url}"
            
            title = info.get('title', 'No title')
            uploader = info.get('uploader', info.get('channel', 'Unknown'))
            description = info.get('description', 'No description')
            duration = info.get('duration', 0)
            view_count = info.get('view_count', 0)
            like_count = info.get('like_count', 0)
            upload_date = info.get('upload_date', '')
            
            # Truncate long descriptions
            if len(description) > 300:
                description = description[:300] + "..."
            
            # Format duration
            if duration and isinstance(duration, (int, float)):
                duration = int(duration)  # Convert to int
                duration_str = f"{duration//60}:{duration%60:02d}" if duration < 3600 else f"{duration//3600}:{(duration%3600)//60:02d}:{duration%60:02d}"
            else:
                duration_str = "Unknown"
            
            # Format upload date
            if upload_date:
                try:
                    date_obj = datetime.strptime(upload_date, '%Y%m%d')
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                except:
                    formatted_date = upload_date
            else:
                formatted_date = "Unknown"
            
            # Format counts
            views_str = f"{int(view_count):,}" if view_count and isinstance(view_count, (int, float)) else "Unknown"
            likes_str = f"{int(like_count):,}" if like_count and isinstance(like_count, (int, float)) else "Unknown"
            
            result = f"""
Video Information:
Title: {title}
Channel: {uploader}
Duration: {duration_str}
Views: {views_str}
Likes: {likes_str}
Upload Date: {formatted_date}
URL: {url}

Description: {description}
"""
            return result
    
    except Exception as e:
        return f"Error getting video info: {str(e)}"

# --- 1. Initialize the LLM ---
print("Initializing Ollama LLM with model 'llama3.2'...")
llm = Ollama(model="llama3.2")

# --- 2. Define the Tools ---
print("Setting up YouTube search tools with yt-dlp...")

# General YouTube search tool
youtube_search_tool = Tool(
    name="youtube_search",
    description="Search for YouTube videos by query. Input should be a search query string. Returns video titles, URLs, channels, views, duration, and upload dates.",
    func=lambda query: search_youtube_ytdlp(query, max_results=3)
)

# Channel-specific search tool
channel_search_tool = Tool(
    name="youtube_channel_search", 
    description="Search for recent videos from a specific YouTube channel. Input should be the channel name (e.g., 'Lex Fridman'). Returns recent videos from that channel with metadata.",
    func=lambda channel: search_channel_ytdlp(channel, max_results=3)
)

# Video info tool
video_info_tool = Tool(
    name="youtube_video_info",
    description="Get detailed information about a specific YouTube video. Input should be a YouTube URL. Returns title, channel, description, views, likes, etc.",
    func=get_video_info_ytdlp
)

tools = [youtube_search_tool, channel_search_tool, video_info_tool]

# --- 3. Get the Prompt Template ---
print("Pulling the ReAct prompt template from LangChain Hub...")
prompt_template = hub.pull("hwchase17/react")

# --- 4. Create the Agent ---
print("Creating the ReAct agent...")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,
)

# --- 5. Create the Agent Executor ---
print("Creating the Agent Executor...")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# --- 6. Test yt-dlp functionality ---
print("\n--- Testing yt-dlp functionality ---")
try:
    test_result = search_youtube_ytdlp("Python tutorial", max_results=1)
    if "Error" not in test_result:
        print("✓ yt-dlp is working correctly!")
    else:
        print(f"⚠ yt-dlp test result: {test_result}")
except Exception as e:
    print(f"✗ Error testing yt-dlp: {e}")
    print("Make sure yt-dlp is installed: pip install yt-dlp")

# --- 7. Run the Agent Executor with a Prompt ---
print("\nInvoking the agent with the YouTube search prompt...")
prompt_text = "What are the most recent videos from the 'Lex Fridman' YouTube channel?"

try:
    response = agent_executor.invoke({
        "input": prompt_text
    })
    print("\n--- Agent's Final Output ---")
    print(response['output'])
except Exception as e:
    print(f"\nAn error occurred while running the agent: {e}")

# Optional: Test the tools directly
print("\n--- Testing the YouTube tools directly ---")
try:
    print("Testing channel search for Lex Fridman:")
    direct_result = channel_search_tool.func("Lex Fridman")
    print(direct_result[:800] + "..." if len(direct_result) > 800 else direct_result)
except Exception as e:
    print(f"Error testing channel tool: {e}")
