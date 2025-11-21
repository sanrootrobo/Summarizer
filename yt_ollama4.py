import os
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pickle
import json
import re

# YouTube API OAuth2 setup
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
CREDENTIALS_FILE = 'credentials.json'  # Your OAuth2 credentials file
TOKEN_FILE = 'token.pickle'  # File to store the access token

def authenticate_youtube_oauth2():
    """
    Authenticate with YouTube Data API using OAuth2 credentials
    Returns authenticated YouTube service object
    """
    creds = None
    
    # Load existing token if available
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no valid credentials, get new ones
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
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                return None, f"Error during OAuth2 flow: {e}"
        
        # Save the credentials for the next run
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

def parse_youtube_request(prompt):
    """
    Parse the user prompt to determine the type of YouTube operation and extract parameters
    Returns: (operation_type, query, max_results, additional_params)
    """
    prompt_lower = prompt.lower()
    
    # Default values
    operation_type = "search"
    query = prompt
    max_results = 3
    additional_params = {}
    
    # Extract max results if specified
    result_patterns = [r'(\d+)\s*(?:results?|videos?)', r'top\s*(\d+)', r'first\s*(\d+)', r'show\s*(\d+)']
    for pattern in result_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            max_results = min(int(match.group(1)), 10)  # Cap at 10 results
            break
    
    # Determine operation type based on keywords
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
    
    elif any(keyword in prompt_lower for keyword in ['video info', 'video details', 'about video', 'watch?v=']):
        operation_type = "video_info"
        # Extract YouTube URL if present
        url_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)'
        match = re.search(url_pattern, prompt)
        if match:
            query = f"https://www.youtube.com/watch?v={match.group(1)}"
    
    elif any(keyword in prompt_lower for keyword in ['trending', 'popular', 'most viewed']):
        operation_type = "trending"
        additional_params['order'] = 'viewCount'
    
    elif any(keyword in prompt_lower for keyword in ['recent', 'latest', 'newest']):
        operation_type = "search"
        additional_params['order'] = 'date'
    
    else:
        # Default search - clean up the query by removing operation indicators
        clean_patterns = [
            r'search\s+(?:for\s+)?',
            r'find\s+(?:me\s+)?',
            r'show\s+(?:me\s+)?',
            r'get\s+(?:me\s+)?',
            r'youtube\s+',
            r'videos?\s+(?:about\s+)?'
        ]
        
        for pattern in clean_patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
    
    return operation_type, query, max_results, additional_params

def youtube_universal_tool(prompt):
    """
    Universal YouTube tool that handles different types of YouTube operations based on the prompt
    """
    youtube_service, error = authenticate_youtube_oauth2()
    
    if error:
        return f"Authentication Error: {error}"
    
    try:
        # Parse the prompt to determine operation
        operation_type, query, max_results, additional_params = parse_youtube_request(prompt)
        
        result = ""
        if operation_type == "channel_search":
            result = search_channel_videos(youtube_service, query, max_results)
        elif operation_type == "video_info":
            result = get_video_info(youtube_service, query)
        elif operation_type == "trending":
            result = search_trending_videos(youtube_service, query, max_results)
        else:  # Default to search
            result = search_youtube_videos(youtube_service, query, max_results, additional_params)
        
        # Add a clear completion indicator to help the agent recognize task completion
        if result and not result.startswith("Error") and not result.startswith("Authentication Error"):
            result += "\n\n[SEARCH COMPLETED SUCCESSFULLY]"
        
        return result
    
    except Exception as e:
        return f"Error processing YouTube request: {str(e)}"

def search_youtube_videos(youtube_service, query, max_results=3, additional_params=None):
    """Search for YouTube videos with optional parameters"""
    try:
        search_params = {
            'q': query,
            'part': 'id,snippet',
            'maxResults': max_results,
            'type': 'video'
        }
        
        # Add additional parameters
        if additional_params:
            if 'order' in additional_params:
                search_params['order'] = additional_params['order']
        
        search_response = youtube_service.search().list(**search_params).execute()
        
        if not search_response.get('items'):
            return f"No videos found for query: {query}"
        
        results = []
        for item in search_response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            if len(description) > 200:
                description = description[:200] + "..."
            
            channel = item['snippet']['channelTitle']
            published = item['snippet']['publishedAt']
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            result_text = f"""
Title: {title}
Channel: {channel}
Published: {published}
URL: {url}
Description: {description}
{'='*50}"""
            results.append(result_text)
        
        return f"YouTube Search Results for '{query}':\n" + "\n".join(results)
        
    except Exception as e:
        return f"Error searching YouTube videos: {str(e)}"

def search_channel_videos(youtube_service, channel_name, max_results=3):
    """Search for recent videos from a specific channel"""
    try:
        # First, search for the channel
        channel_search = youtube_service.search().list(
            q=channel_name,
            part='id,snippet',
            maxResults=1,
            type='channel'
        ).execute()
        
        if not channel_search.get('items'):
            return f"Channel '{channel_name}' not found. Searching for videos with channel name in query..."
            # Fallback to general search
            return search_youtube_videos(youtube_service, f"{channel_name} channel", max_results)
        
        channel_id = channel_search['items'][0]['id']['channelId']
        
        # Get recent videos from the channel
        videos_response = youtube_service.search().list(
            channelId=channel_id,
            part='id,snippet',
            maxResults=max_results,
            order='date',
            type='video'
        ).execute()
        
        if not videos_response.get('items'):
            return f"No recent videos found for channel: {channel_name}"
        
        results = []
        for item in videos_response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            if len(description) > 200:
                description = description[:200] + "..."
            
            published = item['snippet']['publishedAt']
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            result_text = f"""
Title: {title}
Channel: {channel_name}
Published: {published}
URL: {url}
Description: {description}
{'='*50}"""
            results.append(result_text)
        
        return f"Recent videos from {channel_name}:\n" + "\n".join(results)
        
    except Exception as e:
        return f"Error searching channel videos: {str(e)}"

def get_video_info(youtube_service, video_url):
    """Get detailed information about a specific video"""
    try:
        # Extract video ID from URL
        video_id_pattern = r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)'
        match = re.search(video_id_pattern, video_url)
        if not match:
            return f"Invalid YouTube URL: {video_url}"
        
        video_id = match.group(1)
        
        # Get video details
        video_response = youtube_service.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        ).execute()
        
        if not video_response.get('items'):
            return f"Video not found: {video_url}"
        
        video = video_response['items'][0]
        snippet = video['snippet']
        stats = video.get('statistics', {})
        
        title = snippet['title']
        channel = snippet['channelTitle']
        description = snippet['description']
        if len(description) > 300:
            description = description[:300] + "..."
        
        published = snippet['publishedAt']
        view_count = stats.get('viewCount', 'N/A')
        like_count = stats.get('likeCount', 'N/A')
        comment_count = stats.get('commentCount', 'N/A')
        
        result = f"""
Video Information:
Title: {title}
Channel: {channel}
Published: {published}
Views: {view_count}
Likes: {like_count}
Comments: {comment_count}
URL: {video_url}

Description: {description}
"""
        return result
        
    except Exception as e:
        return f"Error getting video info: {str(e)}"

def search_trending_videos(youtube_service, query, max_results=3):
    """Search for trending/popular videos"""
    try:
        search_response = youtube_service.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results,
            order='viewCount',
            type='video'
        ).execute()
        
        if not search_response.get('items'):
            return f"No trending videos found for query: {query}"
        
        results = []
        for item in search_response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            if len(description) > 200:
                description = description[:200] + "..."
            
            channel = item['snippet']['channelTitle']
            published = item['snippet']['publishedAt']
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            result_text = f"""
Title: {title}
Channel: {channel}
Published: {published}
URL: {url}
Description: {description}
{'='*50}"""
            results.append(result_text)
        
        return f"Trending YouTube Videos for '{query}':\n" + "\n".join(results)
        
    except Exception as e:
        return f"Error searching trending videos: {str(e)}"

# --- 1. Initialize the LLM ---
print("Initializing Ollama LLM with model 'llama3.2'...")
llm = Ollama(model="llama3.2")

# --- 2. Define the Universal YouTube Tool ---
print("Setting up universal YouTube search tool with OAuth2...")

# Universal YouTube tool that handles all operations
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
    func=youtube_universal_tool
)

tools = [youtube_tool]

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
    max_iterations=3,  # Reduced iterations to prevent excessive looping
    early_stopping_method="generate",  # Stop early if agent generates final answer
    return_intermediate_steps=True  # Show intermediate steps for debugging
)

# --- 6. Test authentication first ---
print("\n--- Testing YouTube OAuth2 Authentication ---")
youtube_service, auth_error = authenticate_youtube_oauth2()
if auth_error:
    print(f"Authentication failed: {auth_error}")
    print("\nMake sure you have:")
    print("1. Downloaded your OAuth2 credentials.json file from Google Cloud Console")
    print("2. Placed it in the same directory as this script")
    print("3. Enabled YouTube Data API v3 in your Google Cloud project")
else:
    print("✓ YouTube OAuth2 authentication successful!")

# --- 7. Run the Agent Executor with a Prompt ---
print("\nInvoking the agent with the YouTube search prompt...")
prompt_text = "What are the most recent videos from the 'Lex Fridman' YouTube channel? Just show me the results and conclude."

try:
    response = agent_executor.invoke({
        "input": prompt_text
    })
    print("\n--- Agent's Final Output ---")
    print(response['output'])
    
    # Show intermediate steps if available
    if 'intermediate_steps' in response:
        print(f"\n--- Process Summary ---")
        print(f"Total steps taken: {len(response['intermediate_steps'])}")
        for i, (action, observation) in enumerate(response['intermediate_steps']):
            print(f"Step {i+1}: {action.tool} - {action.tool_input[:50]}...")
            
except Exception as e:
    print(f"\nAn error occurred while running the agent: {e}")

# Optional: Test the tool directly with different prompts
print("\n--- Testing the YouTube tool with different prompts ---")
test_prompts = [
    "recent videos from Lex Fridman channel",
    "search for 5 python programming tutorials", 
    "trending AI videos",
    "latest machine learning videos"
]

for test_prompt in test_prompts:
    try:
        print(f"\nTesting: '{test_prompt}'")
        result = youtube_tool.func(test_prompt)
        print(result[:300] + "..." if len(result) > 300 else result)
        print("-" * 50)
    except Exception as e:
        print(f"Error testing prompt '{test_prompt}': {e}")
