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

def search_youtube_oauth2(query, max_results=3):
    """
    Search YouTube videos using OAuth2 authenticated YouTube Data API v3
    Returns a formatted string with video titles, URLs, and descriptions
    """
    youtube_service, error = authenticate_youtube_oauth2()
    
    if error:
        return f"Authentication Error: {error}"
    
    try:
        # Search for videos
        search_response = youtube_service.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results,
            order='date',  # Get most recent videos
            type='video'
        ).execute()
        
        if not search_response.get('items'):
            return f"No videos found for query: {query}"
        
        results = []
        for item in search_response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            # Truncate long descriptions
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
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"

def search_channel_videos_oauth2(channel_name, max_results=3):
    """
    Search for recent videos from a specific channel
    """
    youtube_service, error = authenticate_youtube_oauth2()
    
    if error:
        return f"Authentication Error: {error}"
    
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
            return search_youtube_oauth2(f"{channel_name} channel", max_results)
        
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

# --- 1. Initialize the LLM ---
print("Initializing Ollama LLM with model 'llama3.2'...")
llm = Ollama(model="llama3.2")

# --- 2. Define the Tools ---
print("Setting up YouTube search tools with OAuth2...")

# General YouTube search tool
youtube_search_tool = Tool(
    name="youtube_search",
    description="Search for YouTube videos by query. Input should be a search query string. Returns video titles, URLs, descriptions, and publication dates.",
    func=lambda query: search_youtube_oauth2(query, max_results=3)
)

# Channel-specific search tool
channel_search_tool = Tool(
    name="youtube_channel_search", 
    description="Search for recent videos from a specific YouTube channel. Input should be the channel name. Returns recent videos from that channel.",
    func=lambda channel: search_channel_videos_oauth2(channel, max_results=3)
)

tools = [youtube_search_tool, channel_search_tool]

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
    print("Testing channel search:")
    direct_result = channel_search_tool.func("Lex Fridman")
    print(direct_result[:500] + "..." if len(direct_result) > 500 else direct_result)
except Exception as e:
    print(f"Error testing channel tool: {e}")
