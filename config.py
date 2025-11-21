# config.py

# YouTube API Settings
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

# AI Agent Settings
LLM_MODEL = 'gemma3n'  # Or any other Ollama model
LANGCHAIN_PROMPT_HUB = 'hwchase17/react'
MAX_AGENT_ITERATIONS = 5

# GUI Settings
APP_TITLE = "YouTube AI Assistant"
APP_GEOMETRY = "1400x900"
MAX_SEARCH_RESULTS = 15 # Max results the agent can request

# --- Color Theme ---
# A modern dark theme with gradients for a slick look.
THEME = {
    "bg": "#2b2b2b",
    "bg_light": "#353535",
    "bg_dark": "#404040",
    "fg": "#ffffff",
    "fg_secondary": "#cccccc",
    "accent": "#0078d4",
    # --- Gradient Colors for the Slick UI ---
    "gradient_start": "#4a4a4a", # Darker grey at the top
    "gradient_end": "#2b2b2b",   # Pitch black at the bottom
}
