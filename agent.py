# agent.py

import re
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool

import config

class AssistantAgent:
    """Handles the AI agent, tools, and processing of natural language."""
    
    def __init__(self, youtube_service):
        self.youtube_service = youtube_service
        self.last_search_results = []
        
        try:
            llm = Ollama(model=config.LLM_MODEL)
            prompt_template = hub.pull(config.LANGCHAIN_PROMPT_HUB)
            
            youtube_tool = Tool(
                name="youtube_search",
                description="""Searches YouTube for videos. Use this to find videos based on a query, channel, or topic. You can specify the number of results and the sorting order (relevance, recent, popular). This should only be called once per user request.""",
                func=self.run_youtube_tool
            )
            
            agent = create_react_agent(llm, [youtube_tool], prompt_template)
            
            # --- FIXED SECTION ---
            # The problematic 'early_stopping_method' parameter has been removed.
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=[youtube_tool],
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=config.MAX_AGENT_ITERATIONS,
                return_intermediate_steps=True,
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the AI Agent: {e}")

    def run_youtube_tool(self, prompt: str) -> str:
        """Parses a natural language prompt and executes a YouTube search."""
        self.last_search_results = [] # Reset previous results
        try:
            query, max_results, order, channel_name = self._parse_prompt(prompt)
            videos = self.youtube_service.search_videos(query, max_results, order, channel_name)
            
            if not videos:
                return "No videos were found matching the criteria. I should report this to the user."

            self.last_search_results = videos
            
            summary = f"Successfully found {len(videos)} videos. The search is complete. I should now provide a final answer to the user based on these results:\n"
            summary += "\n".join([f"- '{v['title']}' by {v['channel']}" for v in videos[:5]])
            return summary
            
        except Exception as e:
            return f"An error occurred while searching YouTube: {str(e)}. I should report this error."

    def _parse_prompt(self, prompt: str) -> tuple:
        """Extracts search parameters from the user's prompt."""
        query = prompt
        max_results = 5
        order = 'relevance'
        channel_name = None
        prompt_lower = prompt.lower()

        if match := re.search(r'(\d+)\s*(results?|videos?)', prompt_lower):
            max_results = int(match.group(1))

        if any(k in prompt_lower for k in ['recent', 'latest', 'newest']):
            order = 'date'
        elif any(k in prompt_lower for k in ['trending', 'popular', 'most viewed']):
            order = 'viewCount'

        channel_pattern = r'\s*(?:from|by|on|channel)\s+([\w\s-]+)(?:\s+channel)?'
        if match := re.search(channel_pattern, query, flags=re.IGNORECASE):
            channel_name = match.group(1).strip()
            query = re.sub(channel_pattern, '', query, count=1, flags=re.IGNORECASE).strip()
            if not query:
                query = channel_name

        return query, max_results, order, channel_name

    def invoke(self, prompt: str):
        """Invokes the agent with a given prompt."""
        return self.agent_executor.invoke({"input": prompt})
