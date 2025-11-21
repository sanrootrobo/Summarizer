import os
from langchain_community.llms import Ollama
from langchain_community.tools import YouTubeSearchTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# --- 1. Initialize the LLM ---
# Connect to the local Ollama model, specifying 'llama3.2'.
# Make sure Ollama is running and the model is pulled.
print("Initializing Ollama LLM with model 'llama3.2'...")
llm = Ollama(model="llama3.2")

# --- 2. Define the Tools ---
# The agent will have access to this tool to search YouTube.
# We can customize it, for instance, to get the top 3 results.
print("Setting up YouTubeSearchTool...")
tools = [YouTubeSearchTool(max_results=3)]

# --- 3. Get the Prompt Template ---
# We pull a pre-built ReAct prompt template from LangChain Hub.
# This template is crucial as it guides the LLM on how to reason
# and format its thoughts and actions to use tools correctly.
print("Pulling the ReAct prompt template from LangChain Hub...")
prompt_template = hub.pull("hwchase17/react")

# --- 4. Create the Agent ---
# We combine the LLM, the tools, and the prompt into a runnable agent.
# This agent is the "brain" that will decide which tool to use.
print("Creating the ReAct agent...")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,
)

# --- 5. Create the Agent Executor ---
# The AgentExecutor is the runtime for the agent. It handles the loop of:
# agent -> tool -> agent -> tool -> ... until the task is complete.
print("Creating the Agent Executor...")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's thought process
    handle_parsing_errors=True # Gracefully handles LLM output errors
)

# --- 6. Run the Agent Executor with a Prompt ---
# We use .invoke() to run the agent. The input must be a dictionary.
print("\nInvoking the agent with the YouTube search prompt...")
prompt_text = "What are the most recent videos from the 'Lex Fridman' YouTube channel?"

try:
    response = agent_executor.invoke({
        "input": prompt_text
    })

    print("\n--- Agent's Final Output ---")
    # The final answer is in the 'output' key of the response dictionary.
    print(response['output'])

except Exception as e:
    print(f"\nAn error occurred while running the agent: {e}")
