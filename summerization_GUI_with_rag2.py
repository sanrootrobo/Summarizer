import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import logging
import yaml
from pathlib import Path

# --- Third-Party Imports ---
# pip install langchain-google-genai google-generativeai PyMuPDF python-docx langchain-community langchain faiss-cpu ollama requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import requests
import fitz # PyMuPDF
import docx # python-docx

# --- Backend Classes ---

class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str) -> str | None:
        try:
            with open(Path(filepath), 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10: raise ValueError("Invalid API key")
            logging.info(f"API key loaded successfully from {filepath}.")
            return api_key
        except Exception as e:
            logging.error(f"Error reading API key file '{filepath}': {e}")
            return None

class UnifiedChatProcessor:
    def __init__(self, config: dict):
        self.mode = config.get("mode") # "Gemini" or "RAG"
        self.file_paths = config.get("file_paths", [])
        self.prompt_template = config.get("prompt_template", "")
        self.llm = None
        self.retriever = None
        self.chain = None
        self.chat_history = []

        if self.mode == "Gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=config["model_name"],
                google_api_key=config["api_key"],
                temperature=config["temperature"],
                max_output_tokens=config["max_tokens"]
            )
        elif self.mode == "RAG":
            self.llm = Ollama(model=config["ollama_model"])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _load_and_vectorize_documents(self) -> bool:
        """Loads documents and creates a FAISS vector store retriever."""
        docs = []
        logging.info("Chat Engine: Loading documents...")
        for file_path in self.file_paths:
            try:
                path = Path(file_path)
                if path.suffix.lower() == ".pdf": loader = PyPDFLoader(file_path)
                elif path.suffix.lower() == ".docx": loader = Docx2txtLoader(file_path)
                elif path.suffix.lower() == ".txt": loader = TextLoader(file_path, encoding='utf-8')
                else: continue
                docs.extend(loader.load())
                logging.info(f"Successfully loaded document: {path.name}")
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
        
        if not docs:
            logging.error("No documents were loaded successfully.")
            return False

        logging.info("Chat Engine: Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)

        logging.info("Chat Engine: Creating embeddings and FAISS vector store...")
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.from_documents(texts, embeddings)
            self.retriever = vectorstore.as_retriever()
            logging.info("Chat Engine: Vector store created successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to create embeddings/vector store: {e}", exc_info=True)
            messagebox.showerror("Embedding Error", f"Could not create vector store. Is your Ollama server running and is 'nomic-embed-text' model available?\n\nError: {e}")
            return False

    def initialize(self) -> bool:
        """Initializes the chat processor, including document vectorization and chain creation."""
        if not self._load_and_vectorize_documents():
            return False

        # --- NEW: Custom prompt to inject the user's desired persona/instructions ---
        _template = f"""
        {self.prompt_template}

        Given the above instructions and the following conversation, rephrase the follow up question to be a standalone question.

        Chat History:
        {{chat_history}}
        Follow Up Input: {{question}}
        Standalone question:
        """
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        )
        logging.info(f"Successfully initialized conversational chain in '{self.mode}' mode.")
        return True

    def query(self, question: str) -> str:
        """Sends a question to the active chain and returns the answer."""
        if not self.chain:
            return "Error: The chat chain is not initialized."
        
        logging.info(f"Querying the '{self.mode}' chain...")
        try:
            result = self.chain.invoke({"question": question})
            return result.get("answer", "No answer found in the result.")
        except Exception as e:
            logging.error(f"An error occurred during query: {e}", exc_info=True)
            return f"An error occurred: {e}"

# --- GUI Specific Classes ---
class QueueHandler(logging.Handler):
    def __init__(self, log_queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record): self.log_queue.put(self.format(record))

class ScraperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Chat Agent (Gemini & RAG)")
        self.root.geometry("900x850")

        # --- Variables ---
        self.processing_model_var = tk.StringVar(value="Ollama RAG")
        self.ollama_model_var = tk.StringVar(value="llama3")
        self.api_key_file_var = tk.StringVar()
        self.gemini_model_name_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar()
        self.max_tokens_var = tk.IntVar()
        
        self.chat_processor: UnifiedChatProcessor | None = None

        # --- UI Construction ---
        self.create_widgets()
        self.load_initial_settings()
        self.toggle_engine_settings()
        threading.Thread(target=self.populate_ollama_models_dropdown, daemon=True).start()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        settings_pane = ttk.Frame(main_frame); settings_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        chat_pane = ttk.Frame(main_frame); chat_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Settings Pane ---
        notebook = ttk.Notebook(settings_pane)
        notebook.pack(fill="both", expand=True)

        # --- RESTORED: Three-tab layout ---
        source_tab = ttk.Frame(notebook, padding="10")
        ai_tab = ttk.Frame(notebook, padding="10")
        prompt_tab = ttk.Frame(notebook, padding="10")
        notebook.add(source_tab, text="1. Source & Engine")
        notebook.add(ai_tab, text="2. AI Settings")
        notebook.add(prompt_tab, text="3. Prompt Editor")

        # --- Contents of Tab 1: Source & Engine ---
        engine_frame = ttk.LabelFrame(source_tab, text="Select Processing Engine", padding="10")
        engine_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(engine_frame, text="Ollama RAG", variable=self.processing_model_var, value="Ollama RAG", command=self.toggle_engine_settings).pack(anchor=tk.W)
        ttk.Radiobutton(engine_frame, text="Gemini Chat", variable=self.processing_model_var, value="Gemini Chat", command=self.toggle_engine_settings).pack(anchor=tk.W)

        upload_frame = ttk.LabelFrame(source_tab, text="Local Document Source", padding="10")
        upload_frame.pack(fill=tk.X)
        upload_buttons_frame = ttk.Frame(upload_frame)
        upload_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        self.add_files_button = ttk.Button(upload_buttons_frame, text="Add Files...", command=self.add_files)
        self.add_files_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        self.clear_files_button = ttk.Button(upload_buttons_frame, text="Clear List", command=self.clear_files)
        self.clear_files_button.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        self.file_listbox = tk.Listbox(upload_frame, height=5)
        self.file_listbox.pack(fill=tk.X, pady=5)

        # --- Contents of Tab 2: AI Settings ---
        self.gemini_settings_frame = ttk.LabelFrame(ai_tab, text="Gemini Pro Settings", padding=10)
        self.gemini_settings_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(self.gemini_settings_frame, text="API Key File:").pack(fill=tk.X)
        api_frame = ttk.Frame(self.gemini_settings_frame); api_frame.pack(fill=tk.X, pady=(0,5))
        ttk.Entry(api_frame, textvariable=self.api_key_file_var).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(api_frame, text="...", width=3, command=self.browse_api_key).pack(side=tk.RIGHT)
        ttk.Label(self.gemini_settings_frame, text="Model Name:").pack(fill=tk.X)
        ttk.Entry(self.gemini_settings_frame, textvariable=self.gemini_model_name_var).pack(fill=tk.X, pady=(0,5))
        ttk.Label(self.gemini_settings_frame, text="Temperature:").pack(fill=tk.X)
        ttk.Scale(self.gemini_settings_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var).pack(fill=tk.X, pady=(0,5))
        ttk.Label(self.gemini_settings_frame, text="Max Output Tokens:").pack(fill=tk.X)
        ttk.Spinbox(self.gemini_settings_frame, from_=1024, to=16384, increment=128, textvariable=self.max_tokens_var).pack(fill=tk.X)

        self.ollama_settings_frame = ttk.LabelFrame(ai_tab, text="Ollama Settings", padding=10)
        self.ollama_settings_frame.pack(fill=tk.X)
        ttk.Label(self.ollama_settings_frame, text="LLM Model Name:").pack(fill=tk.X)
        ollama_model_frame = ttk.Frame(self.ollama_settings_frame)
        ollama_model_frame.pack(fill=tk.X)
        self.ollama_model_combobox = ttk.Combobox(ollama_model_frame, textvariable=self.ollama_model_var)
        self.ollama_model_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.refresh_ollama_button = ttk.Button(ollama_model_frame, text="🔄", width=3, command=self.populate_ollama_models_dropdown)
        self.refresh_ollama_button.pack(side=tk.RIGHT)

        # --- Contents of Tab 3: Prompt Editor ---
        ttk.Button(prompt_tab, text="Load Prompt from File...", command=self.load_prompt_from_file).pack(fill=tk.X, pady=(0,5))
        self.prompt_text = scrolledtext.ScrolledText(prompt_tab, wrap=tk.WORD, height=10)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # --- Control Buttons ---
        self.init_button = ttk.Button(settings_pane, text="Initialize Chat", command=self.start_init_thread)
        self.init_button.pack(fill=tk.X, pady=(10, 0))

        # --- Chat Pane ---
        chat_history_frame = ttk.LabelFrame(chat_pane, text="Conversation")
        chat_history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_history_text = scrolledtext.ScrolledText(chat_history_frame, state='disabled', wrap=tk.WORD, font=("Helvetica", 10))
        self.chat_history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        chat_input_frame = ttk.LabelFrame(chat_pane, text="Your Message")
        chat_input_frame.pack(fill=tk.X)
        self.chat_input_var = tk.StringVar()
        self.chat_input_entry = ttk.Entry(chat_input_frame, textvariable=self.chat_input_var, font=("Helvetica", 10))
        self.chat_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.send_button = ttk.Button(chat_input_frame, text="Send", command=self.send_message_thread)
        self.send_button.pack(side=tk.RIGHT, padx=(0, 5), pady=5)
        self.chat_input_entry.bind("<Return>", lambda event: self.send_message_thread())
        self._set_chat_interface_state(tk.DISABLED)

    def _set_child_widgets_state(self, parent, state):
        for widget in parent.winfo_children():
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass # Widget doesn't have a 'state' option
            self._set_child_widgets_state(widget, state)

    def _set_chat_interface_state(self, state):
        self.chat_input_entry.config(state=state)
        self.send_button.config(state=state)

    def toggle_engine_settings(self):
        """Show/hide settings based on the selected engine."""
        engine = self.processing_model_var.get()
        if engine == "Gemini Chat":
            self._set_child_widgets_state(self.gemini_settings_frame, tk.NORMAL)
            self._set_child_widgets_state(self.ollama_settings_frame, tk.DISABLED)
        else: # Ollama RAG
            self._set_child_widgets_state(self.gemini_settings_frame, tk.DISABLED)
            self._set_child_widgets_state(self.ollama_settings_frame, tk.NORMAL)

    def add_files(self):
        files = filedialog.askopenfilenames(title="Select Documents", filetypes=[("All Supported", "*.pdf *.docx *.txt"), ("PDF", "*.pdf"), ("Word", "*.docx"), ("Text", "*.txt")])
        for f in files:
            if f not in self.file_listbox.get(0, tk.END):
                self.file_listbox.insert(tk.END, f)
    
    def clear_files(self):
        self.file_listbox.delete(0, tk.END)

    def load_initial_settings(self):
        self.api_key_file_var.set("gemini_api.key")
        self.gemini_model_name_var.set("gemini-1.5-flash")
        self.temperature_var.set(0.7)
        self.max_tokens_var.set(2048)
        self.load_prompt_from_file(is_initial=True) # Load default prompt

    def browse_api_key(self):
        filepath = filedialog.askopenfilename(title="Select API Key File");
        if filepath: self.api_key_file_var.set(filepath)

    def load_prompt_from_file(self, filepath="prompt.md", is_initial=False):
        if not is_initial:
            filepath = filedialog.askopenfilename(title="Select Prompt File", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if not filepath: return
        try:
            with open(filepath, 'r') as f:
                prompt_content = f.read()
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, prompt_content)
        except Exception as e:
            if not is_initial: messagebox.showerror("Error", f"Could not load prompt file:\n{e}")
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, "# Could not load prompt.md\n\nYou are a helpful AI assistant.")

    def _fetch_ollama_models(self) -> list[str]:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            response.raise_for_status()
            return sorted([m['name'] for m in response.json().get('models', [])])
        except requests.exceptions.RequestException:
            logging.warning("Could not connect to Ollama server.")
            return []
        except Exception: return []

    def populate_ollama_models_dropdown(self):
        self.refresh_ollama_button.config(state=tk.DISABLED)
        model_list = self._fetch_ollama_models()
        if model_list:
            self.ollama_model_combobox['values'] = model_list
            if self.ollama_model_var.get() not in model_list: self.ollama_model_var.set(model_list[0])
        else:
            self.ollama_model_combobox['values'] = []
            self.ollama_model_var.set("[Ollama not running or no models]")
        self.refresh_ollama_button.config(state=tk.NORMAL)

    def _log_message(self, who: str, message: str):
        """Appends a message to the chat history widget."""
        self.chat_history_text.config(state='normal')
        if who == "user": self.chat_history_text.insert(tk.END, f"You:\n", ("bold",))
        elif who == "system": self.chat_history_text.insert(tk.END, f"System:\n", ("bold", "italic"))
        else: self.chat_history_text.insert(tk.END, f"{self.processing_model_var.get()}:\n", ("bold",))
        
        self.chat_history_text.insert(tk.END, f"{message}\n\n")
        self.chat_history_text.config(state='disabled')
        self.chat_history_text.yview(tk.END)

    # --- Threading and Core Logic ---
    def start_init_thread(self):
        """Starts the chat initialization in a background thread."""
        self.init_button.config(state=tk.DISABLED)
        self.chat_processor = None
        self._set_chat_interface_state(tk.DISABLED)
        threading.Thread(target=self.run_chat_initialization, daemon=True).start()

    def run_chat_initialization(self):
        """Gathers config from all tabs and initializes the UnifiedChatProcessor."""
        engine = self.processing_model_var.get()
        file_paths = self.file_listbox.get(0, tk.END)
        prompt = self.prompt_text.get("1.0", tk.END).strip()

        if not file_paths:
            messagebox.showerror("Input Error", "Please add at least one document to chat with.")
            self.init_button.config(state=tk.NORMAL); return

        if not prompt:
            messagebox.showerror("Input Error", "Prompt cannot be empty.")
            self.init_button.config(state=tk.NORMAL); return

        config = {"file_paths": file_paths, "prompt_template": prompt}

        if engine == "Gemini Chat":
            api_key = APIKeyManager.get_api_key(self.api_key_file_var.get())
            if not api_key:
                messagebox.showerror("API Key Error", "Could not load a valid Gemini API key.")
                self.init_button.config(state=tk.NORMAL); return
            config.update({ "mode": "Gemini", "api_key": api_key, "model_name": self.gemini_model_name_var.get(), "temperature": self.temperature_var.get(), "max_tokens": self.max_tokens_var.get()})
        
        elif engine == "Ollama RAG":
            ollama_model = self.ollama_model_var.get()
            if "not running" in ollama_model or not ollama_model:
                messagebox.showerror("Ollama Error", "Please select a valid, running Ollama model.")
                self.init_button.config(state=tk.NORMAL); return
            config.update({"mode": "RAG", "ollama_model": ollama_model})
        
        try:
            self.chat_processor = UnifiedChatProcessor(config)
            if self.chat_processor.initialize():
                logging.info("Chat initialization successful.")
                self._set_chat_interface_state(tk.NORMAL)
                self.chat_history_text.config(state='normal'); self.chat_history_text.delete('1.0', tk.END); self.chat_history_text.config(state='disabled')
                self._log_message("system", f"Chat with {engine} initialized. You can now ask questions about your documents.")
            else:
                logging.error("Chat initialization failed.")
                self.init_button.config(state=tk.NORMAL)
        except Exception as e:
            logging.error(f"Fatal error during initialization: {e}", exc_info=True)
            messagebox.showerror("Fatal Error", f"An unexpected error occurred during initialization:\n\n{e}")
            self.init_button.config(state=tk.NORMAL)

    def send_message_thread(self):
        """Starts the query process in a background thread."""
        user_input = self.chat_input_var.get()
        if not user_input.strip(): return
        
        self._set_chat_interface_state(tk.DISABLED)
        self._log_message("user", user_input)
        self.chat_input_var.set("")

        threading.Thread(target=self.run_query, args=(user_input,), daemon=True).start()

    def run_query(self, user_input):
        """Sends the user's message to the chat processor."""
        if not self.chat_processor:
            self._log_message("system", "Error: Chat is not initialized.")
            self._set_chat_interface_state(tk.NORMAL); return

        response = self.chat_processor.query(user_input)
        self._log_message("model", response)
        self._set_chat_interface_state(tk.NORMAL)
        self.chat_input_entry.focus()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root = tk.Tk()
    app = ScraperApp(root)
    # Define custom tags for chat styling
    app.chat_history_text.tag_configure("bold", font=("Helvetica", 10, "bold"))
    app.chat_history_text.tag_configure("italic", font=("Helvetica", 10, "italic"))
    root.mainloop()
