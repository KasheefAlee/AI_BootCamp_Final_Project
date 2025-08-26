import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
TAVILY_API_KEY = "tvly-devzxczxczxczxczxcGgSIaR3bja"
HUGGINGFACE_API_KEY = "hf_QpbLXdsfsdfsdfsdfsdfsUKETe"
OLLAMA_API_URL = "http://localhost:11434"

# Model Configuration
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "true").lower() == "true"
MATH_MODEL = os.getenv("MATH_MODEL", "distilgpt2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Agent Configuration
AGENT_CONFIG = {
    "max_iterations": 10,
    "temperature": 0.1,
    "verbose": True
}

# Paths
DOCUMENTS_PATH = "documents/"
BENCHMARKS_PATH = "benchmarks/"
CHROMA_DB_PATH = "chroma_db/"

# Create directories if they don't exist
for path in [DOCUMENTS_PATH, BENCHMARKS_PATH, CHROMA_DB_PATH]:
    os.makedirs(path, exist_ok=True)