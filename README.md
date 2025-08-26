# 🤖 Northern PEPs Agentic AI

A 3-day AI bootcamp project for building intelligent agents with Chain of Thought reasoning, task decomposition, and dynamic tool selection.

## 🎯 Project Overview

This project demonstrates the core principles of agentic AI by building a lightweight agent capable of:
- **Task Decomposition**: Breaking complex queries into logical subtasks
- **Dynamic Tool Selection**: Choosing the right tool for each task
- **Chain of Thought Reasoning**: Step-by-step problem solving
- **Performance Evaluation**: Benchmarking against industry standards

## 🧠 System Components

### 1. Controller Agent
- **Input**: Receives complex user queries
- **Logic**: Uses Chain of Thought reasoning to decompose queries
- **Action**: Decides which tool to invoke for each step

### 2. Tool Interfaces
- **Web Search**: Real-time information retrieval using Tavily API
- **Calculator**: Basic arithmetic operations
- **Math Solver**: Word problem solving (GSM8k-style)
- **Document Q&A**: RAG-based document search

### 3. Evaluation Framework
- **LAMA Benchmark**: Factual accuracy evaluation
- **GSM8k Benchmark**: Mathematical reasoning evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for local LLM)
- API keys (Tavily, HuggingFace)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd agentic-ai-bootcamp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys in config.py or environment variables
```

### Usage

#### 🚀 Streamlit Web App (Recommended)
```bash
python run.py
# or
python start_app.py
```
The web app will open at http://localhost:8501

#### Interactive CLI Mode
```bash
python run.py --cli
```

#### Run Evaluation Benchmarks
```bash
python run.py --evaluate
```

#### Demo Mode
```bash
python run.py --demo
```

## 🛠️ Available Tools

1. **Web Search Tool**
   - Searches the web for current information
   - Uses Tavily API for real-time results
   - Returns formatted search results with sources

2. **Calculator Tool**
   - Performs safe mathematical calculations
   - Supports basic arithmetic operations
   - Uses AST parsing for security

3. **Math Solver Tool**
   - Solves word problems step-by-step
   - Extracts numbers and identifies operations
   - Provides reasoning explanations

4. **Document Q&A Tool**
   - Searches local documents (PDF, DOCX, TXT)
   - Uses RAG with sentence transformers
   - Returns relevant document chunks

## 📊 Evaluation Benchmarks

### LAMA Benchmark
- Tests factual accuracy
- Sample questions: geography, literature, science
- Measures answer correctness

### GSM8k Benchmark
- Tests mathematical reasoning
- Word problem solving
- Step-by-step solution verification

## 🏗️ Project Structure

```
agentic-ai-bootcamp/
├── agent_controller.py    # Main agent controller
├── tools/                 # Tool implementations
│   ├── web_search.py     # Web search tool
│   ├── calculator.py     # Calculator tool
│   ├── math_solver.py    # Math solver tool
│   └── document_qa.py    # Document Q&A tool
├── evaluator.py          # Evaluation framework
├── config.py             # Configuration
├── run.py               # CLI interface
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🎓 Learning Objectives

By completing this bootcamp, students will:

1. **Understand Agentic AI Principles**
   - Task decomposition and planning
   - Tool selection and orchestration
   - Chain of Thought reasoning

2. **Build Practical Skills**
   - LangChain integration
   - Tool development
   - Prompt engineering
   - Evaluation frameworks

3. **Evaluate System Performance**
   - Benchmark implementation
   - Accuracy measurement
   - Reasoning assessment

## 🔧 Configuration

Edit `config.py` to customize:
- API keys and endpoints
- Model configurations
- Agent parameters
- File paths

## 📈 Performance Metrics

The system is evaluated on:
- **Factual Accuracy**: LAMA benchmark performance
- **Reasoning Ability**: GSM8k benchmark performance
- **Execution Time**: Response latency
- **Tool Usage**: Appropriate tool selection

## 🚀 Stretch Goals

For advanced students:
- Fine-tune math-solving models
- Implement additional tools
- Enhance evaluation metrics
- Build web interface

## 🤝 Contributing

This is a bootcamp project designed for learning. Feel free to:
- Experiment with different prompts
- Add new tools
- Improve evaluation metrics
- Share your insights



