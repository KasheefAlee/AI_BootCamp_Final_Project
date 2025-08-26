import streamlit as st
import time
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title=" Northern PEPs Agent ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tool-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    """Initialize the agent controller (cached)"""
    try:
        from agent_controller import AgentController
        return AgentController()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None

@st.cache_resource
def initialize_evaluator():
    """Initialize the evaluator (cached)"""
    try:
        from evaluator import Evaluator
        return Evaluator()
    except Exception as e:
        st.error(f"Failed to initialize evaluator: {e}")
        return None

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Northern PEPs AI Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">By Kashif Ali Azim & Sajil Rahim</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "💬 Chat with Agent", "📊 Evaluation", "🛠️ Tools Info", "📚 About"]
        )
        
        st.markdown("---")
        st.markdown("### 🔧 Configuration")
        
        # Agent status
        agent = initialize_agent()
        if agent and agent.agent_executor:
            st.success("✅ Agent Ready")
            st.info(f"Available Tools: {len(agent.get_available_tools())}")
        else:
            st.error("❌ Agent Not Ready")
        
        # Model info
        st.markdown("### 🤖 Model Info")
        st.info("Using Ollama (llama2)")
        st.info("Chain of Thought Reasoning Enabled")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "💬 Chat with Agent":
        show_chat_page(agent)
    elif page == "📊 Evaluation":
        show_evaluation_page()
    elif page == "🛠️ Tools Info":
        show_tools_page(agent)
    elif page == "📚 About":
        show_about_page()

def show_home_page():
    """Home page with project overview"""
    st.header("🏠 Welcome to Agentic AI Bootcamp")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This 3-day AI bootcamp demonstrates the core principles of **agentic AI** by building a lightweight agent capable of:
        
        - **🧠 Task Decomposition**: Breaking complex queries into logical subtasks
        - **🛠️ Dynamic Tool Selection**: Choosing the right tool for each task  
        - **💭 Chain of Thought Reasoning**: Step-by-step problem solving
        - **📊 Performance Evaluation**: Benchmarking against industry standards
        
        ### 🚀 Key Features
        
        - **Controller Agent** with intelligent task decomposition
        - **4 Specialized Tools**: Web Search, Calculator, Math Solver, Document Q&A
        - **Chain of Thought** reasoning for complex problem solving
        - **Evaluation Framework** using LAMA and GSM8k benchmarks
        """)
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        agent = initialize_agent()
        if agent:
            tools = agent.get_available_tools()
            st.metric("Available Tools", len(tools))
            st.metric("Model", "Ollama (llama2)")
            st.metric("Reasoning", "Chain of Thought")
        else:
            st.metric("Available Tools", "0")
            st.metric("Model", "Not Ready")
            st.metric("Reasoning", "Not Ready")
    
    st.markdown("---")
    
    # Quick start section
    st.header("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1. 💬 Chat with Agent
        Try asking complex questions that require multiple tools:
        - "What is the capital of Pakistan and what is 2 + 2?"
        - "Janet's dogs eat 2 pounds of dog food each day. How many pounds do they eat in 7 days?"
        """)
    
    with col2:
        st.markdown("""
        ### 2. 📊 Run Evaluation
        Test the agent's performance on:
        - **LAMA Benchmark**: Factual accuracy
        - **GSM8k Benchmark**: Mathematical reasoning
        """)
    
    with col3:
        st.markdown("""
        ### 3. 🛠️ Explore Tools
        Learn about the available tools:
        - Web Search (Tavily API)
        - Calculator (Safe arithmetic)
        - Math Solver (Word problems)
        - Document Q&A (RAG)
        """)

def show_chat_page(agent):
    """Chat interface with the agent"""
    st.header("💬 Chat with Agent")
    
    if not agent or not agent.agent_executor:
        st.error("❌ Agent is not ready. Please check the configuration.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    response = agent.process_query(prompt)
                    
                    if response.success:
                        # Display final answer
                        st.markdown(f"**Answer:** {response.final_answer}")
                        
                        # Display reasoning steps if available
                        if response.reasoning_steps:
                            with st.expander("🧠 Reasoning Steps"):
                                for i, step in enumerate(response.reasoning_steps, 1):
                                    st.markdown(f"**Step {i}:** {step}")
                        
                        # Display execution time
                        st.info(f"⏱️ Execution time: {response.execution_time:.2f}s")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"**Answer:** {response.final_answer}\n\n⏱️ Execution time: {response.execution_time:.2f}s"
                        })
                    else:
                        st.error(f"❌ Error: {response.final_answer}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"❌ Error: {response.final_answer}"
                        })
                        
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ Error: {str(e)}"
                    })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

def show_evaluation_page():
    """Evaluation page with benchmarks"""
    st.header("📊 Evaluation Benchmarks")
    
    evaluator = initialize_evaluator()
    if not evaluator:
        st.error("❌ Evaluator is not ready.")
        return
    
    st.markdown("""
    ### 🎯 Benchmark Overview
    
    The system is evaluated using two industry-standard benchmarks:
    
    - **📚 LAMA Benchmark**: Tests factual knowledge and recall accuracy
    - **🧮 GSM8k Benchmark**: Tests mathematical reasoning and problem-solving
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 LAMA Benchmark")
        st.markdown("""
        **Purpose**: Evaluate factual accuracy
        
        **Sample Questions**:
        - What is the capital of France?
        - Who wrote Romeo and Juliet?
        - What is the largest planet in our solar system?
        """)
    
    with col2:
        st.markdown("### 🧮 GSM8k Benchmark")
        st.markdown("""
        **Purpose**: Evaluate mathematical reasoning
        
        **Sample Questions**:
        - Janet's dogs eat 2 pounds of dog food each day. How many pounds do they eat in 7 days?
        - There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did they plant?
        """)
    
    st.markdown("---")
    
    # Run evaluation
    if st.button("🚀 Run Full Evaluation", type="primary"):
        with st.spinner("Running evaluation benchmarks..."):
            try:
                results = evaluator.run_full_evaluation()
                
                # Display results
                st.success("✅ Evaluation completed!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "LAMA Accuracy", 
                        f"{results['lama']['accuracy']:.1%}",
                        f"{results['lama']['correct_answers']}/{results['lama']['total_questions']}"
                    )
                
                with col2:
                    st.metric(
                        "GSM8k Accuracy", 
                        f"{results['gsm8k']['accuracy']:.1%}",
                        f"{results['gsm8k']['correct_answers']}/{results['gsm8k']['total_questions']}"
                    )
                
                with col3:
                    st.metric(
                        "Overall Accuracy", 
                        f"{results['summary']['overall_accuracy']:.1%}",
                        f"{results['summary']['total_questions']} questions"
                    )
                
                # Detailed results
                with st.expander("📋 Detailed Results"):
                    st.json(results)
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")

def show_tools_page(agent):
    """Tools information page"""
    st.header("🛠️ Available Tools")
    
    if not agent:
        st.error("❌ Agent is not ready.")
        return
    
    tools_info = agent.get_tool_info()
    
    st.markdown("### 🎯 Tool Overview")
    st.markdown("The agent has access to 4 specialized tools for different types of tasks:")
    
    # Tool cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Web Search Tool
        st.markdown("""
        <div class="tool-card">
            <h4>🔍 Web Search Tool</h4>
            <p><strong>Purpose:</strong> Real-time information retrieval</p>
            <p><strong>API:</strong> Tavily Search API</p>
            <p><strong>Capabilities:</strong></p>
            <ul>
                <li>Current events and news</li>
                <li>Factual information lookup</li>
                <li>Domain-specific searches</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculator Tool
        st.markdown("""
        <div class="tool-card">
            <h4>🧮 Calculator Tool</h4>
            <p><strong>Purpose:</strong> Mathematical calculations</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>Safe expression evaluation</li>
                <li>Basic arithmetic operations</li>
                <li>AST parsing for security</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Math Solver Tool
        st.markdown("""
        <div class="tool-card">
            <h4>📐 Math Solver Tool</h4>
            <p><strong>Purpose:</strong> Word problem solving</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>Step-by-step reasoning</li>
                <li>Number extraction</li>
                <li>Operation identification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Document Q&A Tool
        st.markdown("""
        <div class="tool-card">
            <h4>📄 Document Q&A Tool</h4>
            <p><strong>Purpose:</strong> Document search and retrieval</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>RAG-based search</li>
                <li>PDF, DOCX, TXT support</li>
                <li>Semantic similarity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tool status
    st.markdown("### 📊 Tool Status")
    available_tools = agent.get_available_tools()
    
    for tool_name in ["web_search", "calculator", "math_solver", "document_qa"]:
        if tool_name in available_tools:
            st.success(f"✅ {tool_name.replace('_', ' ').title()}")
        else:
            st.error(f"❌ {tool_name.replace('_', ' ').title()}")

def show_about_page():
    """About page with project information"""
    st.header("📚 About Agentic AI Bootcamp")
    
    st.markdown("""
    ### 🎯 Project Mission
    
    This project is designed for a **3-day AI bootcamp** that immerses students in the core principles of agentic AI. 
    Students will design, build, and test a lightweight agent capable of task decomposition and dynamic tool selection.
    
    ### 🧠 Key Learning Objectives
    
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
    
    ### 🏗️ Technical Architecture
    
    ```
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Streamlit UI  │    │  Agent          │    │   Tools Layer   │
    │                 │◄──►│  Controller     │◄──►│                 │
    │   - Web App     │    │                 │    │  - Web Search   │
    │   - Chat        │    │  - Chain of     │    │  - Calculator   │
    │   - Evaluation  │    │    Thought      │    │  - Math Solver  │
    └─────────────────┘    │  - Memory       │    │  - Document QA  │
                          └─────────────────┘    └─────────────────┘
    ```
    
    ### 🛠️ Technology Stack
    
    - **Framework**: LangChain
    - **Language Model**: Ollama (llama2)
    - **Web Framework**: Streamlit
    - **Vector Store**: ChromaDB
    - **Embeddings**: Sentence Transformers
    - **APIs**: Tavily (web search)
    
    ### 📊 Evaluation Framework
    
    The system is evaluated using industry-standard benchmarks:
    
    - **LAMA Benchmark**: Factual accuracy evaluation
    - **GSM8k Benchmark**: Mathematical reasoning evaluation
    
    ### 🚀 Future Enhancements
    
    - Fine-tuned math-solving models
    - Additional specialized tools
    - Enhanced evaluation metrics
    - Real-time performance monitoring
    
    ### 📝 License
    
    This project is for educational purposes as part of the AI bootcamp curriculum.
    """)

if __name__ == "__main__":
    main()
