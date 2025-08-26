import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool

# Local imports
import config

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a subtask identified from user query"""
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    status: str = "pending"  # pending, completed, failed
    result: Any = None

@dataclass
class AgentResponse:
    """Response from the agent controller"""
    query: str
    tasks: List[Task]
    final_answer: str
    success: bool
    execution_time: float
    reasoning_steps: List[str] = None

class AgentController:
    """Simplified Agent Controller with Chain of Thought reasoning"""
    
    def __init__(self):
        """Initialize the Agent Controller"""
        self.llm = None
        self.tools = []
        self.agent_executor = None
        self.memory = None
        
        # Initialize components
        self._initialize_llm()
        self._initialize_tools()
        self._initialize_agent()
        
        logger.info("Agent Controller initialized successfully")
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if config.USE_LOCAL_MODELS:
                self.llm = OllamaLLM(
                    model="llama2",
                    base_url=config.OLLAMA_API_URL,
                    temperature=config.AGENT_CONFIG["temperature"]
                )
            else:
                # Fallback to a simple mock LLM for testing
                from langchain.llms.base import LLM
                class MockLLM(LLM):
                    def _call(self, prompt, stop=None):
                        return "I understand the query. Let me break this down into steps."
                    @property
                    def _llm_type(self):
                        return "mock"
                
                self.llm = MockLLM()
            
            logger.info(f"LLM initialized: {type(self.llm).__name__}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None
    
    def _initialize_tools(self):
        """Initialize tools using @tool decorators"""
        try:
            # Web Search Tool
            if config.TAVILY_API_KEY:
                try:
                    from tavily import TavilyClient
                    client = TavilyClient(api_key=config.TAVILY_API_KEY)
                    
                    @tool
                    def web_search(query: str) -> str:
                        """Search the web for current information and facts"""
                        try:
                            logger.info(f"Searching web for: {query}")
                            
                            response = client.search(
                                query=query,
                                search_depth="basic",
                                max_results=3,
                                include_answer=True
                            )
                            
                            # Format results
                            if response.get('answer'):
                                return f"Answer: {response['answer']}\n\nSources: {', '.join([r.get('url', 'N/A') for r in response.get('results', [])[:2]])}"
                            else:
                                results = response.get('results', [])
                                if results:
                                    formatted = []
                                    for result in results[:2]:
                                        formatted.append(f"Title: {result.get('title', 'N/A')}\nContent: {result.get('content', 'N/A')[:200]}...\nURL: {result.get('url', 'N/A')}")
                                    return "\n\n".join(formatted)
                                else:
                                    return "No search results found."
                                    
                        except Exception as e:
                            logger.error(f"Web search error: {e}")
                            return f"Search error: {str(e)}"
                    
                    self.tools.append(web_search)
                    logger.info("Web search tool initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize web search tool: {e}")
            
            # Calculator Tool
            try:
                import ast
                import operator
                import re
                
                @tool
                def calculator(expression: str) -> str:
                    """Perform mathematical calculations on expressions"""
                    try:
                        logger.info(f"Calculating: {expression}")
                        # Sanitize input: extract math-only fragment from free-form text
                        extracted = "".join(re.findall(r"[0-9+\-*/().\s]+", expression))
                        expr = extracted.strip() if extracted.strip() else expression.strip()
                        
                        # Parse and evaluate safely
                        tree = ast.parse(expr, mode='eval')
                        
                        def safe_eval(node):
                            """Safely evaluate AST node"""
                            if isinstance(node, ast.Num):
                                return node.n
                            elif isinstance(node, ast.BinOp):
                                operators = {
                                    ast.Add: operator.add,
                                    ast.Sub: operator.sub,
                                    ast.Mult: operator.mul,
                                    ast.Div: operator.truediv,
                                    ast.Pow: operator.pow,
                                    ast.USub: operator.neg,
                                    ast.UAdd: operator.pos,
                                    ast.Mod: operator.mod,
                                    ast.FloorDiv: operator.floordiv
                                }
                                return operators[type(node.op)](safe_eval(node.left), safe_eval(node.right))
                            elif isinstance(node, ast.UnaryOp):
                                operators = {
                                    ast.USub: operator.neg,
                                    ast.UAdd: operator.pos
                                }
                                return operators[type(node.op)](safe_eval(node.operand))
                            else:
                                raise ValueError(f"Unsafe operation: {type(node).__name__}")
                        
                        result = safe_eval(tree.body)
                        return f"Result: {result}"
                        
                    except Exception as e:
                        logger.error(f"Calculator error: {e}")
                        return f"Calculation error: {str(e)}"
                
                self.tools.append(calculator)
                logger.info("Calculator tool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize calculator tool: {e}")
            
            # Math Solver Tool
            try:
                import re
                
                @tool
                def math_solver(problem: str) -> str:
                    """Solve math word problems step by step"""
                    try:
                        logger.info(f"Solving math problem: {problem}")
                        
                        # Extract numbers from the problem
                        numbers = re.findall(r'\d+\.?\d*', problem)
                        
                        if len(numbers) < 2:
                            return f"Problem: {problem}\nCould not extract enough numbers for calculation."
                        
                        # Identify operation based on keywords
                        problem_lower = problem.lower()
                        
                        if any(word in problem_lower for word in ['add', 'plus', '+', 'sum', 'total']):
                            result = float(numbers[0]) + float(numbers[1])
                            return f"Problem: {problem}\nSolution: {numbers[0]} + {numbers[1]} = {result}"
                        
                        elif any(word in problem_lower for word in ['subtract', 'minus', '-', 'difference']):
                            result = float(numbers[0]) - float(numbers[1])
                            return f"Problem: {problem}\nSolution: {numbers[0]} - {numbers[1]} = {result}"
                        
                        elif any(word in problem_lower for word in ['multiply', 'times', '*', 'product']):
                            result = float(numbers[0]) * float(numbers[1])
                            return f"Problem: {problem}\nSolution: {numbers[0]} × {numbers[1]} = {result}"
                        
                        elif any(word in problem_lower for word in ['divide', '/', 'quotient']):
                            result = float(numbers[0]) / float(numbers[1])
                            return f"Problem: {problem}\nSolution: {numbers[0]} ÷ {numbers[1]} = {result}"
                        
                        else:
                            return f"Problem: {problem}\nFound numbers: {numbers}\nPlease specify the operation (add, subtract, multiply, divide)."
                            
                    except Exception as e:
                        logger.error(f"Math solver error: {e}")
                        return f"Math solver error: {str(e)}"
                
                self.tools.append(math_solver)
                logger.info("Math solver tool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize math solver tool: {e}")
            
            # Document Q&A Tool
            try:
                from pathlib import Path
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
                
                # Initialize embeddings and vector store
                embeddings = None
                vectorstore = None
                documents_path = Path(config.DOCUMENTS_PATH)
                supported_extensions = {
                    '.pdf': PyPDFLoader,
                    '.docx': Docx2txtLoader,
                    '.txt': TextLoader
                }
                
                try:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
                    logger.info("Embeddings initialized")
                    
                    # Check if vector store already exists
                    vectorstore_path = Path("chroma_db")
                    if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
                        vectorstore = Chroma(
                            persist_directory=str(vectorstore_path),
                            embedding_function=embeddings
                        )
                        logger.info("Loaded existing vector store")
                    else:
                        # Create new vector store and load documents
                        vectorstore = Chroma(
                            persist_directory=str(vectorstore_path),
                            embedding_function=embeddings
                        )
                        
                        # Load documents if documents directory exists
                        if documents_path.exists():
                            documents = []
                            
                            for file_path in documents_path.rglob('*'):
                                if file_path.suffix.lower() in supported_extensions:
                                    try:
                                        loader_class = supported_extensions[file_path.suffix.lower()]
                                        loader = loader_class(str(file_path))
                                        docs = loader.load()
                                        
                                        # Add source metadata
                                        for doc in docs:
                                            doc.metadata['source'] = file_path.name
                                        
                                        documents.extend(docs)
                                        logger.info(f"Loaded document: {file_path.name}")
                                        
                                    except Exception as e:
                                        logger.error(f"Error loading {file_path}: {e}")
                            
                            if documents:
                                # Split documents into chunks
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000,
                                    chunk_overlap=200
                                )
                                chunks = text_splitter.split_documents(documents)
                                
                                # Add to vector store
                                vectorstore.add_documents(chunks)
                                logger.info(f"Added {len(chunks)} chunks to vector store")
                        
                        logger.info("Created new vector store")
                        
                except Exception as e:
                    logger.error(f"Error initializing embeddings/vectorstore: {e}")
                
                @tool
                def document_qa(query: str) -> str:
                    """Search and query local documents for information"""
                    try:
                        # Fallback: if vector store not ready (e.g., HF auth 401), do simple keyword search
                        if not vectorstore:
                            if not documents_path.exists():
                                return "No documents directory found."
                            # Load raw texts
                            corpus = []  # list of (source, text)
                            for file_path in documents_path.rglob('*'):
                                if file_path.suffix.lower() in supported_extensions:
                                    try:
                                        loader_class = supported_extensions[file_path.suffix.lower()]
                                        loader = loader_class(str(file_path))
                                        docs = loader.load()
                                        for d in docs:
                                            corpus.append((file_path.name, d.page_content))
                                    except Exception as e:
                                        logger.error(f"Error loading {file_path}: {e}")
                            if not corpus:
                                return "No supported documents found to search."
                            # Simple keyword scoring
                            import re
                            tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
                            scored = []
                            for source, text in corpus:
                                tl = text.lower()
                                score = sum(1 for t in tokens if t in tl)
                                if score > 0:
                                    scored.append((score, source, text))
                            if not scored:
                                return "No relevant documents found for your query."
                            scored.sort(reverse=True, key=lambda x: x[0])
                            top = scored[:2]
                            snippets = []
                            for score, source, text in top:
                                snippet = text[:400].replace('\n', ' ')
                                snippets.append(f"{source} (score {score}): {snippet}...")
                            return "Document search (keyword fallback):\n\n" + "\n\n".join(snippets)
                        
                        # Search for relevant documents
                        results = vectorstore.similarity_search(query, k=2)
                        
                        if not results:
                            return "No relevant documents found for your query."
                        
                        # Format results
                        formatted_results = []
                        for i, doc in enumerate(results, 1):
                            formatted_results.append(f"{i}. {doc.page_content[:300]}...\nSource: {doc.metadata.get('source', 'Unknown')}")
                        
                        return "Document search results:\n\n" + "\n\n".join(formatted_results)
                        
                    except Exception as e:
                        logger.error(f"Document Q&A error: {e}")
                        return f"Document search error: {str(e)}"
                
                self.tools.append(document_qa)
                logger.info("Document Q&A tool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize document Q&A tool: {e}")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
    
    def _initialize_agent(self):
        """Initialize the LangChain agent"""
        try:
            if not self.llm:
                logger.error("Cannot initialize agent - no LLM available")
                return
            
            # Check if we have any tools
            if not self.tools:
                logger.error("Cannot initialize agent - no tools available")
                return
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=5
            )
            
            logger.info(f"Initializing agent with {len(self.tools)} tools")
            
            # Create Chain of Thought prompt
            prompt_template = """You are an intelligent AI assistant with Chain of Thought reasoning capabilities.

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do step by step
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (just the input value, not the parameter name)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: When calling tools, use this format:
- For calculator: Action: calculator, Action Input: "2 + 2"
- For web_search: Action: web_search, Action Input: "capital of France"
- For math_solver: Action: math_solver, Action Input: "Janet's dogs eat 2 pounds each day for 7 days"
- For document_qa: Action: document_qa, Action Input: "information about AI"

When reasoning:
1. Break down complex queries into logical steps
2. Use the most appropriate tool for each step
3. Think through each step carefully
4. Combine information from multiple tools if needed
5. Provide clear, step-by-step reasoning

Question: {input}
Thought: {agent_scratchpad}"""

            prompt = PromptTemplate(
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                template=prompt_template
            )
            
            # Create the agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create the agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=config.AGENT_CONFIG["verbose"],
                max_iterations=config.AGENT_CONFIG["max_iterations"],
                handle_parsing_errors=True
            )
            
            logger.info("LangChain agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            self.agent_executor = None
    
    def process_query(self, user_query: str) -> AgentResponse:
        """
        Process user query using Chain of Thought reasoning
        
        Args:
            user_query: The user's input query
            
        Returns:
            AgentResponse with results and reasoning steps
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {user_query}")
        
        try:
            if not self.agent_executor:
                raise Exception("Agent not initialized")

            # Lightweight routing for common queries to avoid parser loops
            lower_q = user_query.lower().strip()

            # Helper to find a tool by name
            def find_tool(tool_name: str):
                for t in self.tools:
                    if getattr(t, "name", "") == tool_name:
                        return t
                return None

            # Intent detection
            has_digits = any(ch.isdigit() for ch in lower_q)
            is_math_like = any(op in lower_q for op in ["+","-","*","/"," times "," plus "," minus "," divided "]) and has_digits
            is_word_problem = any(k in lower_q for k in ["how many","total","sum","difference","product","remainder"]) and has_digits
            needs_docs = any(k in lower_q for k in ["document", "pdf", "docx", "file", "in the documents", "according to the documents", "based on the documents", "my cv"])
            needs_web = (any(k in lower_q for k in ["search","latest","current","news","online","weather","temperature"]) or lower_q.endswith("?"))

            # Build an execution plan that can include multiple tools
            plan = []  # list of (tool_name, input)
            if is_word_problem:
                plan.append(("math_solver", user_query))
            elif is_math_like:
                plan.append(("calculator", user_query))
            if needs_web:
                plan.append(("web_search", user_query))
            if needs_docs:
                plan.append(("document_qa", user_query))

            if plan:
                tasks: List[Task] = []
                reasoning_steps: List[str] = []
                for tool_name, tool_input in plan:
                    tool_obj = find_tool(tool_name)
                    if not tool_obj:
                        continue
                    logger.info(f"Routing directly to {tool_name} tool")
                    observation = tool_obj.invoke(tool_input)
                    tasks.append(Task(description=tool_name, tool_name=tool_name, parameters={"input": tool_input}, status="completed", result=observation))
                    reasoning_steps.append(f"Used {tool_name} on '{tool_input}' -> {str(observation)[:120]}")

                # Compose a final answer from observations
                composed_answer_parts = []
                for t in tasks:
                    if isinstance(t.result, str):
                        composed_answer_parts.append(t.result)
                    else:
                        composed_answer_parts.append(str(t.result))
                final_answer = "\n\n".join(composed_answer_parts) if composed_answer_parts else "No result."
                execution_time = time.time() - start_time
                return AgentResponse(
                    query=user_query,
                    tasks=tasks,
                    final_answer=final_answer,
                    success=True,
                    execution_time=execution_time,
                    reasoning_steps=reasoning_steps
                )

            # Fallback to full agent if routing not triggered
            result = self.agent_executor.invoke({"input": user_query})
            
            # Extract information from result
            final_answer = result.get("output", "No response generated")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Convert intermediate steps to tasks
            tasks = []
            reasoning_steps = []
            
            for i, (action, observation) in enumerate(intermediate_steps):
                task = Task(
                    description=f"Step {i+1}: {action.tool}",
                    tool_name=action.tool,
                    parameters=action.tool_input,
                    status="completed",
                    result=observation
                )
                tasks.append(task)
                
                reasoning_steps.append(f"Step {i+1}: Used {action.tool} with input '{action.tool_input}' -> {observation[:100]}...")
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                query=user_query,
                tasks=tasks,
                final_answer=final_answer,
                success=True,
                execution_time=execution_time,
                reasoning_steps=reasoning_steps
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            execution_time = time.time() - start_time
            
            return AgentResponse(
                query=user_query,
                tasks=[],
                final_answer=f"Error: {str(e)}",
                success=False,
                execution_time=execution_time,
                reasoning_steps=[]
            )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]
    
    def get_tool_info(self) -> Dict[str, str]:
        """Get information about available tools"""
        return {tool.name: tool.description for tool in self.tools}
