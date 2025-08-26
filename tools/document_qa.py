import os
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import logging

logger = logging.getLogger(__name__)

class DocumentQAInput(BaseModel):
    """Input schema for document QA tool"""
    query: str = Field(description="The query to search for in documents")

class DocumentQATool(BaseTool):
    """Simple document Q&A tool using RAG"""
    
    def __init__(self, documents_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize document Q&A tool"""
        self.documents_path = Path(documents_path)
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstore = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_vectorstore()
        super().__init__(
            name="document_qa",
            description="Search and query local documents for information",
            args_schema=DocumentQAInput
        )
    
    def _run(self, query: str) -> str:
        """Query documents for information"""
        try:
            if not self.vectorstore:
                return "Document search not available - vector store not initialized."
            
            # Search for relevant documents
            results = self.vectorstore.similarity_search(query, k=2)
            
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
    
    def _initialize_embeddings(self):
        """Initialize embeddings model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            logger.info(f"Embeddings initialized: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            self.embeddings = None
    
    def _initialize_vectorstore(self):
        """Initialize vector store"""
        try:
            if not self.embeddings:
                logger.error("Cannot initialize vector store - no embeddings available")
                return
            
            # Check if vector store already exists
            vectorstore_path = Path("chroma_db")
            if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
                self.vectorstore = Chroma(
                    persist_directory=str(vectorstore_path),
                    embedding_function=self.embeddings
                )
                logger.info("Loaded existing vector store")
            else:
                # Create new vector store and load documents
                self.vectorstore = Chroma(
                    persist_directory=str(vectorstore_path),
                    embedding_function=self.embeddings
                )
                self._load_documents()
                logger.info("Created new vector store and loaded documents")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vectorstore = None
    
    def _load_documents(self):
        """Load documents from the documents directory"""
        try:
            if not self.documents_path.exists():
                logger.warning(f"Documents path does not exist: {self.documents_path}")
                return
            
            documents = []
            
            # Supported file extensions
            supported_extensions = {
                '.pdf': PyPDFLoader,
                '.docx': Docx2txtLoader,
                '.txt': TextLoader
            }
            
            # Load documents
            for file_path in self.documents_path.rglob('*'):
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
                self.vectorstore.add_documents(chunks)
                logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
