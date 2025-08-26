from tavily import TavilyClient
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class WebSearchInput(BaseModel):
    """Input schema for web search tool"""
    query: str = Field(description="The search query to look up on the web")

class WebSearchTool(BaseTool):
    """Simple web search tool using Tavily"""
    
    def __init__(self, api_key: str):
        """Initialize web search tool"""
        self.client = TavilyClient(api_key=api_key)
        super().__init__(
            name="web_search",
            description="Search the web for current information and facts",
            args_schema=WebSearchInput
        )
    
    def _run(self, query: str) -> str:
        """Search the web using Tavily"""
        try:
            logger.info(f"Searching web for: {query}")
            
            response = self.client.search(
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
