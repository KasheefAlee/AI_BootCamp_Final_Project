import ast
import operator
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class CalculatorInput(BaseModel):
    """Input schema for calculator tool"""
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    """Simple calculator tool for mathematical expressions"""
    
    def __init__(self):
        """Initialize calculator tool"""
        self.safe_operators = {
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
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations on expressions",
            args_schema=CalculatorInput
        )
    
    def _run(self, expression: str) -> str:
        """Evaluate mathematical expression safely"""
        try:
            logger.info(f"Calculating: {expression}")
            
            # Parse and evaluate safely
            tree = ast.parse(expression, mode='eval')
            result = self._safe_eval(tree.body)
            
            return f"Result: {result}"
            
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return f"Calculation error: {str(e)}"
    
    def _safe_eval(self, node):
        """Safely evaluate AST node"""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.safe_operators[type(node.op)](self._safe_eval(node.left), self._safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self.safe_operators[type(node.op)](self._safe_eval(node.operand))
        else:
            raise ValueError(f"Unsafe operation: {type(node).__name__}")
