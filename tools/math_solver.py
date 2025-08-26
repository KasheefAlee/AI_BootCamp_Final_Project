import re
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class MathSolverInput(BaseModel):
    """Input schema for math solver tool"""
    problem: str = Field(description="Math word problem to solve")

class MathSolverTool(BaseTool):
    """Simple math solver tool for word problems"""
    
    def __init__(self):
        """Initialize math solver tool"""
        super().__init__(
            name="math_solver",
            description="Solve math word problems step by step",
            args_schema=MathSolverInput
        )
    
    def _run(self, problem: str) -> str:
        """Solve math word problem"""
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
