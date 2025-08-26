import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from agent_controller import AgentController
import config

logger = logging.getLogger(__name__)

class Evaluator:
    """Simple evaluation framework for LAMA and GSM8k benchmarks"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.agent = AgentController()
        self.results = {}
    
    def evaluate_lama(self, data_path: str = None) -> Dict[str, Any]:
        """
        Evaluate factual accuracy using LAMA benchmark
        
        Args:
            data_path: Path to LAMA benchmark data
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting LAMA evaluation...")
        
        # Sample LAMA questions for testing
        lama_questions = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "category": "geography"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare",
                "category": "literature"
            },
            {
                "question": "What is the largest planet in our solar system?",
                "answer": "Jupiter",
                "category": "science"
            }
        ]
        
        results = {
            "total_questions": len(lama_questions),
            "correct_answers": 0,
            "accuracy": 0.0,
            "responses": []
        }
        
        for i, question in enumerate(lama_questions):
            logger.info(f"Evaluating LAMA question {i+1}/{len(lama_questions)}: {question['question']}")
            
            try:
                # Process query with agent
                response = self.agent.process_query(question['question'])
                
                # Simple answer extraction (can be enhanced)
                agent_answer = response.final_answer.lower()
                correct_answer = question['answer'].lower()
                
                # Check if correct answer is in agent's response
                is_correct = correct_answer in agent_answer
                
                if is_correct:
                    results["correct_answers"] += 1
                
                results["responses"].append({
                    "question": question['question'],
                    "expected_answer": question['answer'],
                    "agent_answer": response.final_answer,
                    "is_correct": is_correct,
                    "execution_time": response.execution_time
                })
                
            except Exception as e:
                logger.error(f"Error evaluating question {i+1}: {e}")
                results["responses"].append({
                    "question": question['question'],
                    "expected_answer": question['answer'],
                    "agent_answer": f"Error: {str(e)}",
                    "is_correct": False,
                    "execution_time": 0
                })
        
        # Calculate accuracy
        results["accuracy"] = results["correct_answers"] / results["total_questions"]
        
        logger.info(f"LAMA evaluation completed. Accuracy: {results['accuracy']:.2%}")
        return results
    
    def evaluate_gsm8k(self, data_path: str = None) -> Dict[str, Any]:
        """
        Evaluate mathematical reasoning using GSM8k benchmark
        
        Args:
            data_path: Path to GSM8k benchmark data
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting GSM8k evaluation...")
        
        # Sample GSM8k questions for testing
        gsm8k_questions = [
            {
                "question": "Janet's dogs eat 2 pounds of dog food each day. How many pounds of dog food do her dogs eat in 7 days?",
                "answer": "14",
                "category": "multiplication"
            },
            {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "answer": "6",
                "category": "subtraction"
            },
            {
                "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "answer": "39",
                "category": "addition_subtraction"
            }
        ]
        
        results = {
            "total_questions": len(gsm8k_questions),
            "correct_answers": 0,
            "accuracy": 0.0,
            "responses": []
        }
        
        for i, question in enumerate(gsm8k_questions):
            logger.info(f"Evaluating GSM8k question {i+1}/{len(gsm8k_questions)}: {question['question']}")
            
            try:
                # Process query with agent
                response = self.agent.process_query(question['question'])
                
                # Extract numerical answer from agent response
                agent_answer = self._extract_number(response.final_answer)
                correct_answer = int(question['answer'])
                
                is_correct = agent_answer == correct_answer
                
                if is_correct:
                    results["correct_answers"] += 1
                
                results["responses"].append({
                    "question": question['question'],
                    "expected_answer": question['answer'],
                    "agent_answer": str(agent_answer) if agent_answer is not None else "No number found",
                    "is_correct": is_correct,
                    "execution_time": response.execution_time,
                    "reasoning_steps": response.reasoning_steps
                })
                
            except Exception as e:
                logger.error(f"Error evaluating question {i+1}: {e}")
                results["responses"].append({
                    "question": question['question'],
                    "expected_answer": question['answer'],
                    "agent_answer": f"Error: {str(e)}",
                    "is_correct": False,
                    "execution_time": 0,
                    "reasoning_steps": []
                })
        
        # Calculate accuracy
        results["accuracy"] = results["correct_answers"] / results["total_questions"]
        
        logger.info(f"GSM8k evaluation completed. Accuracy: {results['accuracy']:.2%}")
        return results
    
    def _extract_number(self, text: str) -> int:
        """Extract the first number from text"""
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else None
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run both LAMA and GSM8k evaluations"""
        logger.info("Starting full evaluation...")
        
        results = {
            "lama": self.evaluate_lama(),
            "gsm8k": self.evaluate_gsm8k(),
            "summary": {}
        }
        
        # Calculate summary statistics
        lama_accuracy = results["lama"]["accuracy"]
        gsm8k_accuracy = results["gsm8k"]["accuracy"]
        
        results["summary"] = {
            "overall_accuracy": (lama_accuracy + gsm8k_accuracy) / 2,
            "factual_accuracy": lama_accuracy,
            "reasoning_accuracy": gsm8k_accuracy,
            "total_questions": results["lama"]["total_questions"] + results["gsm8k"]["total_questions"]
        }
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Full evaluation completed. Overall accuracy: {results['summary']['overall_accuracy']:.2%}")
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        try:
            output_path = Path(config.BENCHMARKS_PATH) / "evaluation_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # LAMA Results
        print(f"\n📚 LAMA Benchmark (Factual Accuracy)")
        print(f"Accuracy: {results['lama']['accuracy']:.2%}")
        print(f"Correct: {results['lama']['correct_answers']}/{results['lama']['total_questions']}")
        
        # GSM8k Results
        print(f"\n🧮 GSM8k Benchmark (Mathematical Reasoning)")
        print(f"Accuracy: {results['gsm8k']['accuracy']:.2%}")
        print(f"Correct: {results['gsm8k']['correct_answers']}/{results['gsm8k']['total_questions']}")
        
        # Summary
        print(f"\n📊 SUMMARY")
        print(f"Overall Accuracy: {results['summary']['overall_accuracy']:.2%}")
        print(f"Total Questions: {results['summary']['total_questions']}")
        
        print("\n" + "="*50)
