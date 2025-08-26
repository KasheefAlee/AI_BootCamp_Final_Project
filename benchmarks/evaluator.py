import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import time
from pathlib import Path
import re

from agentcontroller import AgentController

logger = logging.getLogger(__name__)

class BenchmarkEvaluator:
    def __init__(self, agent_controller: AgentController):
        """
        Initialize benchmark evaluator
        
        Args:
            agent_controller: The agent controller to evaluate
        """
        self.agent = agent_controller
        self.benchmarks_path = Path("benchmarks")
        self.benchmarks_path.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            'lama': {'correct': 0, 'total': 0, 'results': []},
            'gsm8k': {'correct': 0, 'total': 0, 'results': []},
            'overall': {'accuracy': 0.0, 'avg_time': 0.0}
        }
    
    def evaluate_lama_benchmark(self, lama_file: str = None, sample_size: int = 100) -> Dict[str, Any]:
        """
        Evaluate agent on LAMA benchmark for factual accuracy
        
        Args:
            lama_file: Path to LAMA benchmark file
            sample_size: Number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        logger.info("Starting LAMA benchmark evaluation...")
        
        # Load or create LAMA dataset
        lama_data = self._load_or_create_lama_data(lama_file, sample_size)
        
        correct_answers = 0
        total_time = 0
        results = []
        
        for i, item in enumerate(lama_data):
            try:
                logger.info(f"LAMA evaluation {i+1}/{len(lama_data)}: {item['query'][:50]}...")
                
                start_time = time.time()
                response = self.agent.process_query(item['query'])
                execution_time = time.time() - start_time
                
                total_time += execution_time
                
                # Check if answer is correct
                is_correct = self._check_lama_answer(
                    response.final_answer, 
                    item['expected_answer']
                )
                
                if is_correct:
                    correct_answers += 1
                
                result = {
                    'query': item['query'],
                    'expected_answer': item['expected_answer'],
                    'agent_answer': response.final_answer,
                    'is_correct': is_correct,
                    'execution_time': execution_time,
                    'success': response.success,
                    'tools_used': [task.tool_name for task in response.tasks]
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in LAMA evaluation {i}: {e}")
                results.append({
                    'query': item['query'],
                    'expected_answer': item['expected_answer'],
                    'agent_answer': f"Error: {str(e)}",
                    'is_correct': False,
                    'execution_time': 0,
                    'success': False,
                    'tools_used': []
                })
        
        accuracy = correct_answers / len(lama_data) if lama_data else 0
        avg_time = total_time / len(lama_data) if lama_data else 0
        
        self.results['lama'] = {
            'correct': correct_answers,
            'total': len(lama_data),
            'accuracy': accuracy,
            'avg_time': avg_time,
            'results': results
        }
        
        logger.info(f"LAMA benchmark completed: {accuracy:.2%} accuracy")
        return self.results['lama']
    
    def evaluate_gsm8k_benchmark(self, gsm8k_file: str = None, sample_size: int = 100) -> Dict[str, Any]:
        """
        Evaluate agent on GSM8K benchmark for mathematical reasoning
        
        Args:
            gsm8k_file: Path to GSM8K benchmark file
            sample_size: Number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        logger.info("Starting GSM8K benchmark evaluation...")
        
        # Load or create GSM8K dataset
        gsm8k_data = self._load_or_create_gsm8k_data(gsm8k_file, sample_size)
        
        correct_answers = 0
        total_time = 0
        results = []
        
        for i, item in enumerate(gsm8k_data):
            try:
                logger.info(f"GSM8K evaluation {i+1}/{len(gsm8k_data)}: {item['problem'][:50]}...")
                
                start_time = time.time()
                response = self.agent.process_query(item['problem'])
                execution_time = time.time() - start_time
                
                total_time += execution_time
                
                # Extract numerical answer from agent response
                agent_answer = self._extract_numerical_answer(response.final_answer)
                
                # Check if answer is correct
                is_correct = self._check_numerical_answer(agent_answer, item['answer'])
                
                if is_correct:
                    correct_answers += 1
                
                result = {
                    'problem': item['problem'],
                    'expected_answer': item['answer'],
                    'agent_answer': agent_answer,
                    'full_response': response.final_answer,
                    'is_correct': is_correct,
                    'execution_time': execution_time,
                    'success': response.success,
                    'tools_used': [task.tool_name for task in response.tasks]
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in GSM8K evaluation {i}: {e}")
                results.append({
                    'problem': item['problem'],
                    'expected_answer': item['answer'],
                    'agent_answer': None,
                    'full_response': f"Error: {str(e)}",
                    'is_correct': False,
                    'execution_time': 0,
                    'success': False,
                    'tools_used': []
                })
        
        accuracy = correct_answers / len(gsm8k_data) if gsm8k_data else 0
        avg_time = total_time / len(gsm8k_data) if gsm8k_data else 0
        
        self.results['gsm8k'] = {
            'correct': correct_answers,
            'total': len(gsm8k_data),
            'accuracy': accuracy,
            'avg_time': avg_time,
            'results': results
        }
        
        logger.info(f"GSM8K benchmark completed: {accuracy:.2%} accuracy")
        return self.results['gsm8k']
    
    def run_full_evaluation(self, lama_samples: int = 50, gsm8k_samples: int = 50) -> Dict[str, Any]:
        """
        Run complete evaluation on both benchmarks
        
        Args:
            lama_samples: Number of LAMA samples
            gsm8k_samples: Number of GSM8K samples
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting full benchmark evaluation...")
        
        # Run LAMA evaluation
        lama_results = self.evaluate_lama_benchmark(sample_size=lama_samples)
        
        # Run GSM8K evaluation
        gsm8k_results = self.evaluate_gsm8k_benchmark(sample_size=gsm8k_samples)
        
        # Calculate overall metrics
        total_correct = lama_results['correct'] + gsm8k_results['correct']
        total_questions = lama_results['total'] + gsm8k_results['total']
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        overall_time = (
            lama_results['avg_time'] * lama_results['total'] + 
            gsm8k_results['avg_time'] * gsm8k_results['total']
        ) / total_questions if total_questions > 0 else 0
        
        self.results['overall'] = {
            'accuracy': overall_accuracy,
            'avg_time': overall_time,
            'total_questions': total_questions,
            'total_correct': total_correct
        }
        
        # Save results
        self._save_results()
        
        logger.info(f"Full evaluation completed: {overall_accuracy:.2%} overall accuracy")
        return self.results
    
    def _load_or_create_lama_data(self, file_path: str = None, sample_size: int = 100) -> List[Dict[str, Any]]:
        """Load or create LAMA benchmark data"""
        if file_path and Path(file_path).exists():
            # Load existing LAMA data
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data[:sample_size]
        else:
            # Create sample LAMA-style questions
            lama_samples = [
                {
                    "query": "What is the capital of France?",
                    "expected_answer": "Paris",
                    "relation": "capital"
                },
                {
                    "query": "Who founded Microsoft?",
                    "expected_answer": "Bill Gates",
                    "relation": "founder"
                },
                {
                    "query": "What is the largest planet in our solar system?",
                    "expected_answer": "Jupiter",
                    "relation": "largest"
                },
                {
                    "query": "When was the Declaration of Independence signed?",
                    "expected_answer": "1776",
                    "relation": "date"
                },
                {
                    "query": "What is the chemical symbol for gold?",
                    "expected_answer": "Au",
                    "relation": "symbol"
                },
                {
                    "query": "Who wrote Romeo and Juliet?",
                    "expected_answer": "Shakespeare",
                    "relation": "author"
                },
                {
                    "query": "What is the speed of light?",
                    "expected_answer": "299792458",
                    "relation": "constant"
                },
                {
                    "query": "What is the smallest country in the world?",
                    "expected_answer": "Vatican",
                    "relation": "smallest"
                }
            ]
            
            # Extend with more samples if needed
            while len(lama_samples) < sample_size:
                lama_samples.extend(lama_samples[:min(len(lama_samples), sample_size - len(lama_samples))])
            
            return lama_samples[:sample_size]
    
    def _load_or_create_gsm8k_data(self, file_path: str = None, sample_size: int = 100) -> List[Dict[str, Any]]:
        """Load or create GSM8K benchmark data"""
        if file_path and Path(file_path).exists():
            # Load existing GSM8K data
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data[:sample_size]
        else:
            # Create sample GSM8K-style problems
            gsm8k_samples = [
                {
                    "problem": "Janet has 3 cards with 7 cats on each card. How many cats does she have in total?",
                    "answer": 21
                },
                {
                    "problem": "A bakery sold 23 cupcakes in the morning and 14 cupcakes in the afternoon. How many cupcakes did they sell in total?",
                    "answer": 37
                },
                {
                    "problem": "Tom has 45 marbles. He gives 12 marbles to his friend. How many marbles does Tom have left?",
                    "answer": 33
                },
                {
                    "problem": "A school has 8 classrooms with 25 students in each classroom. How many students are there in total?",
                    "answer": 200
                },
                {
                    "problem": "Sarah bought 4 packs of stickers. Each pack has 15 stickers. How many stickers does Sarah have?",
                    "answer": 60
                },
                {
                    "problem": "A pizza is cut into 8 slices. If 3 people equally share the pizza, how many slices does each person get?",
                    "answer": 2.67
                },
                {
                    "problem": "John runs 5 miles every day for 7 days. How many miles does he run in total?",
                    "answer": 35
                },
                {
                    "problem": "A box contains 144 pencils. If pencils are packed in groups of 12, how many groups are there?",
                    "answer": 12
                }
            ]
            
            # Extend with more samples if needed
            while len(gsm8k_samples) < sample_size:
                gsm8k_samples.extend(gsm8k_samples[:min(len(gsm8k_samples), sample_size - len(gsm8k_samples))])
            
            return gsm8k_samples[:sample_size]
    
    def _check_lama_answer(self, agent_answer: str, expected_answer: str) -> bool:
        """Check if LAMA answer is correct"""
        if not agent_answer or not expected_answer:
            return False
        
        agent_answer = agent_answer.lower().strip()
        expected_answer = expected_answer.lower().strip()
        
        # Direct match
        if expected_answer in agent_answer:
            return True
        
        # Handle common variations
        variations = {
            'united states': ['usa', 'us', 'america'],
            'united kingdom': ['uk', 'britain', 'england'],
            'william shakespeare': ['shakespeare'],
            'bill gates': ['gates', 'william gates'],
        }
        
        for canonical, variants in variations.items():
            if expected_answer == canonical:
                return any(variant in agent_answer for variant in variants)
            elif expected_answer in variants and canonical in agent_answer:
                return True
        
        return False
    
    def _extract_numerical_answer(self, text: str) -> float:
        """Extract numerical answer from text"""
        if not text:
            return None
        
        # Look for numbers in the text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            try:
                # Return the last number found (often the final answer)
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def _check_numerical_answer(self, agent_answer: float, expected_answer: float, tolerance: float = 0.01) -> bool:
        """Check if numerical answer is correct within tolerance"""
        if agent_answer is None or expected_answer is None:
            return False
        
        return abs(agent_answer - expected_answer) <= tolerance
    
    def _save_results(self):
        """Save evaluation results to file"""
        results_file = self.benchmarks_path / "evaluation_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        report = []
        report.append("=== AGENT EVALUATION REPORT ===\n")
        
        # LAMA Results
        if self.results['lama']['total'] > 0:
            lama = self.results['lama']
            report.append(f"LAMA Benchmark (Factual Accuracy):")
            report.append(f"  Correct: {lama['correct']}/{lama['total']}")
            report.append(f"  Accuracy: {lama['accuracy']:.2%}")
            report.append(f"  Avg Time: {lama['avg_time']:.2f}s")
            report.append("")
        
        # GSM8K Results
        if self.results['gsm8k']['total'] > 0:
            gsm8k = self.results['gsm8k']
            report.append(f"GSM8K Benchmark (Mathematical Reasoning):")
            report.append(f"  Correct: {gsm8k['correct']}/{gsm8k['total']}")
            report.append(f"  Accuracy: {gsm8k['accuracy']:.2%}")
            report.append(f"  Avg Time: {gsm8k['avg_time']:.2f}s")
            report.append("")
        
        # Overall Results
        overall = self.results['overall']
        if overall.get('total_questions', 0) > 0:
            report.append(f"Overall Performance:")
            report.append(f"  Total Questions: {overall['total_questions']}")
            report.append(f"  Total Correct: {overall['total_correct']}")
            report.append(f"  Overall Accuracy: {overall['accuracy']:.2%}")
            report.append(f"  Average Time: {overall['avg_time']:.2f}s")
        
        return "\n".join(report)