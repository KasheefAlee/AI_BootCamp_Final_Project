#!/usr/bin/env python3
"""
Agentic AI Bootcamp - Main CLI Interface
A 3-day bootcamp project for building intelligent agents with Chain of Thought reasoning.
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('agent.log')
        ]
    )

def cli_mode():
    """Run the agent in CLI mode"""
    from agent_controller import AgentController
    
    print("🤖 Agentic AI Bootcamp - CLI Mode")
    print("=" * 50)
    print("Initializing agent...")
    
    try:
        # Initialize agent
        agent = AgentController()
        
        if not agent.agent_executor:
            print("❌ Failed to initialize agent. Check logs for details.")
            return
        
        print("✅ Agent initialized successfully!")
        print(f"\nAvailable tools: {', '.join(agent.get_available_tools())}")
        
        print("\nEnter your queries (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n🤔 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("🤖 Processing...")
                
                # Process query
                response = agent.process_query(query)
                
                if response.success:
                    print(f"\n✅ Answer: {response.final_answer}")
                    
                    if response.reasoning_steps:
                        print(f"\n🧠 Reasoning Steps:")
                        for step in response.reasoning_steps:
                            print(f"  • {step}")
                    
                    print(f"\n⏱️  Execution time: {response.execution_time:.2f}s")
                else:
                    print(f"❌ Error: {response.final_answer}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to start CLI mode: {e}")
        logging.error(f"CLI mode error: {e}")

def evaluation_mode():
    """Run the evaluation framework"""
    from evaluator import Evaluator
    
    print("📊 Agentic AI Bootcamp - Evaluation Mode")
    print("=" * 50)
    print("Running LAMA and GSM8k benchmarks...")
    
    try:
        evaluator = Evaluator()
        results = evaluator.run_full_evaluation()
        evaluator.print_results(results)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logging.error(f"Evaluation error: {e}")

def demo_mode():
    """Run demo queries"""
    from agent_controller import AgentController
    
    print("🎯 Agentic AI Bootcamp - Demo Mode")
    print("=" * 50)
    
    demo_queries = [
        "What is the capital of Pakistan and what is 2 + 2?",
        "Janet's dogs eat 2 pounds of dog food each day. How many pounds do they eat in 7 days?",
        "What is the current weather in London?",
        "Who wrote Romeo and Juliet?"
    ]
    
    try:
        agent = AgentController()
        
        if not agent.agent_executor:
            print("❌ Failed to initialize agent for demo.")
            return
        
        print("✅ Agent initialized successfully!")
        print(f"Available tools: {', '.join(agent.get_available_tools())}")
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*60}")
            print(f"Demo {i}/{len(demo_queries)}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            try:
                response = agent.process_query(query)
                
                if response.success:
                    print(f"✅ Answer: {response.final_answer}")
                    
                    if response.reasoning_steps:
                        print(f"\n🧠 Reasoning Steps:")
                        for step in response.reasoning_steps:
                            print(f"  • {step}")
                    
                    print(f"⏱️  Time: {response.execution_time:.2f}s")
                else:
                    print(f"❌ Error: {response.final_answer}")
                
            except Exception as e:
                print(f"❌ Demo {i} failed: {e}")
            
            input("\nPress Enter to continue to next demo...")
        
        print("\n🎉 Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logging.error(f"Demo error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic AI Bootcamp - Intelligent Agent with Chain of Thought Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run Streamlit web app
  python run.py --cli              # Interactive CLI mode
  python run.py --evaluate         # Run LAMA and GSM8k benchmarks
  python run.py --demo             # Run demo queries
        """
    )
    
    parser.add_argument(
        '--cli', 
        action='store_true', 
        help='Run in interactive CLI mode'
    )
    
    parser.add_argument(
        '--evaluate', 
        action='store_true', 
        help='Run evaluation benchmarks (LAMA and GSM8k)'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true', 
        help='Run demo queries'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check if any mode is specified
    if not any([args.cli, args.evaluate, args.demo]):
        print("🤖 Agentic AI Bootcamp")
        print("=" * 50)
        print("Welcome to the 3-day AI bootcamp project!")
        print("\nStarting Streamlit web app...")
        print("The app will open in your browser at http://localhost:8501")
        
        # Run Streamlit app
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return
    
    # Run selected mode
    if args.cli:
        cli_mode()
    elif args.evaluate:
        evaluation_mode()
    elif args.demo:
        demo_mode()

if __name__ == "__main__":
    main()