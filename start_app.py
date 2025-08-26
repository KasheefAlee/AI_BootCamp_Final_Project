#!/usr/bin/env python3
"""
Quick startup script for the Agentic AI Bootcamp Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Start the Streamlit app"""
    print("🤖 Agentic AI Bootcamp")
    print("=" * 50)
    print("Starting Streamlit web app...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("=" * 50)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
