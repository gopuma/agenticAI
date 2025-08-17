#!/usr/bin/env python3
"""
Launch script for the Advanced AI Agent Platform
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        import streamlit
        import boto3
        import google.generativeai
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_setup():
    """Check environment setup"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Copy .env.example to .env and configure your keys.")
    
    # Check AWS credentials
    if not (os.getenv("AWS_ACCESS_KEY_ID") or os.path.exists(os.path.expanduser("~/.aws/credentials"))):
        print("⚠️  AWS credentials not configured. Run 'aws configure' or set environment variables.")
    
    # Check Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not found in environment. Set it in .env or as environment variable.")

def main():
    """Main function to launch the application"""
    print("🚀 Launching Advanced AI Agent Platform...")
    
    if not check_requirements():
        sys.exit(1)
    
    check_env_setup()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")

if __name__ == "__main__":
    main()
