# based on https://github.com/paquino11/chatpdf-rag-deepseek-r1
import os, sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application."""
    app_path = Path("chatpdf_rag_deepseek/app/page.py")
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    