# Installation

## Prerequisites

- Python 3.8+
- macOS, Linux, or WSL on Windows

## Setup

```bash
# Clone the repo
cd gguf-workbench

# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

## Output Directory

All modified GGUF files are saved to the `transformed_models/` directory in the project root. This directory is automatically created if it doesn't exist and is ignored by git.