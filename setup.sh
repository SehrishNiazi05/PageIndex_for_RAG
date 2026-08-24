#!/usr/bin/env bash
# One-time setup for the dental RAG project on macOS.
# Usage: bash setup.sh

set -e

echo "== Checking Homebrew =="
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Install it first: https://brew.sh"
    exit 1
fi

echo "== Installing poppler (for PDF text extraction via pdftotext) =="
brew install poppler || true

echo "== Creating Python virtual environment =="
python3 -m venv venv
source venv/bin/activate

echo "== Cloning PageIndex (self-hosted tree indexing) =="
if [ ! -d "PageIndex" ]; then
    git clone https://github.com/VectifyAI/PageIndex.git
fi

echo "== Installing PageIndex's own requirements =="
pip install --upgrade pip
pip install -r PageIndex/requirements.txt

echo "== Installing this project's requirements =="
pip install openai python-dotenv tqdm

echo "== Setting up folders =="
mkdir -p textbooks trees

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example -- edit it and add your DEEPSEEK_API_KEY."
fi

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. Put your 9 textbook PDFs into ./textbooks/"
echo "  2. Edit .env and add your DEEPSEEK_API_KEY"
echo "  3. source venv/bin/activate"
echo "  4. python3 batch_index.py"
echo "  5. python3 query_cli.py"
