#!/usr/bin/env bash
# Sets up a virtual environment (first run only) and starts AI Insight Lab.
set -e
cd "$(dirname "$0")/backend"

if [ ! -d "venv" ]; then
  echo "Setting up virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  pip install -q -r requirements.txt
else
  source venv/bin/activate
fi

echo "Starting AI Insight Lab at http://127.0.0.1:8000"
uvicorn app.main:app --reload --port 8000
