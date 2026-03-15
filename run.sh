#!/bin/bash
# Run from project root: bash run.sh
cd "$(dirname "$0")"
.venv/bin/streamlit run src/ui/app.py
