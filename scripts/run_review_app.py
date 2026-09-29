#!/usr/bin/env python3
"""Convenience script to start the Human-in-the-loop Review Web App."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthesis.review_app.server import run_server

def main():
    parser = argparse.ArgumentParser(description="Start the Human Review Web App")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--input", type=str, default="data/synthesis/candidates_with_suggested_rubrics.jsonl", help="Input candidates path")
    parser.add_argument("--store", type=str, default="data/synthesis/human_review_store.json", help="Persistent store path")
    args = parser.parse_args()

    run_server(
        port=args.port,
        input_path=Path(args.input),
        store_path=Path(args.store),
    )

if __name__ == "__main__":
    main()
