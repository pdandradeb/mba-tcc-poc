#!/usr/bin/env python3
"""Run simplified human review without executing models."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.response_review.server import main
from src.quick_response_review.store import QuickResponseStore

if __name__ == '__main__':
    main(store_class=QuickResponseStore, static=ROOT/'src/quick_response_review/static',
         default_port=8083, default_store='local_reviews/quick_responses.json',
         export_filename='avaliacao-rapida.json')
