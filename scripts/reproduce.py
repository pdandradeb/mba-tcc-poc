#!/usr/bin/env python3
"""Verify the saved study inputs and regenerate its tables without model calls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/20260926T183838Z-c8c5b03b/results.json"
REVIEWS = "output/revisao/avaliacao-humana-simplificada-2026-09-27.json"
CLARIFICATION = "output/revisao/esclarecimento-avaliacao-humana-2026-09-28.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(entries):
    for name, expected in entries.items():
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Arquivo ausente ou hash divergente: {name}")


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    check_hashes(manifest["files_sha256"])
    run = json.loads((ROOT / RUN).read_text())
    reviews = json.loads((ROOT / REVIEWS).read_text())
    clarification = json.loads((ROOT / CLARIFICATION).read_text())
    cases = [json.loads(line) for line in (ROOT / "data/cases_reviewed.jsonl").read_text().splitlines()]
    assert len(cases) == 100 and all(case["synthetic"] for case in cases)
    assert len({case["id"] for case in cases}) == 100
    assert len(run["records"]) == len(reviews["records"]) == 400
    assert set(run["manifest"]["case_ids"]) == {case["id"] for case in cases}
    assert reviews["inputs"]["results_sha256"] == digest(ROOT / RUN)
    assert reviews["inputs"]["cases_sha256"] == digest(ROOT / "data/cases_reviewed.jsonl")
    assert reviews["inputs"]["corpus_sha256"] == digest(ROOT / "data/public_corpus.json")
    assert clarification["review_export_sha256"] == digest(ROOT / REVIEWS)
    assert set((row["case_id"], row["method"]) for row in run["records"]) == {
        (case["id"], method)
        for case in cases
        for method in ("heuristic", "fts", "neuro_symbolic", "vector")
    }
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, "latex/generate_experiment_tables.py", "--poc", ".", "--results", RUN],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    subprocess.run([sys.executable, "latex/generate_human_review_tables.py"], cwd=ROOT, env=environment, check=True)
    check_hashes(manifest["expected_outputs_sha256"])
    print(f"Pacote conferido: 100 casos, 400 registros, 400 julgamentos e {len(manifest['expected_outputs_sha256'])} saídas idênticas.")


if __name__ == "__main__":
    main()
