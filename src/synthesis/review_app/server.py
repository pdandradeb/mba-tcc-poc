"""Local lightweight web server for human-in-the-loop review of candidate benchmark cases."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_INPUT_PATH = Path("data/synthesis/candidates_with_suggested_rubrics.jsonl")
DEFAULT_STORE_PATH = Path("data/synthesis/human_review_store.json")
DEFAULT_EXPORT_PATH = Path("data/cases_reviewed.jsonl")
STATIC_DIR = Path(__file__).parent / "static"


class ReviewStore:
    def __init__(self, input_path: Path = DEFAULT_INPUT_PATH, store_path: Path = DEFAULT_STORE_PATH):
        self.input_path = Path(input_path)
        self.store_path = Path(store_path)
        self.cases: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.store_path.exists():
            with open(self.store_path, "r", encoding="utf-8") as f:
                self.cases = json.load(f)
            print(f"[review-store] Loaded {len(self.cases)} cases from persistent store {self.store_path}", file=sys.stderr)
            return

        if self.input_path.exists():
            with open(self.input_path, "r", encoding="utf-8") as f:
                self.cases = [json.loads(line) for line in f if line.strip()]
            print(f"[review-store] Initialized store with {len(self.cases)} cases from {self.input_path}", file=sys.stderr)
            self.save_to_disk()
            return

        # Empty fallback
        self.cases = []

    def save_to_disk(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.store_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.store_path)

    def get_summary_list(self) -> list[dict[str, Any]]:
        return [
            {
                "id": c["id"],
                "curated_id": c.get("curated_id", c["id"]),
                "question": c["question"],
                "category": c.get("category", "geral"),
                "review_status": c.get("review_status", "pending"),
                "source_model": c.get("source_model", "unknown"),
                "handoff_expected_reviewed": c.get("handoff_expected_reviewed"),
            }
            for c in self.cases
        ]

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        for c in self.cases:
            if c["id"] == case_id or c.get("curated_id") == case_id:
                return c
        return None

    def get_next_pending_id(self, current_id: str) -> str | None:
        idx = next((i for i, c in enumerate(self.cases) if c["id"] == current_id or c.get("curated_id") == current_id), -1)
        # First check subsequent items
        for c in self.cases[idx + 1 :]:
            if c.get("review_status") == "pending":
                return c["id"]
        # Check from start
        for c in self.cases[: max(0, idx)]:
            if c.get("review_status") == "pending":
                return c["id"]
        # If none pending, just return next case in list if available
        if idx >= 0 and idx + 1 < len(self.cases):
            return self.cases[idx + 1]["id"]
        return None

    def approve_with_suggestions(self, case_id: str, reviewer: str = "Pedro Saulo Brito") -> str | None:
        case = self.get_case(case_id)
        if not case:
            raise KeyError(f"Case {case_id} not found")

        case["expected_facts_reviewed"] = list(case.get("expected_facts_suggested", []))
        case["forbidden_claims_reviewed"] = list(case.get("forbidden_claims_suggested", []))
        case["handoff_expected_reviewed"] = bool(case.get("handoff_expected_suggested", False))
        case["handoff_reason_reviewed"] = str(case.get("handoff_reason_suggested", ""))
        case["grounding_chunk_ids_reviewed"] = list(case.get("grounding_chunk_ids_suggested", []))
        case["review_status"] = "approved"
        case["rejection_reason"] = ""
        case["reviewed_by"] = reviewer
        case["reviewed_at"] = datetime.now(timezone.utc).isoformat()

        self.save_to_disk()
        return self.get_next_pending_id(case_id)

    def save_custom_review(self, case_id: str, payload: dict[str, Any], reviewer: str = "Pedro Saulo Brito") -> str | None:
        case = self.get_case(case_id)
        if not case:
            raise KeyError(f"Case {case_id} not found")

        if "question" in payload:
            case["question"] = str(payload["question"]).strip()
        if "category" in payload:
            case["category"] = str(payload["category"]).strip()
        if "intent_type" in payload:
            case["intent_type"] = str(payload["intent_type"]).strip()

        case["expected_facts_reviewed"] = [str(f).strip() for f in payload.get("expected_facts_reviewed", []) if str(f).strip()]
        case["forbidden_claims_reviewed"] = [str(f).strip() for f in payload.get("forbidden_claims_reviewed", []) if str(f).strip()]

        handoff_val = payload.get("handoff_expected_reviewed")
        if handoff_val is not None:
            case["handoff_expected_reviewed"] = bool(handoff_val)
        case["handoff_reason_reviewed"] = str(payload.get("handoff_reason_reviewed", "")).strip()

        if "grounding_chunk_ids_reviewed" in payload:
            case["grounding_chunk_ids_reviewed"] = list(payload["grounding_chunk_ids_reviewed"])

        case["review_status"] = payload.get("review_status", "approved")
        case["rejection_reason"] = str(payload.get("rejection_reason", "")).strip()
        case["reviewed_by"] = reviewer
        case["reviewed_at"] = datetime.now(timezone.utc).isoformat()

        self.save_to_disk()
        return self.get_next_pending_id(case_id)

    def reject_case(self, case_id: str, reason: str, reviewer: str = "Pedro Saulo Brito") -> str | None:
        case = self.get_case(case_id)
        if not case:
            raise KeyError(f"Case {case_id} not found")

        case["review_status"] = "rejected"
        case["rejection_reason"] = reason or "Rejeitado pelo revisor"
        case["reviewed_by"] = reviewer
        case["reviewed_at"] = datetime.now(timezone.utc).isoformat()

        self.save_to_disk()
        return self.get_next_pending_id(case_id)

    def get_stats(self) -> dict[str, Any]:
        total = len(self.cases)
        counts = {"pending": 0, "approved": 0, "rejected": 0}
        categories = {}
        for c in self.cases:
            st = c.get("review_status", "pending")
            counts[st] = counts.get(st, 0) + 1
            cat = c.get("category", "geral")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total": total,
            "counts": counts,
            "categories": categories,
            "percent_completed": round(((counts["approved"] + counts["rejected"]) / total * 100), 1) if total else 0.0,
        }

    def export_canonical_dataset(self, export_path: Path = DEFAULT_EXPORT_PATH) -> tuple[int, str]:
        approved = [c for c in self.cases if c.get("review_status") == "approved"]
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)

        with open(export_file, "w", encoding="utf-8") as f:
            for c in approved:
                record = {
                    "id": c.get("curated_id", c["id"]),
                    "turns": [{"role": "user", "content": c["question"]}],
                    "evaluation_rubric": {
                        "expected_facts": c.get("expected_facts_reviewed") or c.get("expected_facts_suggested", []),
                        "forbidden_claims": c.get("forbidden_claims_reviewed") or c.get("forbidden_claims_suggested", []),
                    },
                    "category": c.get("category", "geral"),
                    "intent_type": c.get("intent_type", "duvida_geral"),
                    "handoff_expected": bool(c.get("handoff_expected_reviewed")),
                    "handoff_reason": c.get("handoff_reason_reviewed", ""),
                    "grounding_chunk_ids": c.get("grounding_chunk_ids_reviewed") or c.get("grounding_chunk_ids_suggested", []),
                    "source_model": c.get("source_model"),
                    "curator_model": c.get("curator_model"),
                    "annotator_model": c.get("annotator_model"),
                    "reviewed_by": c.get("reviewed_by", "Pedro Saulo Brito"),
                    "reviewed_at": c.get("reviewed_at"),
                    "synthetic": True,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return len(approved), str(export_file)


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    store: ReviewStore

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(index_file.read_bytes())
                return
            else:
                self.send_error(404, "Static index.html not found")
                return

        if path.startswith("/static/"):
            rel_file = path[8:]
            target = STATIC_DIR / rel_file
            if target.exists() and not target.is_dir():
                return super().do_GET()

        if path == "/api/cases":
            self._send_json(self.store.get_summary_list())
            return

        if path == "/api/stats":
            self._send_json(self.store.get_stats())
            return

        if path.startswith("/api/cases/"):
            case_id = path.split("/")[3]
            case = self.store.get_case(case_id)
            if case:
                self._send_json(case)
            else:
                self._send_error(404, f"Case '{case_id}' not found")
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        reviewer = payload.get("reviewer", "Pedro Saulo Brito")

        if path == "/api/export":
            count, out_file = self.store.export_canonical_dataset()
            self._send_json({"status": "ok", "exported_count": count, "path": out_file})
            return

        if path.startswith("/api/cases/"):
            parts = path.split("/")
            case_id = parts[3]
            action = parts[4] if len(parts) > 4 else "save"

            if action == "approve_suggestions":
                next_id = self.store.approve_with_suggestions(case_id, reviewer=reviewer)
                self._send_json({"status": "ok", "action": "approved_suggestions", "next_id": next_id})
                return

            if action == "save":
                next_id = self.store.save_custom_review(case_id, payload, reviewer=reviewer)
                self._send_json({"status": "ok", "action": "saved", "next_id": next_id})
                return

            if action == "reject":
                reason = payload.get("rejection_reason", "Rejeitado pelo revisor")
                next_id = self.store.reject_case(case_id, reason=reason, reviewer=reviewer)
                self._send_json({"status": "ok", "action": "rejected", "next_id": next_id})
                return

        self._send_error(400, "Invalid POST endpoint")

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str):
        self._send_json({"status": "error", "message": message}, status=status)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    input_path: Path = DEFAULT_INPUT_PATH,
    store_path: Path = DEFAULT_STORE_PATH,
):
    """Run the local HTTP review server."""
    store = ReviewStore(input_path=input_path, store_path=store_path)
    ReviewRequestHandler.store = store

    server = HTTPServer((host, port), ReviewRequestHandler)
    print(f"============================================================", file=sys.stderr)
    print(f"  MBA TCC - Interface de Revisão Humana Ativa               ", file=sys.stderr)
    print(f"  URL: http://{host}:{port}                                 ", file=sys.stderr)
    print(f"  Casos carregados: {len(store.cases)}                      ", file=sys.stderr)
    print(f"============================================================", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[review-server] Stopping server.", file=sys.stderr)
    finally:
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
