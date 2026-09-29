"""Explicit publication manifest: unknown documents and changed content are rejected."""
import hashlib
import json
from pathlib import Path
import re

DATA = Path(__file__).resolve().parents[1] / 'data'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_public_text(text, *, corpus=False):
    patterns = [r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', r'-----BEGIN .*PRIVATE KEY',
                r'AIza[\w-]{20,}', r'\bsk-[\w-]{16,}', r'okf/(?:internal|assistente|system|sources)/',
                r'(?i)\b(?:senha|password|api_key)\s*[:=]\s*\S+',
                r'(?i)\b(?:onboarding|offboarding|wifi)\b']
    if corpus:
        patterns += [r'(?i)\bcomiss\w*', r'public\.[a-z_]+', r'condutive-services|/Users/']
    if any(re.search(p, text) for p in patterns):
        raise ValueError('Non-public content detected; inspect locally before publication')


def load_corpus(directory=DATA):
    directory = Path(directory)
    raw = (directory / 'public_corpus.json').read_bytes()
    manifest = json.loads((directory / 'publication_manifest.json').read_text())
    if sha256(raw) != manifest['corpus_sha256']:
        raise ValueError('Corpus changed without publication review')
    chunks = json.loads(raw)
    if not chunks or len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Empty corpus or duplicate chunk IDs')
    for chunk in chunks:
        if chunk['visibility'] != 'public' or chunk['source_path'] not in manifest['documents']:
            raise ValueError('Document is not in the publication allowlist')
        if sha256(chunk['content'].encode()) != chunk['content_hash']:
            raise ValueError('Chunk content hash mismatch')
        check_public_text(json.dumps(chunk, ensure_ascii=False), corpus=True)
    if set(manifest['documents']) != {c['source_path'] for c in chunks}:
        raise ValueError('Publication manifest and corpus differ')
    return sha256(raw), chunks
