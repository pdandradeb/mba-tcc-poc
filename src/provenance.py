"""Method code identity, separate from result serialization and scheduling."""
from pathlib import Path
from .corpus import sha256

ROOT=Path(__file__).resolve().parents[1]
COMMON={'src/__init__.py','src/cases.py','src/config.py','src/corpus.py','src/models.py',
        'src/pipeline.py','src/heuristic_engine.py','src/nlu_extractor.py','src/closing_llm.py',
        'src/grounding.py','src/coaching_units.py','src/coaching_fact_audit.py','src/telemetry.py',
        'data/cases.jsonl','data/public_corpus.json','data/publication_manifest.json',
        'pyproject.toml','uv.lock'}


def implementation_files(method):
    paths=set(COMMON)
    if method!='heuristic':paths.update({'src/store.py','src/providers.py','sql/001_knowledge.sql'})
    if method in {'vector','neuro_symbolic'}:paths.update({'src/methods.py','src/review_recording.py'})
    return sorted(paths)


def current_method_hashes(method):
    return {name:sha256((ROOT/name).read_bytes()) for name in implementation_files(method)}


def recorded_method_hashes(manifest,method):
    hashes=manifest['source_sha256']
    return {name:hashes[name] for name in implementation_files(method)}
