from dataclasses import dataclass
import os
import math

EMBEDDING_DIMENSION = 1536


def validate_timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Timeout must be a finite positive number of seconds')
    return float(value)

@dataclass(frozen=True)
class Settings:
    database_url: str
    model: str
    embedding_model: str
    top_k: int = 3
    llm_timeout_s: float = 60
    review_timeout_s: float = 120
    embedding_timeout_s: float = 60

    def __post_init__(self):
        for value in (self.llm_timeout_s, self.review_timeout_s, self.embedding_timeout_s):
            validate_timeout(value)

    @property
    def timeouts(self):
        return {'llm': self.llm_timeout_s, 'review': self.review_timeout_s,
                'embedding': self.embedding_timeout_s}

    @classmethod
    def from_env(cls):
        return cls(os.environ.get('DATABASE_URL', 'postgresql://mba_local:local-experiment-only@127.0.0.1:55432/mba_poc'),
                   os.environ.get('LLM_MODEL', 'gemini/gemini-3.8-flash'),
                   os.environ.get('EMBEDDING_MODEL', 'gemini/gemini-embedding-2'),
                   llm_timeout_s=float(os.environ.get('LLM_TIMEOUT_S', '60')),
                   review_timeout_s=float(os.environ.get('REVIEW_TIMEOUT_S', '120')),
                   embedding_timeout_s=float(os.environ.get('EMBEDDING_TIMEOUT_S', '60')))
