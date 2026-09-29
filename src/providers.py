import math
from time import perf_counter
from .telemetry import response_metadata
from .config import EMBEDDING_DIMENSION, validate_timeout

class NeuralEmbedding:
    dimension = EMBEDDING_DIMENSION
    def __init__(self, model, provider=None, *, timeout_s=60):
        self.model = model
        self.provider = provider
        self.timeout_s = validate_timeout(timeout_s)
        self.execution_kind = 'fixture' if provider else 'live'
        self.events = []

    def embed(self, texts):
        if not texts or any(not t.strip() for t in texts):
            raise ValueError('Empty embedding input')
        start = perf_counter()
        event = {'component': 'embedding', 'execution_kind': self.execution_kind, 'requested_model': self.model, 'inputs': len(texts), 'timeout_s': self.timeout_s}
        try:
            if self.provider:
                response = self.provider(model=self.model, input=texts, dimensions=self.dimension, timeout=self.timeout_s)
            else:
                from litellm import embedding
                response = embedding(model=self.model, input=texts, dimensions=self.dimension, timeout=self.timeout_s)
            rows = sorted(response.data, key=lambda r: r['index'])
            if [r['index'] for r in rows] != list(range(len(texts))):
                raise ValueError('Embedding response indices do not match inputs')
            vectors = []
            for row in rows:
                vector = row['embedding']
                if len(vector) != self.dimension or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                    raise ValueError(f'Embedding must be finite and have {self.dimension} dimensions')
                norm = math.sqrt(sum(v*v for v in vector))
                if not norm or not math.isfinite(norm):
                    raise ValueError('Invalid embedding norm')
                vectors.append([v/norm for v in vector])
            event.update(status='success', dimension=self.dimension, **response_metadata(response))
            return vectors
        except Exception as exc:
            event.update(status='failed', error_type=type(exc).__name__)
            raise RuntimeError('Neural embedding failed; no lexical fallback') from exc
        finally:
            event['elapsed_s'] = perf_counter()-start
            self.events.append(event)
