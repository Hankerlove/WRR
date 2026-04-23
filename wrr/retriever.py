from __future__ import annotations

from .backend import BaseVideoBackend
from .memory import EventCache
from .types import Query, RetrievalHit


class QueryConditionedRetriever:
    def __init__(self, backend: BaseVideoBackend, top_k: int) -> None:
        self.backend = backend
        self.top_k = top_k

    def retrieve(self, query: Query, cache: EventCache) -> list[RetrievalHit]:
        query_embedding = self.backend.embed_query(query)
        return cache.retrieve(query_embedding, top_k=self.top_k)
