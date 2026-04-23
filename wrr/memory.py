from __future__ import annotations

from dataclasses import replace

from .config import CacheConfig
from .types import EventSlot, RetrievalHit, WindowState
from .vector_ops import average, cosine_similarity


class EventCache:
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.events: list[EventSlot] = []
        self._last_state: WindowState | None = None
        self._counter = 0

    def reset(self) -> None:
        self.events.clear()
        self._last_state = None
        self._counter = 0

    def consider_write(self, state: WindowState) -> bool:
        if not state.summary:
            self._last_state = state
            return False
        if not self.events:
            self._write_new_slot(state, salience=1.0)
            self._last_state = state
            return True
        change = 1.0 - cosine_similarity(state.embedding, self._last_state.embedding) if self._last_state else 1.0
        best_event, best_similarity = self._most_similar_event(state)
        novelty = 1.0 - max(best_similarity, 0.0)
        if change >= self.config.write_threshold and novelty >= self.config.novelty_threshold:
            self._write_new_slot(state, salience=(change + novelty) / 2.0)
            self._last_state = state
            return True
        if best_event is not None and best_similarity >= self.config.merge_threshold:
            self._merge_into(best_event.event_id, state)
        self._last_state = state
        return False

    def retrieve(self, query_embedding: list[float], top_k: int | None = None) -> list[RetrievalHit]:
        if not self.events:
            return []
        k = top_k or self.config.top_k
        hits = [
            RetrievalHit(event=event, score=cosine_similarity(query_embedding, event.embedding))
            for event in self.events
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    def _write_new_slot(self, state: WindowState, salience: float) -> None:
        self._counter += 1
        slot = EventSlot(
            event_id=f"evt-{self._counter}",
            summary=state.summary,
            embedding=list(state.embedding),
            start_time=state.timestamp,
            end_time=state.timestamp,
            salience=salience,
            support_count=1,
        )
        self.events.append(slot)
        self._trim()

    def _merge_into(self, event_id: str, state: WindowState) -> None:
        for idx, event in enumerate(self.events):
            if event.event_id != event_id:
                continue
            merged_summary = event.summary
            if state.summary not in event.summary:
                merged_summary = f"{event.summary} | {state.summary}"
            merged_embedding = average([event.embedding, state.embedding])
            self.events[idx] = replace(
                event,
                summary=merged_summary,
                embedding=merged_embedding,
                end_time=state.timestamp,
                salience=max(event.salience, 0.5),
                support_count=event.support_count + 1,
            )
            return

    def _most_similar_event(self, state: WindowState) -> tuple[EventSlot | None, float]:
        if not self.events:
            return None, 0.0
        best_event = max(self.events, key=lambda event: cosine_similarity(event.embedding, state.embedding))
        best_score = cosine_similarity(best_event.embedding, state.embedding)
        return best_event, best_score

    def _trim(self) -> None:
        if len(self.events) <= self.config.max_events:
            return
        self.events.sort(key=lambda event: (event.salience, event.end_time))
        self.events = self.events[-self.config.max_events :]
