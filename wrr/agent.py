from __future__ import annotations

from .backend import BaseVideoBackend
from .config import AgentConfig
from .controller import build_controller
from .memory import EventCache
from .retriever import QueryConditionedRetriever
from .types import AnswerProposal, DecisionRecord, Episode, EpisodeResult, FrameObservation, Query, QueryRuntimeState, RetrievalHit, WindowState
from .watcher import RecentWindowWatcher


class WatchRetrieveRespondAgent:
    def __init__(self, backend: BaseVideoBackend, config: AgentConfig) -> None:
        self.backend = backend
        self.config = config
        self.watcher = RecentWindowWatcher(backend, config.window_size)
        self.cache = EventCache(config.cache)
        self.retriever = QueryConditionedRetriever(backend, config.cache.top_k)
        self.controller = build_controller(config.controller)
        self.active_queries: dict[str, QueryRuntimeState] = {}

    def reset(self) -> None:
        self.watcher.reset()
        self.cache.reset()
        self.active_queries.clear()

    def run_episode(self, episode: Episode) -> EpisodeResult:
        self.reset()
        decisions: list[DecisionRecord] = []
        pending_queries = sorted(episode.queries, key=lambda query: query.timestamp)
        query_idx = 0

        for frame in sorted(episode.frames, key=lambda item: item.timestamp):
            self.observe(frame)
            while query_idx < len(pending_queries) and pending_queries[query_idx].timestamp <= frame.timestamp:
                self.register_query(pending_queries[query_idx])
                query_idx += 1
            for runtime_state in self.active_queries.values():
                if runtime_state.answered:
                    continue
                decision = self.step_query(episode.episode_id, runtime_state)
                decisions.append(decision)
        return EpisodeResult(episode_id=episode.episode_id, decisions=decisions)

    def observe(self, frame: FrameObservation) -> None:
        state = self.watcher.observe(frame)
        self.cache.consider_write(state)

    def register_query(self, query: Query) -> None:
        self.active_queries[query.query_id] = QueryRuntimeState(query=query)

    def step_query(self, episode_id: str, runtime_state: QueryRuntimeState) -> DecisionRecord:
        current_state, current_proposal, retrieval_hits, contextual_proposal = self.inspect_query_state(runtime_state)

        candidate_for_stability = contextual_proposal if contextual_proposal.confidence >= current_proposal.confidence else current_proposal
        self._update_stability(runtime_state, candidate_for_stability.answer, candidate_for_stability.confidence)
        decision = self.controller.decide(runtime_state, current_proposal, contextual_proposal, retrieval_hits)
        runtime_state.wait_steps += 1

        if decision.action.value == "RESPOND":
            runtime_state.answered = True
            runtime_state.answer_time = current_state.timestamp

        return DecisionRecord(
            episode_id=episode_id,
            query_id=runtime_state.query.query_id,
            timestamp=current_state.timestamp,
            action=decision.action,
            answer=decision.answer,
            confidence=decision.confidence,
            used_retrieval=decision.used_retrieval,
            reason=decision.reason,
            retrieved_event_ids=decision.retrieved_event_ids,
        )

    def inspect_query_state(
        self,
        runtime_state: QueryRuntimeState,
    ) -> tuple[WindowState, AnswerProposal, list[RetrievalHit], AnswerProposal]:
        current_state = self.watcher.current_state
        if current_state is None:
            raise RuntimeError("Cannot inspect a query state before observing any frames.")
        current_proposal = self.backend.propose_answer(runtime_state.query, current_state, retrieved_hits=())
        retrieval_hits = self.retriever.retrieve(runtime_state.query, self.cache)
        contextual_proposal = self.backend.propose_answer(
            runtime_state.query,
            current_state,
            retrieved_hits=retrieval_hits,
        )
        return current_state, current_proposal, retrieval_hits, contextual_proposal

    def _update_stability(self, runtime_state: QueryRuntimeState, answer: str | None, confidence: float) -> None:
        if answer is None or confidence < self.config.controller.stability_confidence:
            runtime_state.last_answer = None
            runtime_state.consecutive_stable_steps = 0
            return
        if runtime_state.last_answer == answer:
            runtime_state.consecutive_stable_steps += 1
        else:
            runtime_state.last_answer = answer
            runtime_state.consecutive_stable_steps = 1
