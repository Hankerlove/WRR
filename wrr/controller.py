from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .config import ControllerConfig
from .types import Action, AgentDecision, AnswerProposal, QueryRuntimeState, QueryType, RetrievalHit

ACTION_INDEX = {
    Action.WAIT: 0,
    Action.RETRIEVE: 1,
    Action.RESPOND: 2,
}
INDEX_ACTION = {value: key for key, value in ACTION_INDEX.items()}


class BaseEvidenceController(ABC):
    @abstractmethod
    def decide(
        self,
        runtime_state: QueryRuntimeState,
        current_proposal: AnswerProposal,
        contextual_proposal: AnswerProposal,
        retrieval_hits: list[RetrievalHit],
    ) -> AgentDecision:
        raise NotImplementedError


class HeuristicEvidenceController(BaseEvidenceController):
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def decide(
        self,
        runtime_state: QueryRuntimeState,
        current_proposal: AnswerProposal,
        contextual_proposal: AnswerProposal,
        retrieval_hits: list[RetrievalHit],
    ) -> AgentDecision:
        query = runtime_state.query
        best_retrieval_score = retrieval_hits[0].score if retrieval_hits else 0.0
        contextual_is_better = contextual_proposal.confidence >= current_proposal.confidence + self.config.improvement_margin
        stable_enough = runtime_state.consecutive_stable_steps >= self.config.min_stable_steps
        retrieved_ids = [hit.event.event_id for hit in retrieval_hits]

        if query.query_type == QueryType.PROACTIVE:
            if current_proposal.confidence >= self.config.proactive_threshold and stable_enough:
                return AgentDecision(
                    action=Action.RESPOND,
                    answer=current_proposal.answer,
                    confidence=current_proposal.confidence,
                    used_retrieval=False,
                    reason="current window satisfies proactive condition",
                )
            if contextual_is_better and contextual_proposal.confidence >= self.config.proactive_threshold and stable_enough:
                return AgentDecision(
                    action=Action.RESPOND,
                    answer=contextual_proposal.answer,
                    confidence=contextual_proposal.confidence,
                    used_retrieval=True,
                    reason="retrieved context stabilizes proactive response",
                    retrieved_event_ids=retrieved_ids,
                )
            return AgentDecision(
                action=Action.WAIT,
                answer=None,
                confidence=max(current_proposal.confidence, contextual_proposal.confidence),
                used_retrieval=bool(retrieval_hits),
                reason="proactive condition not yet satisfied",
                retrieved_event_ids=retrieved_ids,
            )

        if query.query_type == QueryType.LIVE:
            if current_proposal.confidence >= self.config.respond_threshold:
                return AgentDecision(
                    action=Action.RESPOND,
                    answer=current_proposal.answer,
                    confidence=current_proposal.confidence,
                    used_retrieval=False,
                    reason="current evidence is sufficient for a live query",
                )
            if contextual_is_better and contextual_proposal.confidence >= self.config.respond_threshold:
                return AgentDecision(
                    action=Action.RESPOND,
                    answer=contextual_proposal.answer,
                    confidence=contextual_proposal.confidence,
                    used_retrieval=True,
                    reason="retrieved context improves the live answer",
                    retrieved_event_ids=retrieved_ids,
                )
            return AgentDecision(
                action=Action.WAIT,
                answer=None,
                confidence=max(current_proposal.confidence, contextual_proposal.confidence),
                used_retrieval=bool(retrieval_hits),
                reason="live query confidence is still too low",
                retrieved_event_ids=retrieved_ids,
            )

        if contextual_is_better and best_retrieval_score >= self.config.retrieve_threshold:
            if contextual_proposal.confidence >= self.config.respond_threshold:
                return AgentDecision(
                    action=Action.RESPOND,
                    answer=contextual_proposal.answer,
                    confidence=contextual_proposal.confidence,
                    used_retrieval=True,
                    reason="retrieved memory resolves a retro query",
                    retrieved_event_ids=retrieved_ids,
                )
            return AgentDecision(
                action=Action.RETRIEVE,
                answer=None,
                confidence=contextual_proposal.confidence,
                used_retrieval=True,
                reason="retrieval is promising but evidence is not sufficient yet",
                retrieved_event_ids=retrieved_ids,
            )

        if current_proposal.confidence >= self.config.respond_threshold:
            return AgentDecision(
                action=Action.RESPOND,
                answer=current_proposal.answer,
                confidence=current_proposal.confidence,
                used_retrieval=False,
                reason="current window already supports the answer",
            )

        if runtime_state.wait_steps >= self.config.max_wait_steps:
            best = contextual_proposal if contextual_proposal.confidence > current_proposal.confidence else current_proposal
            return AgentDecision(
                action=Action.RESPOND,
                answer=best.answer,
                confidence=best.confidence,
                used_retrieval=best is contextual_proposal,
                reason="maximum wait budget reached",
                retrieved_event_ids=retrieved_ids if best is contextual_proposal else [],
            )

        return AgentDecision(
            action=Action.WAIT,
            answer=None,
            confidence=max(current_proposal.confidence, contextual_proposal.confidence),
            used_retrieval=bool(retrieval_hits),
            reason="waiting for stronger evidence",
            retrieved_event_ids=retrieved_ids,
        )


class LearnedEvidenceController(BaseEvidenceController):
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.fallback = HeuristicEvidenceController(config)
        self._model = None
        self._metadata: dict | None = None

    def decide(
        self,
        runtime_state: QueryRuntimeState,
        current_proposal: AnswerProposal,
        contextual_proposal: AnswerProposal,
        retrieval_hits: list[RetrievalHit],
    ) -> AgentDecision:
        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path is None:
            return self.fallback.decide(runtime_state, current_proposal, contextual_proposal, retrieval_hits)

        self._ensure_loaded(checkpoint_path)
        if self._model is None or self._metadata is None:
            return self.fallback.decide(runtime_state, current_proposal, contextual_proposal, retrieval_hits)

        import torch

        features = build_feature_vector(runtime_state, current_proposal, contextual_proposal, retrieval_hits)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)[0]
            probs = torch.softmax(logits, dim=-1)
        action_index = int(torch.argmax(probs).item())
        action = INDEX_ACTION[action_index]
        policy_confidence = float(torch.max(probs).item())
        if policy_confidence < 0.45:
            return self.fallback.decide(runtime_state, current_proposal, contextual_proposal, retrieval_hits)

        retrieved_ids = [hit.event.event_id for hit in retrieval_hits]
        best = contextual_proposal if contextual_proposal.confidence >= current_proposal.confidence else current_proposal
        if action == Action.RESPOND and best.answer is None:
            return self.fallback.decide(runtime_state, current_proposal, contextual_proposal, retrieval_hits)
        if action == Action.RESPOND:
            return AgentDecision(
                action=Action.RESPOND,
                answer=best.answer,
                confidence=max(best.confidence, policy_confidence),
                used_retrieval=best is contextual_proposal,
                reason="learned gate selected RESPOND",
                retrieved_event_ids=retrieved_ids if best is contextual_proposal else [],
            )
        if action == Action.RETRIEVE:
            return AgentDecision(
                action=Action.RETRIEVE,
                answer=None,
                confidence=policy_confidence,
                used_retrieval=bool(retrieval_hits),
                reason="learned gate selected RETRIEVE",
                retrieved_event_ids=retrieved_ids,
            )
        return AgentDecision(
            action=Action.WAIT,
            answer=None,
            confidence=policy_confidence,
            used_retrieval=bool(retrieval_hits),
            reason="learned gate selected WAIT",
            retrieved_event_ids=retrieved_ids,
        )

    def _ensure_loaded(self, checkpoint_path: str) -> None:
        if self._model is not None and self._metadata is not None:
            return
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("Learned gate inference requires torch.") from exc

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        input_dim = int(checkpoint["input_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ACTION_INDEX)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._model = model
        self._metadata = checkpoint


def build_feature_vector(
    runtime_state: QueryRuntimeState,
    current_proposal: AnswerProposal,
    contextual_proposal: AnswerProposal,
    retrieval_hits: list[RetrievalHit],
) -> list[float]:
    query = runtime_state.query
    best_retrieval_score = retrieval_hits[0].score if retrieval_hits else 0.0
    contextual_is_better = 1.0 if contextual_proposal.confidence > current_proposal.confidence else 0.0
    wait_ratio = min(runtime_state.wait_steps / max(runtime_state.wait_steps + 1, 1), 1.0)
    stable_ratio = min(runtime_state.consecutive_stable_steps / max(runtime_state.consecutive_stable_steps + 1, 1), 1.0)
    return [
        current_proposal.confidence,
        contextual_proposal.confidence,
        best_retrieval_score,
        contextual_proposal.confidence - current_proposal.confidence,
        contextual_is_better,
        1.0 if current_proposal.answer else 0.0,
        1.0 if contextual_proposal.answer else 0.0,
        wait_ratio,
        stable_ratio,
        1.0 if query.query_type == QueryType.RETRO else 0.0,
        1.0 if query.query_type == QueryType.LIVE else 0.0,
        1.0 if query.query_type == QueryType.PROACTIVE else 0.0,
    ]


def build_controller(config: ControllerConfig) -> BaseEvidenceController:
    if config.policy == "heuristic":
        return HeuristicEvidenceController(config)
    if config.policy == "learned":
        return LearnedEvidenceController(config)
    raise ValueError(f"Unsupported controller policy: {config.policy}")
