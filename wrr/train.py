from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import WatchRetrieveRespondAgent
from .config import AgentConfig
from .controller import ACTION_INDEX, build_feature_vector
from .eval import answer_matches_query
from .types import Action, Episode, QueryRuntimeState


@dataclass(slots=True)
class GateTrainingResult:
    num_examples: int
    final_loss: float
    train_accuracy: float
    checkpoint_path: str


def train_controller_gate(
    episodes: list[Episode],
    config: AgentConfig,
    agent: WatchRetrieveRespondAgent,
    output_path: str | Path,
    epochs: int = 5,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
) -> GateTrainingResult:
    features, labels = collect_gate_examples(episodes, agent)
    if not features:
        raise RuntimeError("No gate-training examples were collected from the provided episodes.")

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Gate training requires torch.") from exc

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, len(ACTION_INDEX)),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        preds = torch.argmax(model(x), dim=-1)
        train_accuracy = float((preds == y).float().mean().item())

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "train_accuracy": train_accuracy,
        "final_loss": final_loss,
        "project_title": config.project_title,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return GateTrainingResult(
        num_examples=len(features),
        final_loss=final_loss,
        train_accuracy=train_accuracy,
        checkpoint_path=str(destination),
    )


def collect_gate_examples(
    episodes: list[Episode],
    agent: WatchRetrieveRespondAgent,
) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []

    for episode in episodes:
        agent.reset()
        pending_queries = sorted(episode.queries, key=lambda query: query.timestamp)
        query_idx = 0
        for frame in sorted(episode.frames, key=lambda item: item.timestamp):
            agent.observe(frame)
            while query_idx < len(pending_queries) and pending_queries[query_idx].timestamp <= frame.timestamp:
                agent.register_query(pending_queries[query_idx])
                query_idx += 1

            for runtime_state in agent.active_queries.values():
                if runtime_state.answered:
                    continue
                current_state, current_proposal, retrieval_hits, contextual_proposal = agent.inspect_query_state(runtime_state)
                candidate = contextual_proposal if contextual_proposal.confidence >= current_proposal.confidence else current_proposal
                agent._update_stability(runtime_state, candidate.answer, candidate.confidence)
                features.append(build_feature_vector(runtime_state, current_proposal, contextual_proposal, retrieval_hits))
                label = _oracle_action(runtime_state, current_state.timestamp, current_proposal, contextual_proposal, retrieval_hits)
                labels.append(ACTION_INDEX[label])
                if label == Action.RESPOND:
                    runtime_state.answered = True
                    runtime_state.answer_time = current_state.timestamp
                runtime_state.wait_steps += 1

    return features, labels


def _oracle_action(
    runtime_state: QueryRuntimeState,
    timestamp: float,
    current_proposal,
    contextual_proposal,
    retrieval_hits,
) -> Action:
    query = runtime_state.query
    in_response_window = _in_response_window(timestamp, query.response_window)
    before_response_window = _before_response_window(timestamp, query.response_window)
    current_correct = answer_matches_query(current_proposal.answer, query)
    contextual_correct = answer_matches_query(contextual_proposal.answer, query)
    retrieval_promising = bool(retrieval_hits) and contextual_proposal.confidence > current_proposal.confidence

    if before_response_window:
        return Action.RETRIEVE if retrieval_promising else Action.WAIT

    if current_correct or contextual_correct:
        return Action.RESPOND

    if retrieval_promising:
        return Action.RETRIEVE

    return Action.WAIT


def _in_response_window(timestamp: float, response_window: tuple[float, float] | None) -> bool:
    if response_window is None:
        return True
    start, end = response_window
    return start <= timestamp <= end


def _before_response_window(timestamp: float, response_window: tuple[float, float] | None) -> bool:
    if response_window is None:
        return False
    start, _ = response_window
    return timestamp < start
