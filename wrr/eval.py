from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .types import Action, DecisionRecord, Episode, Query


@dataclass(slots=True)
class EvaluationSummary:
    overall_accuracy: float
    overall_timing: float
    by_type_accuracy: dict[str, float]
    by_type_timing: dict[str, float]
    by_task_accuracy: dict[str, float]
    answered_queries: int
    total_queries: int


def evaluate_episode(episode: Episode, decisions: list[DecisionRecord]) -> EvaluationSummary:
    final_responses = _latest_responses(decisions)
    accuracy_bucket: dict[str, list[float]] = defaultdict(list)
    timing_bucket: dict[str, list[float]] = defaultdict(list)
    task_bucket: dict[str, list[float]] = defaultdict(list)
    answered = 0

    for query in episode.queries:
        record = final_responses.get(query.query_id)
        key = query.query_type.value
        if record is not None:
            answered += 1
        accuracy = 1.0 if answer_matches_query(record.answer if record else None, query) else 0.0
        accuracy_bucket[key].append(accuracy)
        timing_bucket[key].append(_timing_score(record.timestamp if record else None, query.response_window))
        task_name = str(query.metadata.get("task", "unknown"))
        task_bucket[task_name].append(accuracy)

    all_accuracy = [score for scores in accuracy_bucket.values() for score in scores]
    all_timing = [score for scores in timing_bucket.values() for score in scores]
    return EvaluationSummary(
        overall_accuracy=sum(all_accuracy) / len(all_accuracy) if all_accuracy else 0.0,
        overall_timing=sum(all_timing) / len(all_timing) if all_timing else 0.0,
        by_type_accuracy={key: sum(scores) / len(scores) for key, scores in accuracy_bucket.items()},
        by_type_timing={key: sum(scores) / len(scores) for key, scores in timing_bucket.items()},
        by_task_accuracy={key: sum(scores) / len(scores) for key, scores in task_bucket.items()},
        answered_queries=answered,
        total_queries=len(episode.queries),
    )


def evaluate_manifest(episodes: list[Episode], decisions_by_episode: dict[str, list[DecisionRecord]]) -> EvaluationSummary:
    merged_accuracy: dict[str, list[float]] = defaultdict(list)
    merged_timing: dict[str, list[float]] = defaultdict(list)
    merged_task_accuracy: dict[str, list[float]] = defaultdict(list)
    answered_queries = 0
    total_queries = 0

    for episode in episodes:
        summary = evaluate_episode(episode, decisions_by_episode.get(episode.episode_id, []))
        total_queries += summary.total_queries
        answered_queries += summary.answered_queries
        for key, value in summary.by_type_accuracy.items():
            merged_accuracy[key].append(value)
        for key, value in summary.by_type_timing.items():
            merged_timing[key].append(value)
        for key, value in summary.by_task_accuracy.items():
            merged_task_accuracy[key].append(value)

    overall_accuracy_values = [value for values in merged_accuracy.values() for value in values]
    overall_timing_values = [value for values in merged_timing.values() for value in values]
    return EvaluationSummary(
        overall_accuracy=sum(overall_accuracy_values) / len(overall_accuracy_values) if overall_accuracy_values else 0.0,
        overall_timing=sum(overall_timing_values) / len(overall_timing_values) if overall_timing_values else 0.0,
        by_type_accuracy={key: sum(values) / len(values) for key, values in merged_accuracy.items()},
        by_type_timing={key: sum(values) / len(values) for key, values in merged_timing.items()},
        by_task_accuracy={key: sum(values) / len(values) for key, values in merged_task_accuracy.items()},
        answered_queries=answered_queries,
        total_queries=total_queries,
    )


def _latest_responses(decisions: list[DecisionRecord]) -> dict[str, DecisionRecord]:
    responses: dict[str, DecisionRecord] = {}
    for decision in decisions:
        if decision.action != Action.RESPOND:
            continue
        responses[decision.query_id] = decision
    return responses


def answer_matches_query(predicted: str | None, query: Query) -> bool:
    if query.target_answer is None:
        return predicted is not None
    if predicted is None:
        return False
    normalized_pred = _normalize(predicted)
    accepted_answers = query.metadata.get("accepted_answers")
    if accepted_answers:
        for answer in accepted_answers:
            normalized_answer = _normalize(str(answer))
            if normalized_pred == normalized_answer:
                return True
            if normalized_answer in normalized_pred or normalized_pred in normalized_answer:
                return True
        return False
    normalized_target = _normalize(query.target_answer)
    if normalized_pred == normalized_target:
        return True
    if normalized_target in normalized_pred or normalized_pred in normalized_target:
        return True
    return False


def _timing_score(response_time: float | None, response_window: tuple[float, float] | None) -> float:
    if response_window is None:
        return 1.0 if response_time is not None else 0.0
    if response_time is None:
        return 0.0
    start, end = response_window
    if response_time < start:
        return 0.0
    if response_time <= end:
        return 1.0
    slack = max(end - start, 1.0)
    delay = response_time - end
    return max(0.0, 1.0 - delay / slack)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
