from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .benchmarks.ovo import BACKWARD_TASKS, FORWARD_TASKS, REALTIME_TASKS
from .eval import answer_matches_query
from .types import Episode, QueryType


@dataclass(slots=True)
class BenchmarkScoreReport:
    benchmark: str
    metrics: dict[str, float]
    breakdown: dict[str, dict[str, float]]
    text: str


def load_run_output(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def extract_final_responses(run_output: dict) -> dict[str, dict[str, dict]]:
    by_episode: dict[str, dict[str, dict]] = {}
    decisions = run_output.get("decisions", {})
    for episode_id, episode_decisions in decisions.items():
        per_query: dict[str, dict] = {}
        for decision in episode_decisions:
            if decision["action"] != "RESPOND":
                continue
            per_query[decision["query_id"]] = decision
        by_episode[episode_id] = per_query
    return by_episode


def detect_benchmark(episodes: list[Episode]) -> str | None:
    names = {str(episode.metadata.get("benchmark", "")).lower() for episode in episodes if episode.metadata.get("benchmark")}
    if len(names) == 1:
        return next(iter(names))
    return None


def score_ovo(episodes: list[Episode], run_output: dict) -> BenchmarkScoreReport:
    responses = extract_final_responses(run_output)
    task_scores: dict[str, list[float]] = {task: [] for task in sorted(BACKWARD_TASKS | REALTIME_TASKS | FORWARD_TASKS)}

    for episode in episodes:
        episode_responses = responses.get(episode.episode_id, {})
        for query in episode.queries:
            task = str(query.metadata.get("task", "")).upper()
            record = episode_responses.get(query.query_id)
            response = record["answer"] if record is not None else None
            if task in BACKWARD_TASKS or task in REALTIME_TASKS:
                task_scores[task].append(1.0 if answer_matches_query(response, query) else 0.0)
            elif task == "REC":
                task_scores[task].append(_score_rec(response, query))
            elif task in {"SSR", "CRR"}:
                task_scores[task].append(_score_yes_no(response, query))

    backward = _group_task_scores(task_scores, BACKWARD_TASKS)
    realtime = _group_task_scores(task_scores, REALTIME_TASKS)
    forward = _group_task_scores(task_scores, FORWARD_TASKS)

    backward_avg = _mean(list(backward.values()))
    realtime_avg = _mean(list(realtime.values()))
    forward_avg = _mean(list(forward.values()))
    total_avg = _mean([value for value in [backward_avg, realtime_avg, forward_avg] if value >= 0])

    lines = [
        "OVO-Bench Evaluation",
        "Evaluate Backward Tracing...",
    ]
    for task in sorted(backward):
        lines.append(f"Task: {task}, Acc: {backward[task]:.2f}")
    lines.append(f"Backward Avg.: {_format_metric(backward_avg)}")
    lines.append("")
    lines.append("Evaluate Real-time Visual Perception...")
    for task in sorted(realtime):
        lines.append(f"Task: {task}, Acc: {realtime[task]:.2f}")
    lines.append(f"Realtime Avg.: {_format_metric(realtime_avg)}")
    lines.append("")
    lines.append("Evaluate Forward Active Responding...")
    for task in sorted(forward):
        lines.append(f"Task: {task}, Acc: {forward[task]:.2f}")
    lines.append(f"Forward Avg.: {_format_metric(forward_avg)}")
    lines.append("")
    lines.append(f"Total Avg.: {_format_metric(total_avg)}")

    metrics = {
        "backward_avg": backward_avg,
        "realtime_avg": realtime_avg,
        "forward_avg": forward_avg,
        "total_avg": total_avg,
    }
    breakdown = {
        "backward": backward,
        "realtime": realtime,
        "forward": forward,
    }
    return BenchmarkScoreReport(benchmark="ovo", metrics=metrics, breakdown=breakdown, text="\n".join(lines))


def score_river(episodes: list[Episode], run_output: dict) -> BenchmarkScoreReport:
    responses = extract_final_responses(run_output)
    buckets: dict[str, list[float]] = {
        "retro_mc": [],
        "retro_oe": [],
        "live_mc": [],
        "live_oe": [],
        "instant_loc": [],
        "instant_mc": [],
        "instant_oe": [],
        "streaming_oe": [],
    }
    task_accuracy: dict[str, list[float]] = {}

    for episode in episodes:
        episode_responses = responses.get(episode.episode_id, {})
        for query in episode.queries:
            record = episode_responses.get(query.query_id)
            response = record["answer"] if record is not None else None
            response_time = float(record["timestamp"]) if record is not None else None
            task_name = str(query.metadata.get("task", query.query_type.value))
            score_mode = str(query.metadata.get("score_mode", "oe")).lower()
            response_mode = str(query.metadata.get("response_mode", "instant")).lower()

            correct = 1.0 if answer_matches_query(response, query) else 0.0
            task_accuracy.setdefault(task_name, []).append(correct)

            if query.query_type == QueryType.RETRO:
                buckets["retro_mc" if score_mode == "mc" else "retro_oe"].append(correct)
                continue

            if query.query_type == QueryType.LIVE:
                buckets["live_mc" if score_mode == "mc" else "live_oe"].append(correct)
                continue

            timing = _river_loc_score(response_time, query.response_window)
            buckets["instant_loc"].append(timing)
            if response_mode == "streaming":
                buckets["streaming_oe"].append(correct)
            else:
                buckets["instant_mc" if score_mode == "mc" else "instant_oe"].append(correct)

    metric_values = {key: _mean_percent(values) for key, values in buckets.items() if values}
    retro_avg = _mean([metric_values[key] for key in ["retro_mc", "retro_oe"] if key in metric_values])
    live_avg = _mean([metric_values[key] for key in ["live_mc", "live_oe"] if key in metric_values])
    pro_avg = _mean([metric_values[key] for key in ["instant_loc", "instant_mc", "instant_oe", "streaming_oe"] if key in metric_values])
    overall_avg = _mean([value for value in [retro_avg, live_avg, pro_avg] if value >= 0])

    lines = ["RIVER Evaluation", "Evaluate Retro-Memory..."]
    if "retro_mc" in metric_values:
        lines.append(f"Retro-MC Acc: {metric_values['retro_mc']:.2f}")
    if "retro_oe" in metric_values:
        lines.append(f"Retro-OE Acc: {metric_values['retro_oe']:.2f}")
    lines.append(f"Retro Avg.: {_format_metric(retro_avg)}")
    lines.append("")
    lines.append("Evaluate Live-Perception...")
    if "live_mc" in metric_values:
        lines.append(f"Live-MC Acc: {metric_values['live_mc']:.2f}")
    if "live_oe" in metric_values:
        lines.append(f"Live-OE Acc: {metric_values['live_oe']:.2f}")
    lines.append(f"Live Avg.: {_format_metric(live_avg)}")
    lines.append("")
    lines.append("Evaluate Pro-Response...")
    if "instant_loc" in metric_values:
        lines.append(f"Instant-Loc: {metric_values['instant_loc']:.2f}")
    if "instant_mc" in metric_values:
        lines.append(f"Instant-MC: {metric_values['instant_mc']:.2f}")
    if "instant_oe" in metric_values:
        lines.append(f"Instant-OE: {metric_values['instant_oe']:.2f}")
    if "streaming_oe" in metric_values:
        lines.append(f"Streaming-OE: {metric_values['streaming_oe']:.2f}")
    lines.append(f"Pro Avg.: {_format_metric(pro_avg)}")
    lines.append("")
    lines.append(f"Overall Avg.: {_format_metric(overall_avg)}")

    metrics = {
        **metric_values,
        "retro_avg": retro_avg,
        "live_avg": live_avg,
        "pro_avg": pro_avg,
        "overall_avg": overall_avg,
    }
    breakdown = {
        "by_task_accuracy": {task: _mean_percent(values) for task, values in task_accuracy.items()},
    }
    return BenchmarkScoreReport(benchmark="river", metrics=metrics, breakdown=breakdown, text="\n".join(lines))


def score_streamingbench(episodes: list[Episode], run_output: dict) -> BenchmarkScoreReport:
    responses = extract_final_responses(run_output)
    category_scores: dict[str, list[float]] = {
        "real_time_visual_understanding": [],
        "omni_source_understanding": [],
        "contextual_understanding": [],
        "sequential_question_answering": [],
    }
    proactive_timing: dict[str, list[float]] = {
        "proactive_le_1s": [],
        "proactive_le_2s": [],
        "proactive_le_5s": [],
    }
    proactive_content: list[float] = []
    task_accuracy: dict[str, list[float]] = {}

    for episode in episodes:
        episode_responses = responses.get(episode.episode_id, {})
        for query in episode.queries:
            record = episode_responses.get(query.query_id)
            response = record["answer"] if record is not None else None
            response_time = float(record["timestamp"]) if record is not None else None
            category = _normalize_streamingbench_category(query.metadata.get("subset"))
            task_name = str(query.metadata.get("task", category))
            correct = 1.0 if answer_matches_query(response, query) else 0.0
            task_accuracy.setdefault(task_name, []).append(correct)

            if category == "proactive_output":
                proactive_content.append(correct)
                for tolerance in (1.0, 2.0, 5.0):
                    proactive_timing[f"proactive_le_{int(tolerance)}s"].append(
                        1.0 if correct and _streamingbench_proactive_hit(response_time, query.response_window, tolerance) else 0.0
                    )
                continue

            category_scores.setdefault(category, []).append(correct)

    metric_values = {key: _mean_percent(values) for key, values in category_scores.items() if values}
    proactive_metrics = {key: _mean_percent(values) for key, values in proactive_timing.items() if values}
    if proactive_content:
        proactive_metrics["proactive_content_acc"] = _mean_percent(proactive_content)
    main_avg = _mean([value for value in metric_values.values() if value >= 0])
    proactive_avg = _mean(
        [value for key, value in proactive_metrics.items() if key != "proactive_content_acc" and value >= 0]
    )

    lines = [
        "StreamingBench Evaluation",
        "Evaluate Main Benchmark...",
    ]
    labels = {
        "real_time_visual_understanding": "Real-Time Visual Understanding",
        "omni_source_understanding": "Omni-Source Understanding",
        "contextual_understanding": "Contextual Understanding",
        "sequential_question_answering": "Sequential Question Answering",
    }
    for key in (
        "real_time_visual_understanding",
        "omni_source_understanding",
        "contextual_understanding",
        "sequential_question_answering",
    ):
        if key in metric_values:
            lines.append(f"{labels[key]}: {metric_values[key]:.2f}")
    lines.append(f"Main Avg.: {_format_metric(main_avg)}")
    if proactive_metrics:
        lines.append("")
        lines.append("Evaluate Proactive Output...")
        if "proactive_content_acc" in proactive_metrics:
            lines.append(f"Content Acc: {proactive_metrics['proactive_content_acc']:.2f}")
        proactive_labels = {
            "proactive_le_1s": "<= 1s",
            "proactive_le_2s": "<= 2s",
            "proactive_le_5s": "<= 5s",
        }
        for key in ("proactive_le_1s", "proactive_le_2s", "proactive_le_5s"):
            if key in proactive_metrics:
                lines.append(f"{proactive_labels[key]}: {proactive_metrics[key]:.2f}")
        lines.append(f"Proactive Avg.: {_format_metric(proactive_avg)}")

    metrics = {
        **metric_values,
        **proactive_metrics,
        "main_avg": main_avg,
        "proactive_avg": proactive_avg,
    }
    breakdown = {
        "by_task_accuracy": {task: _mean_percent(values) for task, values in task_accuracy.items()},
    }
    return BenchmarkScoreReport(
        benchmark="streamingbench",
        metrics=metrics,
        breakdown=breakdown,
        text="\n".join(lines),
    )


def _group_task_scores(task_scores: dict[str, list[float]], tasks: set[str]) -> dict[str, float]:
    return {task: _mean_percent(task_scores[task]) for task in sorted(tasks) if task_scores.get(task)}


def _score_rec(response: str | None, query) -> float:
    if response is None:
        return 0.0
    numbers = re.findall(r"\d+", response)
    predicted = "".join(numbers)
    accepted = [str(answer) for answer in query.metadata.get("accepted_answers", [query.target_answer])]
    return 1.0 if predicted in accepted else 0.0


def _score_yes_no(response: str | None, query) -> float:
    if response is None:
        return 0.0
    text = response.strip().lower()
    accepted = [str(answer).strip().lower() for answer in query.metadata.get("accepted_answers", [query.target_answer])]
    for answer in accepted:
        if answer == text or answer in text or text in answer:
            return 1.0
    return 0.0


def _river_loc_score(response_time: float | None, response_window: tuple[float, float] | None) -> float:
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


def _streamingbench_proactive_hit(
    response_time: float | None,
    response_window: tuple[float, float] | None,
    tolerance: float,
) -> bool:
    if response_time is None:
        return False
    target_time = response_window[0] if response_window is not None else 0.0
    if response_time < target_time:
        return False
    return response_time - target_time <= tolerance


def _normalize_streamingbench_category(raw_value: object) -> str:
    text = str(raw_value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    alias_map = {
        "real_time_visual_understanding": "real_time_visual_understanding",
        "realtime_visual_understanding": "real_time_visual_understanding",
        "omni_source_understanding": "omni_source_understanding",
        "contextual_understanding": "contextual_understanding",
        "sequential_question_answering": "sequential_question_answering",
        "proactive_output": "proactive_output",
    }
    return alias_map.get(text, text or "real_time_visual_understanding")


def _mean_percent(values: list[float]) -> float:
    if not values:
        return -1.0
    return 100.0 * sum(values) / len(values)


def _mean(values: list[float]) -> float:
    valid = [value for value in values if value >= 0]
    if not valid:
        return -1.0
    return sum(valid) / len(valid)


def _format_metric(value: float) -> str:
    return "N/A" if value < 0 else f"{value:.2f}"
