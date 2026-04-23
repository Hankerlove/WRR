from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import WatchRetrieveRespondAgent
from .backend import build_backend
from .benchmarks import convert_ovo_annotations, convert_river_annotations
from .config import load_config
from .data import load_episode, load_manifest
from .eval import evaluate_episode, evaluate_manifest
from .scoring import detect_benchmark, extract_final_responses, load_run_output, score_ovo, score_river
from .train import train_controller_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="WRR CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="run a single episode")
    demo_parser.add_argument("--config", required=True)
    demo_parser.add_argument("--episode", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="evaluate a single episode")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--episode", required=True)

    run_manifest_parser = subparsers.add_parser("run-manifest", help="run a manifest of episodes")
    run_manifest_parser.add_argument("--config", required=True)
    run_manifest_parser.add_argument("--manifest", required=True)
    run_manifest_parser.add_argument("--limit", type=int, default=None)
    run_manifest_parser.add_argument("--output", default=None)
    run_manifest_parser.add_argument("--score-mode", choices=["auto", "generic", "ovo", "river"], default="auto")

    train_gate_parser = subparsers.add_parser("train-gate", help="train a lightweight controller gate")
    train_gate_parser.add_argument("--config", required=True)
    train_gate_parser.add_argument("--manifest", required=True)
    train_gate_parser.add_argument("--output", required=True)
    train_gate_parser.add_argument("--limit", type=int, default=None)
    train_gate_parser.add_argument("--epochs", type=int, default=5)
    train_gate_parser.add_argument("--hidden-dim", type=int, default=32)
    train_gate_parser.add_argument("--learning-rate", type=float, default=1e-3)

    prepare_ovo_parser = subparsers.add_parser("prepare-ovo", help="convert OVO-Bench annotations to a WRR manifest")
    prepare_ovo_parser.add_argument("--annotations", required=True)
    prepare_ovo_parser.add_argument("--video-root", required=True)
    prepare_ovo_parser.add_argument("--output", required=True)
    prepare_ovo_parser.add_argument("--sampling-fps", type=float, default=1.0)
    prepare_ovo_parser.add_argument("--response-slack", type=float, default=1.0)

    prepare_river_parser = subparsers.add_parser("prepare-river", help="convert RIVER annotations to a WRR manifest")
    prepare_river_parser.add_argument("--annotations", required=True)
    prepare_river_parser.add_argument("--video-root", required=True)
    prepare_river_parser.add_argument("--output", required=True)
    prepare_river_parser.add_argument("--sampling-fps", type=float, default=1.0)
    prepare_river_parser.add_argument("--field-map", default=None)
    prepare_river_parser.add_argument("--response-slack", type=float, default=1.0)

    score_ovo_parser = subparsers.add_parser("score-ovo", help="score a run-manifest output using OVO-style reporting")
    score_ovo_parser.add_argument("--manifest", required=True)
    score_ovo_parser.add_argument("--run-output", required=True)
    score_ovo_parser.add_argument("--limit", type=int, default=None)

    score_river_parser = subparsers.add_parser("score-river", help="score a run-manifest output using RIVER-style reporting")
    score_river_parser.add_argument("--manifest", required=True)
    score_river_parser.add_argument("--run-output", required=True)
    score_river_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.config, args.episode)
        return
    if args.command == "evaluate":
        run_evaluation(args.config, args.episode)
        return
    if args.command == "run-manifest":
        run_manifest(args.config, args.manifest, limit=args.limit, output=args.output, score_mode=args.score_mode)
        return
    if args.command == "train-gate":
        run_train_gate(
            args.config,
            args.manifest,
            args.output,
            limit=args.limit,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
        )
        return
    if args.command == "prepare-ovo":
        count = convert_ovo_annotations(
            annotation_path=args.annotations,
            video_root=args.video_root,
            output_path=args.output,
            sampling_fps=args.sampling_fps,
            response_slack=args.response_slack,
        )
        print(json.dumps({"episodes_written": count, "output": args.output}, ensure_ascii=True))
        return
    if args.command == "prepare-river":
        count = convert_river_annotations(
            annotation_path=args.annotations,
            video_root=args.video_root,
            output_path=args.output,
            sampling_fps=args.sampling_fps,
            field_map_path=args.field_map,
            response_slack=args.response_slack,
        )
        print(json.dumps({"episodes_written": count, "output": args.output}, ensure_ascii=True))
        return
    if args.command == "score-ovo":
        run_score_ovo(args.manifest, args.run_output, limit=args.limit)
        return
    if args.command == "score-river":
        run_score_river(args.manifest, args.run_output, limit=args.limit)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def run_demo(config_path: str, episode_path: str) -> None:
    config = load_config(config_path)
    episode = load_episode(episode_path)
    agent = WatchRetrieveRespondAgent(build_backend(config), config)
    result = agent.run_episode(episode)
    for decision in result.decisions:
        payload = {
            "timestamp": decision.timestamp,
            "query_id": decision.query_id,
            "action": decision.action.value,
            "answer": decision.answer,
            "confidence": round(decision.confidence, 4),
            "used_retrieval": decision.used_retrieval,
            "reason": decision.reason,
            "retrieved_event_ids": decision.retrieved_event_ids,
        }
        print(json.dumps(payload, ensure_ascii=True))


def run_evaluation(config_path: str, episode_path: str) -> None:
    config = load_config(config_path)
    episode = load_episode(episode_path)
    agent = WatchRetrieveRespondAgent(build_backend(config), config)
    result = agent.run_episode(episode)
    summary = evaluate_episode(episode, result.decisions)
    payload = {
        "overall_accuracy": round(summary.overall_accuracy, 4),
        "overall_timing": round(summary.overall_timing, 4),
        "by_type_accuracy": {key: round(value, 4) for key, value in summary.by_type_accuracy.items()},
        "by_type_timing": {key: round(value, 4) for key, value in summary.by_type_timing.items()},
        "by_task_accuracy": {key: round(value, 4) for key, value in summary.by_task_accuracy.items()},
        "answered_queries": summary.answered_queries,
        "total_queries": summary.total_queries,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def run_manifest(
    config_path: str,
    manifest_path: str,
    limit: int | None = None,
    output: str | None = None,
    score_mode: str = "auto",
) -> None:
    config = load_config(config_path)
    episodes = load_manifest(manifest_path, limit=limit)
    agent = WatchRetrieveRespondAgent(build_backend(config), config)
    decisions_by_episode: dict[str, list[dict]] = {}
    decision_objects_by_episode = {}

    for episode in episodes:
        result = agent.run_episode(episode)
        decision_objects_by_episode[episode.episode_id] = result.decisions
        decisions_by_episode[episode.episode_id] = [
            {
                "timestamp": decision.timestamp,
                "query_id": decision.query_id,
                "action": decision.action.value,
                "answer": decision.answer,
                "confidence": decision.confidence,
                "used_retrieval": decision.used_retrieval,
                "reason": decision.reason,
                "retrieved_event_ids": decision.retrieved_event_ids,
            }
            for decision in result.decisions
        ]

    summary = evaluate_manifest(episodes, decision_objects_by_episode)
    payload = {
        "overall_accuracy": round(summary.overall_accuracy, 4),
        "overall_timing": round(summary.overall_timing, 4),
        "by_type_accuracy": {key: round(value, 4) for key, value in summary.by_type_accuracy.items()},
        "by_type_timing": {key: round(value, 4) for key, value in summary.by_type_timing.items()},
        "by_task_accuracy": {key: round(value, 4) for key, value in summary.by_task_accuracy.items()},
        "answered_queries": summary.answered_queries,
        "total_queries": summary.total_queries,
    }

    run_output = {
        "summary": payload,
        "decisions": decisions_by_episode,
        "final_responses": extract_final_responses({"decisions": decisions_by_episode}),
    }

    benchmark_report = None
    effective_score_mode = score_mode
    if effective_score_mode == "auto":
        effective_score_mode = detect_benchmark(episodes) or "generic"
    if effective_score_mode == "ovo":
        benchmark_report = score_ovo(episodes, run_output)
    elif effective_score_mode == "river":
        benchmark_report = score_river(episodes, run_output)

    if benchmark_report is not None:
        print(benchmark_report.text)
        run_output["benchmark_score"] = {
            "benchmark": benchmark_report.benchmark,
            "metrics": benchmark_report.metrics,
            "breakdown": benchmark_report.breakdown,
        }
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(run_output, indent=2, ensure_ascii=True))


def run_train_gate(
    config_path: str,
    manifest_path: str,
    output_path: str,
    limit: int | None = None,
    epochs: int = 5,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
) -> None:
    config = load_config(config_path)
    episodes = load_manifest(manifest_path, limit=limit)
    agent = WatchRetrieveRespondAgent(build_backend(config), config)
    result = train_controller_gate(
        episodes=episodes,
        config=config,
        agent=agent,
        output_path=output_path,
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
    )
    payload = {
        "num_examples": result.num_examples,
        "final_loss": round(result.final_loss, 6),
        "train_accuracy": round(result.train_accuracy, 4),
        "checkpoint_path": result.checkpoint_path,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def run_score_ovo(manifest_path: str, run_output_path: str, limit: int | None = None) -> None:
    episodes = load_manifest(manifest_path, limit=limit, metadata_only=True)
    run_output = load_run_output(run_output_path)
    report = score_ovo(episodes, run_output)
    print(report.text)


def run_score_river(manifest_path: str, run_output_path: str, limit: int | None = None) -> None:
    episodes = load_manifest(manifest_path, limit=limit, metadata_only=True)
    run_output = load_run_output(run_output_path)
    report = score_river(episodes, run_output)
    print(report.text)


if __name__ == "__main__":
    main()
