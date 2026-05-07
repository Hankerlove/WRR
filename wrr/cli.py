from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from .agent import WatchRetrieveRespondAgent
from .backend import build_backend
from .benchmarks import convert_ovo_annotations, convert_river_annotations
from .config import load_config
from .data import iter_manifest, load_episode, load_manifest
from .eval import evaluate_episode, evaluate_manifest
from .scoring import detect_benchmark, extract_final_responses, load_run_output, score_ovo, score_river
from .train import train_controller_gate
from .types import Action, DecisionRecord


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
    run_manifest_parser.add_argument(
        "--score-mode",
        choices=["auto", "generic", "ovo", "river", "streamingbench"],
        default="auto",
    )

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

    prepare_streamingbench_parser = subparsers.add_parser(
        "prepare-streamingbench",
        help="convert StreamingBench annotations to a WRR manifest",
    )
    prepare_streamingbench_parser.add_argument("--annotations", required=True)
    prepare_streamingbench_parser.add_argument("--video-root", required=True)
    prepare_streamingbench_parser.add_argument("--output", required=True)
    prepare_streamingbench_parser.add_argument("--sampling-fps", type=float, default=1.0)
    prepare_streamingbench_parser.add_argument("--response-slack", type=float, default=1.0)
    prepare_streamingbench_parser.add_argument("--subset", default=None)

    score_ovo_parser = subparsers.add_parser("score-ovo", help="score a run-manifest output using OVO-style reporting")
    score_ovo_parser.add_argument("--manifest", required=True)
    score_ovo_parser.add_argument("--run-output", required=True)
    score_ovo_parser.add_argument("--limit", type=int, default=None)

    score_river_parser = subparsers.add_parser("score-river", help="score a run-manifest output using RIVER-style reporting")
    score_river_parser.add_argument("--manifest", required=True)
    score_river_parser.add_argument("--run-output", required=True)
    score_river_parser.add_argument("--limit", type=int, default=None)

    score_streamingbench_parser = subparsers.add_parser(
        "score-streamingbench",
        help="score a run-manifest output using StreamingBench-style reporting",
    )
    score_streamingbench_parser.add_argument("--manifest", required=True)
    score_streamingbench_parser.add_argument("--run-output", required=True)
    score_streamingbench_parser.add_argument("--limit", type=int, default=None)

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
    if args.command == "prepare-streamingbench":
        from .benchmarks.streamingbench import convert_streamingbench_annotations

        count = convert_streamingbench_annotations(
            annotation_path=args.annotations,
            video_root=args.video_root,
            output_path=args.output,
            sampling_fps=args.sampling_fps,
            response_slack=args.response_slack,
            subset_override=args.subset,
        )
        print(json.dumps({"episodes_written": count, "output": args.output}, ensure_ascii=True))
        return
    if args.command == "score-ovo":
        run_score_ovo(args.manifest, args.run_output, limit=args.limit)
        return
    if args.command == "score-river":
        run_score_river(args.manifest, args.run_output, limit=args.limit)
        return
    if args.command == "score-streamingbench":
        run_score_streamingbench(args.manifest, args.run_output, limit=args.limit)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def run_demo(config_path: str, episode_path: str) -> None:
    config = load_config(config_path)
    episode = load_episode(
        episode_path,
        max_frames_per_video=config.max_frames_per_video,
        sampling_fps=config.sampling_fps,
    )
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
    episode = load_episode(
        episode_path,
        max_frames_per_video=config.max_frames_per_video,
        sampling_fps=config.sampling_fps,
    )
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
    metadata_episodes = load_manifest(
        manifest_path,
        limit=limit,
        metadata_only=True,
    )
    output_path = Path(output) if output is not None else None
    decisions_by_episode: dict[str, list[dict]] = {}
    metadata_episode_ids = {episode.episode_id for episode in metadata_episodes}
    if output_path is not None and output_path.exists():
        try:
            existing_output = load_run_output(output_path)
        except json.JSONDecodeError:
            print(
                json.dumps(
                    {"warning": f"Ignoring incomplete run output: {str(output_path)}"},
                    ensure_ascii=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        else:
            decisions_by_episode.update(
                {
                    episode_id: decisions
                    for episode_id, decisions in existing_output.get("decisions", {}).items()
                    if episode_id in metadata_episode_ids
                }
            )

    completed_episode_ids = set(decisions_by_episode)
    total_episodes = len(metadata_episodes)
    episodes_to_run = sum(1 for episode in metadata_episodes if episode.episode_id not in completed_episode_ids)

    episode_iterator = iter_manifest(
        manifest_path,
        limit=limit,
        max_frames_per_video=config.max_frames_per_video,
        sampling_fps=config.sampling_fps,
        skip_episode_ids=completed_episode_ids,
    )
    print(
        json.dumps(
            {
                "episodes_loaded": total_episodes,
                "episodes_completed": len(completed_episode_ids),
                "episodes_to_run": episodes_to_run,
                "sampling_fps": config.sampling_fps,
                "max_frames_per_video": config.max_frames_per_video,
            },
            ensure_ascii=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    agent = WatchRetrieveRespondAgent(build_backend(config), config)

    iterator = _progress(episode_iterator, desc="WRR episodes", total=episodes_to_run)
    for episode in iterator:
        if episode.episode_id in completed_episode_ids:
            continue
        result = agent.run_episode(episode, show_progress=True)
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
        completed_episode_ids.add(episode.episode_id)
        if output_path is not None:
            _write_manifest_run_output(
                output_path,
                metadata_episodes,
                decisions_by_episode,
                score_mode=score_mode,
                completed_episodes=len(completed_episode_ids),
                total_episodes=total_episodes,
                include_benchmark_score=False,
            )
        del result
        del episode
        _release_runtime_memory()

    run_output, benchmark_report = _build_manifest_run_output(
        metadata_episodes,
        decisions_by_episode,
        score_mode=score_mode,
        completed_episodes=len(completed_episode_ids),
        total_episodes=total_episodes,
        include_benchmark_score=True,
    )

    if benchmark_report is not None:
        print(benchmark_report.text)
    else:
        print(json.dumps(run_output["summary"], indent=2, ensure_ascii=True))

    if output_path is not None:
        _write_json(output_path, run_output)


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
    episodes = load_manifest(
        manifest_path,
        limit=limit,
        max_frames_per_video=config.max_frames_per_video,
        sampling_fps=config.sampling_fps,
    )
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


def run_score_streamingbench(manifest_path: str, run_output_path: str, limit: int | None = None) -> None:
    from .scoring import score_streamingbench

    episodes = load_manifest(manifest_path, limit=limit, metadata_only=True)
    run_output = load_run_output(run_output_path)
    report = score_streamingbench(episodes, run_output)
    print(report.text)


def _write_manifest_run_output(
    output_path: Path,
    episodes,
    decisions_by_episode: dict[str, list[dict]],
    score_mode: str,
    completed_episodes: int,
    total_episodes: int,
    include_benchmark_score: bool,
) -> None:
    run_output, _ = _build_manifest_run_output(
        episodes,
        decisions_by_episode,
        score_mode=score_mode,
        completed_episodes=completed_episodes,
        total_episodes=total_episodes,
        include_benchmark_score=include_benchmark_score,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, run_output)


def _build_manifest_run_output(
    episodes,
    decisions_by_episode: dict[str, list[dict]],
    score_mode: str,
    completed_episodes: int,
    total_episodes: int,
    include_benchmark_score: bool,
):
    decision_objects_by_episode = _decision_dicts_to_records(decisions_by_episode)
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
        "status": {
            "completed_episodes": completed_episodes,
            "total_episodes": total_episodes,
        },
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
    elif effective_score_mode == "streamingbench":
        from .scoring import score_streamingbench

        benchmark_report = score_streamingbench(episodes, run_output)

    if include_benchmark_score and benchmark_report is not None:
        run_output["benchmark_score"] = {
            "benchmark": benchmark_report.benchmark,
            "metrics": benchmark_report.metrics,
            "breakdown": benchmark_report.breakdown,
        }
    return run_output, benchmark_report


def _decision_dicts_to_records(decisions_by_episode: dict[str, list[dict]]) -> dict[str, list[DecisionRecord]]:
    records_by_episode: dict[str, list[DecisionRecord]] = {}
    for episode_id, decisions in decisions_by_episode.items():
        records_by_episode[episode_id] = [
            DecisionRecord(
                episode_id=episode_id,
                query_id=decision["query_id"],
                timestamp=float(decision["timestamp"]),
                action=Action(decision["action"]),
                answer=decision.get("answer"),
                confidence=float(decision.get("confidence", 0.0)),
                used_retrieval=bool(decision.get("used_retrieval", False)),
                reason=str(decision.get("reason", "")),
                retrieved_event_ids=list(decision.get("retrieved_event_ids", [])),
            )
            for decision in decisions
        ]
    return records_by_episode


def _release_runtime_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    tmp_path.replace(path)


def _progress(items, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, total=total)


if __name__ == "__main__":
    main()
