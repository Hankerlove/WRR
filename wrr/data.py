from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .media import sample_frame_paths, sample_video_frames
from .types import Episode, FrameObservation, Query, QueryType


def load_episode(path: str | Path) -> Episode:
    raw = json.loads(Path(path).read_text())
    return episode_from_dict(raw, base_dir=Path(path).resolve().parent)


def load_manifest(path: str | Path, limit: int | None = None, metadata_only: bool = False) -> list[Episode]:
    manifest_path = Path(path)
    episodes: list[Episode] = []
    with manifest_path.open() as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            episodes.append(episode_from_dict(raw, base_dir=manifest_path.resolve().parent, metadata_only=metadata_only))
    return episodes


def save_manifest(path: str | Path, episodes: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=True) + "\n")


def episode_from_dict(raw: dict, base_dir: Path | None = None, metadata_only: bool = False) -> Episode:
    frames = _load_frames(raw, base_dir=base_dir, metadata_only=metadata_only)
    queries = [
        Query(
            query_id=query["query_id"],
            text=query["text"],
            timestamp=float(query["timestamp"]),
            query_type=QueryType(query.get("query_type", "retro")),
            target_answer=_maybe_to_string(query.get("target_answer")),
            response_window=_maybe_window(query.get("response_window")),
            metadata=query.get("metadata", {}),
        )
        for query in raw["queries"]
    ]
    video_path = raw.get("video_path")
    if video_path is not None and base_dir is not None:
        video_path = _resolve_relative_path(video_path, base_dir)
    return Episode(
        episode_id=raw["episode_id"],
        frames=frames,
        queries=queries,
        video_path=video_path,
        metadata=raw.get("metadata", {}),
    )


def _maybe_window(raw_window: list[float] | None) -> tuple[float, float] | None:
    if raw_window is None:
        return None
    return float(raw_window[0]), float(raw_window[1])


def _maybe_to_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _load_frames(raw: dict, base_dir: Path | None, metadata_only: bool = False) -> list[FrameObservation]:
    if metadata_only:
        return []
    if "frames" in raw:
        frames: list[FrameObservation] = []
        for frame in raw["frames"]:
            image_path = frame.get("image_path")
            if image_path is not None and base_dir is not None:
                image_path = _resolve_relative_path(image_path, base_dir)
            frames.append(
                FrameObservation(
                    timestamp=float(frame["timestamp"]),
                    caption=frame.get("caption", ""),
                    image_path=image_path,
                    frame_index=frame.get("frame_index"),
                    source_video=frame.get("source_video"),
                    metadata=frame.get("metadata", {}),
                )
            )
        return frames

    if "frame_paths" in raw:
        frame_paths = raw["frame_paths"]
        if base_dir is not None:
            frame_paths = [_resolve_relative_path(path, base_dir) for path in frame_paths]
        timestamps = raw.get("frame_timestamps")
        return sample_frame_paths(frame_paths, timestamps=timestamps)

    if "video_path" in raw:
        if base_dir is None:
            raise ValueError("base_dir is required to resolve a manifest video path.")
        video_path = _resolve_relative_path(raw["video_path"], base_dir)
        return sample_video_frames(
            video_path=video_path,
            sampling_fps=float(raw.get("sampling_fps", 1.0)),
            max_frames=raw.get("max_frames"),
            start_time=float(raw.get("start_time", 0.0)),
            end_time=raw.get("end_time"),
        )

    raise ValueError("Episode data must include one of: frames, frame_paths, or video_path.")


def _resolve_relative_path(path: str, base_dir: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())
