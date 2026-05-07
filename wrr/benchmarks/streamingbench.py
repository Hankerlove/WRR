from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

from ..data import save_manifest
from ..types import QueryType

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def convert_streamingbench_annotations(
    annotation_path: str | Path,
    video_root: str | Path,
    output_path: str | Path,
    sampling_fps: float = 1.0,
    response_slack: float = 1.0,
    subset_override: str | None = None,
) -> int:
    video_dir = Path(video_root)
    video_index = _build_video_index(video_dir)
    episodes_by_video: dict[str, dict] = {}
    resolver_cache: dict[str, str] = {}

    for item, source_name in _iter_records(Path(annotation_path)):
        subset = _normalize_subset(subset_override or item.get("subset") or source_name)
        query_id = str(item.get("question_id") or item.get("id") or f"q-{len(episodes_by_video)}")
        question = str(item["question"]).strip()
        task_type = str(item.get("task_type", subset)).strip()
        options = _parse_options(item.get("options"))
        accepted_answers = _build_accepted_answers(item.get("answer"), options)
        prompt = _build_prompt(question, options, subset)
        raw_timestamp = item.get("time_stamp", item.get("timestamp", item.get("time", "00:00:00")))
        target_time = _parse_timestamp(raw_timestamp)
        query_type, query_time, response_window = _build_query_schedule(subset, target_time, response_slack)
        video_path = _resolve_video_path(item, query_id, video_dir, resolver_cache, video_index)

        episode = episodes_by_video.get(video_path)
        if episode is None:
            episode = {
                "episode_id": _episode_id_from_video_path(video_path),
                "video_path": video_path,
                "sampling_fps": sampling_fps,
                "queries": [],
                "metadata": {
                    "benchmark": "streamingbench",
                    "source_video": Path(video_path).name,
                },
            }
            episodes_by_video[video_path] = episode

        episode["queries"].append(
            {
                "query_id": query_id,
                "text": prompt,
                "timestamp": query_time,
                "query_type": query_type.value,
                "target_answer": accepted_answers[0] if accepted_answers else None,
                "response_window": response_window,
                "metadata": {
                    "benchmark": "streamingbench",
                    "subset": subset,
                    "task": task_type,
                    "score_mode": "mc",
                    "accepted_answers": accepted_answers,
                    "question": question,
                    "options": options,
                    "frames_required": str(item.get("frames_required", "")).strip().lower(),
                    "temporal_clue_type": str(item.get("temporal_clue_type", "")).strip(),
                    "raw_timestamp": str(raw_timestamp),
                },
            }
        )

    episodes = list(episodes_by_video.values())
    for episode in episodes:
        episode["queries"].sort(key=lambda query: (float(query["timestamp"]), str(query["query_id"])))
    save_manifest(output_path, episodes)
    return len(episodes)


def _iter_records(annotation_path: Path) -> list[tuple[dict, str]]:
    if annotation_path.is_file():
        return _load_annotation_file(annotation_path)

    records: list[tuple[dict, str]] = []
    for file_path in sorted(annotation_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            continue
        records.extend(_load_annotation_file(file_path))
    if not records:
        raise ValueError(f"No supported annotation files were found under {annotation_path}.")
    return records


def _load_annotation_file(path: Path) -> list[tuple[dict, str]]:
    suffix = path.suffix.lower()
    source_name = _normalize_subset(path.stem)
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return [(dict(row), source_name) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows: list[tuple[dict, str]] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append((json.loads(line), source_name))
        return rows

    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            raw = raw["data"]
        elif "records" in raw and isinstance(raw["records"], list):
            raw = raw["records"]
        else:
            raw = [raw]
    return [(dict(item), source_name) for item in raw]


def _normalize_subset(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    alias_map = {
        "real_time_visual_understanding": "real_time_visual_understanding",
        "omni_source_understanding": "omni_source_understanding",
        "contextual_understanding": "contextual_understanding",
        "sequential_question_answering": "sequential_question_answering",
        "proactive_output": "proactive_output",
        "realtime_visual_understanding": "real_time_visual_understanding",
        "omni_source": "omni_source_understanding",
        "contextual": "contextual_understanding",
        "sqa": "sequential_question_answering",
        "proactive": "proactive_output",
        "real": "real_time_visual_understanding",
        "omni": "omni_source_understanding",
    }
    return alias_map.get(text, text)


def _parse_options(raw_options: object) -> list[str]:
    if raw_options is None:
        return []
    if isinstance(raw_options, list):
        return [str(option) for option in raw_options]
    text = str(raw_options).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(option) for option in parsed]
    except (SyntaxError, ValueError):
        pass
    return [part.strip() for part in text.split("||") if part.strip()]


def _build_accepted_answers(raw_answer: object, options: list[str]) -> list[str]:
    if raw_answer is None:
        return []
    answer = str(raw_answer).strip()
    if not answer:
        return []
    if len(answer) == 1 and answer.isalpha() and options:
        index = ord(answer.upper()) - ord("A")
        if 0 <= index < len(options):
            option_text = _strip_option_prefix(options[index])
            return [answer.upper(), option_text, options[index]]
    for index, option in enumerate(options):
        option_text = _strip_option_prefix(option)
        letter = chr(65 + index)
        if answer.upper() == letter or answer.lower() == option_text.lower():
            return [letter, option_text, option]
    return [answer]


def _build_prompt(question: str, options: list[str], subset: str) -> str:
    prefix = "You are watching a streaming video."
    if subset == "proactive_output":
        prefix += " Wait until the target event occurs, then answer immediately."
    if not options:
        return f"{prefix}\nQuestion: {question}"
    formatted = "; ".join(f"{chr(65 + idx)}. {_strip_option_prefix(option)}" for idx, option in enumerate(options))
    return (
        f"{prefix}\n"
        f"Question: {question}\n"
        f"Options: {formatted}\n"
        "Respond with the best option. You may answer with the letter or the option text."
    )


def _build_query_schedule(
    subset: str,
    target_time: float,
    response_slack: float,
) -> tuple[QueryType, float, list[float] | None]:
    if subset == "proactive_output":
        return QueryType.PROACTIVE, 0.0, [target_time, target_time + response_slack]
    if subset in {"contextual_understanding", "sequential_question_answering"}:
        return QueryType.RETRO, target_time, [target_time, target_time + response_slack]
    return QueryType.LIVE, target_time, [target_time, target_time + response_slack]


def _parse_timestamp(raw_value: object) -> float:
    if raw_value is None:
        return 0.0
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    text = str(raw_value).strip()
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(text)


def _resolve_video_path(
    item: dict,
    query_id: str,
    video_root: Path,
    resolver_cache: dict[str, str],
    video_index: dict[str, list[Path]],
) -> str:
    for key in ("video_path", "video", "video_name", "video_file", "video_id"):
        if key not in item or item[key] in {None, ""}:
            continue
        value = item[key]
        if isinstance(value, dict):
            value = value.get("path") or value.get("file_name") or value.get("name")
        if value in {None, ""}:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return str(path)
        candidate = (video_root / path).resolve()
        if candidate.exists():
            return str(candidate)

    candidate_keys = _candidate_video_keys(query_id)
    for key in candidate_keys:
        if key in resolver_cache:
            return resolver_cache[key]
        matches = _search_video_candidates(video_index, key)
        if matches:
            resolved = str(matches[0])
            resolver_cache[key] = resolved
            return resolved
    raise FileNotFoundError(f"Could not resolve a video file for StreamingBench query {query_id}.")


def _candidate_video_keys(query_id: str) -> list[str]:
    keys: list[str] = []
    sample_match = re.search(r"(sample_\d+)", query_id, flags=re.IGNORECASE)
    if sample_match is not None:
        keys.append(sample_match.group(1))
    if "_" in query_id:
        keys.append(query_id.rsplit("_", 1)[0])
    keys.append(query_id)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        normalized = key.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _build_video_index(video_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        resolved = path.resolve()
        for key in {path.stem, path.name}:
            index.setdefault(key, []).append(resolved)
    return index


def _search_video_candidates(video_index: dict[str, list[Path]], candidate_key: str) -> list[Path]:
    direct_matches = list(video_index.get(candidate_key, []))
    if not direct_matches:
        direct_matches = []
        for key, paths in video_index.items():
            if key.startswith(candidate_key):
                direct_matches.extend(paths)
    deduped = sorted({path for path in direct_matches}, key=lambda path: (len(str(path)), str(path)))
    return deduped


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-Da-d]\.\s*", "", option).strip()


def _episode_id_from_video_path(video_path: str) -> str:
    path = Path(video_path)
    parent = re.sub(r"\W+", "_", path.parent.name.strip().lower()).strip("_")
    stem = re.sub(r"\W+", "_", path.stem.strip().lower()).strip("_")
    if parent:
        return f"{parent}-{stem}"
    return stem
