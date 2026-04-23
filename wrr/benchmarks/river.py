from __future__ import annotations

import json
from pathlib import Path

from ..data import save_manifest
from ..types import QueryType

DEFAULT_FIELD_MAP = {
    "records_field": None,
    "episode_id": "id",
    "video_path": "video",
    "question": "question",
    "query_time": "query_time",
    "task": "task",
    "answer": "answer",
    "response_start": "response_start",
    "response_end": "response_end",
    "options": "options",
    "gt_index": "gt",
    "score_mode": "score_mode",
    "response_mode": "response_mode",
}


def convert_river_annotations(
    annotation_path: str | Path,
    video_root: str | Path,
    output_path: str | Path,
    sampling_fps: float = 1.0,
    field_map_path: str | Path | None = None,
    response_slack: float = 1.0,
) -> int:
    field_map = dict(DEFAULT_FIELD_MAP)
    if field_map_path is not None:
        field_map.update(json.loads(Path(field_map_path).read_text()))

    raw = json.loads(Path(annotation_path).read_text())
    records = raw
    if field_map["records_field"] is not None:
        records = raw[field_map["records_field"]]

    video_dir = Path(video_root)
    episodes: list[dict] = []
    for item in records:
        task_name = str(item.get(field_map["task"], "retro")).lower()
        query_type = _map_river_task(task_name)
        query_time = float(item.get(field_map["query_time"], 0.0))
        video_path = _resolve_video_path(video_dir, item[field_map["video_path"]])
        question = str(item[field_map["question"]])
        accepted_answers = _build_accepted_answers(item, field_map)
        prompt = _build_prompt(question, item, field_map)
        score_mode = _detect_score_mode(item, field_map)
        response_mode = _detect_response_mode(task_name, item, field_map)

        response_window = None
        if field_map["response_start"] in item or field_map["response_end"] in item:
            start = float(item.get(field_map["response_start"], query_time))
            end = float(item.get(field_map["response_end"], start + response_slack))
            response_window = [start, end]
        elif query_type == QueryType.PROACTIVE:
            response_window = [query_time, query_time + response_slack]

        episodes.append(
            {
                "episode_id": str(item.get(field_map["episode_id"], len(episodes))),
                "video_path": video_path,
                "sampling_fps": sampling_fps,
                "queries": [
                    {
                        "query_id": f"{item.get(field_map['episode_id'], len(episodes))}-0",
                        "text": prompt,
                        "timestamp": query_time,
                        "query_type": query_type.value,
                        "target_answer": accepted_answers[0] if accepted_answers else item.get(field_map["answer"]),
                        "response_window": response_window,
                        "metadata": {
                            "benchmark": "river",
                            "task": task_name,
                            "score_mode": score_mode,
                            "response_mode": response_mode,
                            "accepted_answers": accepted_answers,
                            "raw_item": item,
                        },
                    }
                ],
                "metadata": {
                    "benchmark": "river",
                    "task": task_name,
                },
            }
        )

    save_manifest(output_path, episodes)
    return len(episodes)


def _build_prompt(question: str, item: dict, field_map: dict[str, str | None]) -> str:
    options_key = field_map["options"]
    gt_key = field_map["gt_index"]
    if options_key is not None and options_key in item:
        options = item[options_key]
        formatted = "; ".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options))
        return (
            f"Question: {question}\n"
            f"Options: {formatted}\n"
            "Respond with the best option. You may answer with the letter or the option text."
        )
    return question


def _build_accepted_answers(item: dict, field_map: dict[str, str | None]) -> list[str]:
    options_key = field_map["options"]
    gt_key = field_map["gt_index"]
    answer_key = field_map["answer"]
    if options_key is not None and gt_key is not None and options_key in item and gt_key in item:
        options = item[options_key]
        gt_index = int(item[gt_key])
        letter = chr(65 + gt_index)
        option = options[gt_index]
        return [letter, option, f"{letter}. {option}"]
    if answer_key is not None and answer_key in item:
        answer = str(item[answer_key])
        return [answer]
    return []


def _detect_score_mode(item: dict, field_map: dict[str, str | None]) -> str:
    score_mode_key = field_map.get("score_mode")
    if score_mode_key is not None and score_mode_key in item:
        return str(item[score_mode_key]).lower()
    options_key = field_map["options"]
    if options_key is not None and options_key in item:
        return "mc"
    return "oe"


def _detect_response_mode(task_name: str, item: dict, field_map: dict[str, str | None]) -> str:
    response_mode_key = field_map.get("response_mode")
    if response_mode_key is not None and response_mode_key in item:
        return str(item[response_mode_key]).lower()
    if "stream" in task_name:
        return "streaming"
    return "instant"


def _map_river_task(task_name: str) -> QueryType:
    normalized = task_name.lower()
    if "retro" in normalized or "memory" in normalized:
        return QueryType.RETRO
    if "live" in normalized or "perception" in normalized:
        return QueryType.LIVE
    if "pro" in normalized or "active" in normalized or "response" in normalized:
        return QueryType.PROACTIVE
    return QueryType.RETRO


def _resolve_video_path(video_root: Path, relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str((video_root / path).resolve())
