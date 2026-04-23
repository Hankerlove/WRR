from __future__ import annotations

import json
import os
from pathlib import Path

from ..data import save_manifest
from ..types import QueryType

BACKWARD_TASKS = {"EPM", "ASI", "HLD"}
REALTIME_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
FORWARD_TASKS = {"REC", "SSR", "CRR"}


def convert_ovo_annotations(
    annotation_path: str | Path,
    video_root: str | Path,
    output_path: str | Path,
    sampling_fps: float = 1.0,
    response_slack: float = 1.0,
) -> int:
    annotation_file = Path(annotation_path)
    video_dir = Path(video_root)
    records = json.loads(annotation_file.read_text())
    episodes: list[dict] = []

    for item in records:
        task = item["task"]
        video_path = _resolve_video_path(video_dir, item["video"])
        if task in BACKWARD_TASKS or task in REALTIME_TASKS:
            query_type = QueryType.RETRO if task in BACKWARD_TASKS else QueryType.LIVE
            query_time = float(item.get("realtime", 0.0))
            accepted_answers = _mc_accepted_answers(item["options"], int(item["gt"]))
            question_text = _format_mc_question(item["question"], item["options"])
            episodes.append(
                {
                    "episode_id": str(item["id"]),
                    "video_path": video_path,
                    "sampling_fps": sampling_fps,
                    "queries": [
                        {
                            "query_id": f"{item['id']}-0",
                            "text": question_text,
                            "timestamp": query_time,
                            "query_type": query_type.value,
                            "target_answer": accepted_answers[0],
                            "response_window": [query_time, query_time + response_slack],
                            "metadata": {
                                "benchmark": "ovo",
                                "task": task,
                                "score_mode": "mc",
                                "accepted_answers": accepted_answers,
                                "question": item["question"],
                                "options": item["options"],
                                "ground_truth_index": int(item["gt"]),
                            },
                        }
                    ],
                    "metadata": {
                        "benchmark": "ovo",
                        "task": task,
                        "source_video": item["video"],
                    },
                }
            )
            continue

        if task not in FORWARD_TASKS:
            continue

        for index, test_info in enumerate(item.get("test_info", [])):
            accepted_answers, prompt_text = _build_forward_query(task, item, test_info)
            query_time = 0.0
            target_time = float(test_info.get("realtime", 0.0))
            episodes.append(
                {
                    "episode_id": f"{item['id']}-{index}",
                    "video_path": video_path,
                    "sampling_fps": sampling_fps,
                    "queries": [
                        {
                            "query_id": f"{item['id']}-{index}-0",
                            "text": prompt_text,
                            "timestamp": query_time,
                            "query_type": QueryType.PROACTIVE.value,
                            "target_answer": accepted_answers[0],
                            "response_window": [target_time, target_time + response_slack],
                            "metadata": {
                                "benchmark": "ovo",
                                "task": task,
                                "score_mode": "count" if task == "REC" else "binary",
                                "response_mode": "forward",
                                "accepted_answers": accepted_answers,
                                "raw_test_info": test_info,
                            },
                        }
                    ],
                    "metadata": {
                        "benchmark": "ovo",
                        "task": task,
                        "source_video": item["video"],
                        "parent_id": item["id"],
                    },
                }
            )

    save_manifest(output_path, episodes)
    return len(episodes)


def _resolve_video_path(video_root: Path, relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str((video_root / path).resolve())


def _format_mc_question(question: str, options: list[str]) -> str:
    formatted = "; ".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options))
    return (
        f"Question: {question}\n"
        f"Options: {formatted}\n"
        "Respond with the best option. You may answer with the letter or the option text."
    )


def _mc_accepted_answers(options: list[str], gt_index: int) -> list[str]:
    option = options[gt_index]
    letter = chr(65 + gt_index)
    return [letter, option, f"{letter}. {option}"]


def _build_forward_query(task: str, item: dict, test_info: dict) -> tuple[list[str], str]:
    if task == "REC":
        count = str(test_info["count"])
        prompt = (
            f"You are watching a video. Count how many times the people in the video perform this action: "
            f"{item['activity']}. Answer with a single number."
        )
        return [count], prompt
    if task == "SSR":
        target = "Yes" if int(test_info["type"]) == 1 else "No"
        prompt = (
            "You are watching a tutorial video. "
            f"Is the person currently performing this step: {test_info['step']}? Answer Yes or No."
        )
        return [target, target[0]], prompt
    if task == "CRR":
        target = "Yes" if int(test_info["type"]) == 1 else "No"
        prompt = (
            "Decide whether the latest video evidence is sufficient to answer the following question. "
            f"{item['question']} Answer Yes or No."
        )
        return [target, target[0]], prompt
    raise ValueError(f"Unsupported OVO forward task: {task}")
