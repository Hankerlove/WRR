from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wrr.data import load_manifest, save_manifest
from wrr.scoring import score_ovo, score_river


class BenchmarkScoringTest(unittest.TestCase):
    def test_score_ovo_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "ovo_manifest.jsonl"
            save_manifest(
                manifest_path,
                [
                    {
                        "episode_id": "ovo-backward",
                        "frames": [{"timestamp": 0.0, "caption": "red cup"}],
                        "queries": [
                            {
                                "query_id": "q1",
                                "text": "What color is the cup?",
                                "timestamp": 0.0,
                                "query_type": "retro",
                                "target_answer": "A",
                                "metadata": {
                                    "benchmark": "ovo",
                                    "task": "ASI",
                                    "accepted_answers": ["A", "red", "A. red"],
                                },
                            }
                        ],
                        "metadata": {"benchmark": "ovo", "task": "ASI"},
                    },
                    {
                        "episode_id": "ovo-forward",
                        "frames": [{"timestamp": 0.0, "caption": "open the fridge"}],
                        "queries": [
                            {
                                "query_id": "q2",
                                "text": "Is the person performing this step?",
                                "timestamp": 0.0,
                                "query_type": "proactive",
                                "target_answer": "Yes",
                                "response_window": [1.0, 2.0],
                                "metadata": {
                                    "benchmark": "ovo",
                                    "task": "SSR",
                                    "accepted_answers": ["Yes", "Y"],
                                },
                            }
                        ],
                        "metadata": {"benchmark": "ovo", "task": "SSR"},
                    },
                ],
            )
            episodes = load_manifest(manifest_path)
            run_output = {
                "decisions": {
                    "ovo-backward": [
                        {"query_id": "q1", "action": "RESPOND", "answer": "red", "timestamp": 0.0}
                    ],
                    "ovo-forward": [
                        {"query_id": "q2", "action": "RESPOND", "answer": "Yes", "timestamp": 1.2}
                    ],
                }
            }
            report = score_ovo(episodes, run_output)
            self.assertIn("Backward Avg.", report.text)
            self.assertIn("Forward Avg.", report.text)
            self.assertGreaterEqual(report.metrics["total_avg"], 0.0)

    def test_score_river_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "river_manifest.jsonl"
            save_manifest(
                manifest_path,
                [
                    {
                        "episode_id": "river-retro",
                        "frames": [{"timestamp": 0.0, "caption": "a red cup was placed on the table"}],
                        "queries": [
                            {
                                "query_id": "q1",
                                "text": "What color was the cup?",
                                "timestamp": 0.0,
                                "query_type": "retro",
                                "target_answer": "red",
                                "metadata": {
                                    "benchmark": "river",
                                    "task": "retro-memory",
                                    "score_mode": "oe",
                                    "accepted_answers": ["red"],
                                },
                            }
                        ],
                        "metadata": {"benchmark": "river", "task": "retro-memory"},
                    },
                    {
                        "episode_id": "river-live",
                        "frames": [{"timestamp": 1.0, "caption": "the fridge is open"}],
                        "queries": [
                            {
                                "query_id": "q2",
                                "text": "What is open right now?",
                                "timestamp": 1.0,
                                "query_type": "live",
                                "target_answer": "fridge",
                                "metadata": {
                                    "benchmark": "river",
                                    "task": "live-perception",
                                    "score_mode": "oe",
                                    "accepted_answers": ["fridge"],
                                },
                            }
                        ],
                        "metadata": {"benchmark": "river", "task": "live-perception"},
                    },
                    {
                        "episode_id": "river-pro",
                        "frames": [{"timestamp": 2.0, "caption": "the person opens the fridge"}],
                        "queries": [
                            {
                                "query_id": "q3",
                                "text": "Tell me when the person opens the fridge.",
                                "timestamp": 0.0,
                                "query_type": "proactive",
                                "target_answer": "opens the fridge",
                                "response_window": [2.0, 2.5],
                                "metadata": {
                                    "benchmark": "river",
                                    "task": "pro-response",
                                    "score_mode": "oe",
                                    "response_mode": "instant",
                                    "accepted_answers": ["opens the fridge"],
                                },
                            }
                        ],
                        "metadata": {"benchmark": "river", "task": "pro-response"},
                    },
                ],
            )
            episodes = load_manifest(manifest_path)
            run_output = {
                "decisions": {
                    "river-retro": [
                        {"query_id": "q1", "action": "RESPOND", "answer": "red", "timestamp": 0.0}
                    ],
                    "river-live": [
                        {"query_id": "q2", "action": "RESPOND", "answer": "fridge", "timestamp": 1.0}
                    ],
                    "river-pro": [
                        {"query_id": "q3", "action": "RESPOND", "answer": "opens the fridge", "timestamp": 2.1}
                    ],
                }
            }
            report = score_river(episodes, run_output)
            self.assertIn("Instant-Loc", report.text)
            self.assertIn("Retro Avg.", report.text)
            self.assertGreaterEqual(report.metrics["overall_avg"], 0.0)


if __name__ == "__main__":
    unittest.main()
