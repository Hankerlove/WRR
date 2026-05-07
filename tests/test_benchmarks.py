from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wrr.benchmarks.ovo import convert_ovo_annotations
from wrr.benchmarks.streamingbench import convert_streamingbench_annotations


class BenchmarkAdapterTest(unittest.TestCase):
    def test_ovo_converter_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            annotation_path = root / "ovo.json"
            video_root = root / "videos"
            video_root.mkdir(parents=True, exist_ok=True)
            raw = [
                {
                    "id": "sample-1",
                    "video": "clip.mp4",
                    "task": "ASI",
                    "question": "What color is the cup?",
                    "options": ["red", "blue", "green", "yellow"],
                    "gt": 0,
                    "realtime": 3.0,
                },
                {
                    "id": "sample-2",
                    "video": "clip.mp4",
                    "task": "SSR",
                    "question": "Is the person done?",
                    "test_info": [
                        {
                            "step": "open the fridge",
                            "type": 1,
                            "realtime": 4.0,
                        }
                    ],
                },
            ]
            annotation_path.write_text(json.dumps(raw))
            output_path = root / "ovo_manifest.jsonl"
            count = convert_ovo_annotations(annotation_path, video_root, output_path, sampling_fps=1.0)
            self.assertEqual(count, 2)
            lines = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["queries"][0]["metadata"]["task"], "ASI")
            self.assertEqual(lines[1]["queries"][0]["metadata"]["task"], "SSR")

    def test_streamingbench_converter_groups_queries_by_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            annotation_path = root / "Real_Time_Visual_Understanding.csv"
            video_root = root / "videos"
            video_root.mkdir(parents=True, exist_ok=True)
            (video_root / "sample_1.mp4").write_bytes(b"")
            annotation_path.write_text(
                "\n".join(
                    [
                        "question_id,task_type,question,time_stamp,answer,options,frames_required,temporal_clue_type",
                        "\"Object Tracking_sample_1_1\",\"Object Tracking\",\"What is on the table?\",\"00:00:05\",\"A\",\"['A. cup', 'B. plate', 'C. bowl', 'D. fork']\",\"single\",\"Current\"",
                        "\"Object Tracking_sample_1_2\",\"Object Tracking\",\"What is still on the table?\",\"00:00:10\",\"B\",\"['A. cup', 'B. plate', 'C. bowl', 'D. fork']\",\"single\",\"Current\"",
                    ]
                )
            )
            output_path = root / "streamingbench_manifest.jsonl"
            count = convert_streamingbench_annotations(annotation_path, video_root, output_path, sampling_fps=1.0)
            self.assertEqual(count, 1)
            lines = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(len(lines[0]["queries"]), 2)
            self.assertEqual(lines[0]["metadata"]["benchmark"], "streamingbench")
            self.assertEqual(lines[0]["queries"][0]["metadata"]["subset"], "real_time_visual_understanding")


if __name__ == "__main__":
    unittest.main()
