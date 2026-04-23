from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wrr.benchmarks.ovo import convert_ovo_annotations


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


if __name__ == "__main__":
    unittest.main()
