# Watch, Retrieve, Respond: An Event-Driven Video Agent for Real-Time Video Understanding

`WRR/` is the codebase for the paper project:

> **Watch, Retrieve, Respond: An Event-Driven Video Agent for Real-Time Video Understanding**

This repository is built around a **usable research codebase** rather than a paper-only scaffold. The current implementation includes:

- a recent-window watcher,
- a sparse event cache,
- query-conditioned retrieval,
- a controller over `WAIT / RETRIEVE / RESPOND`,
- a deterministic mock backend for testing,
- a real open-source VLM backend based on `llava-hf/LLaVA-NeXT-Video-7B-hf`,
- local video sampling from `.mp4`,
- manifest-based dataset loading,
- OVO-Bench, RIVER, and StreamingBench manifest preparation commands,
- a one-stage controller-gate training path.

## Project Layout

```text
WRR/
├── configs/
│   ├── default.json
│   ├── demo.json
│   ├── llava_next_video.json
│   ├── llava_next_video_learned.json
│   └── river_field_map.example.json
├── examples/
│   └── demo_episode.json
├── tests/
│   ├── test_agent.py
│   └── test_cache.py
├── wrr/
│   ├── __init__.py
│   ├── agent.py
│   ├── backend.py
│   ├── benchmarks/
│   ├── cli.py
│   ├── config.py
│   ├── controller.py
│   ├── data.py
│   ├── eval.py
│   ├── media.py
│   ├── memory.py
│   ├── retriever.py
│   ├── train.py
│   ├── types.py
│   ├── vector_ops.py
│   └── watcher.py
├── pyproject.toml
└── README.md
```

## Design Choices

The code follows the lighter version of the method:

- **No multi-stage training by default**
- **No reinforcement learning**
- **No manual annotation requirement**
- **Frozen-backbone mindset**
- **Manifest-first data flow**

Two backend modes are currently available:

1. `mock`: deterministic and fast, used for debugging and unit tests
2. `llava_next_video`: real open-source video-language inference through Hugging Face Transformers

## Install

```bash
cd /Users/hongao/Desktop/papers/WRR
python -m pip install -e .
```

## Quick Start

Run the toy demo:

```bash
python -m wrr.cli demo --config configs/demo.json --episode examples/demo_episode.json
```

Evaluate the toy demo:

```bash
python -m wrr.cli evaluate --config configs/demo.json --episode examples/demo_episode.json
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Real VLM Inference

The default real-backend config is [llava_next_video.json](/Users/hongao/Desktop/papers/WRR/configs/llava_next_video.json).

Run a prepared episode with the real VLM:

```bash
python -m wrr.cli demo \
  --config configs/llava_next_video.json \
  --episode /path/to/episode.json
```

Run a full manifest:

```bash
python -m wrr.cli run-manifest \
  --config configs/llava_next_video.json \
  --manifest data/ovo_manifest.jsonl \
  --score-mode auto \
  --output outputs/ovo_eval.json
```

## Dataset Preparation

### OVO-Bench

Convert official OVO annotations and source videos into WRR's internal manifest format:

```bash
python -m wrr.cli prepare-ovo \
  --annotations /path/to/OVO-Bench/data/ovo_bench_new.json \
  --video-root /path/to/OVO-Bench/data/src_videos \
  --output data/ovo_manifest.jsonl \
  --sampling-fps 1.0
```

### RIVER

Convert RIVER-style annotations into WRR's internal manifest format:

```bash
python -m wrr.cli prepare-river \
  --annotations /path/to/river_annotations.json \
  --video-root /path/to/videos \
  --output data/river_manifest.jsonl \
  --sampling-fps 1.0 \
  --field-map configs/river_field_map.example.json
```

The RIVER converter is intentionally generic because annotation keys can differ across local releases. If your local annotation schema differs, edit `configs/river_field_map.example.json`.

### StreamingBench

Convert StreamingBench annotations into WRR's internal manifest format:

```bash
python -m wrr.cli prepare-streamingbench \
  --annotations /path/to/StreamingBench/annotations_or_csv_dir \
  --video-root /path/to/StreamingBench/data \
  --output data/streamingbench_manifest.jsonl \
  --sampling-fps 1.0
```

`prepare-streamingbench` accepts a single `.csv` / `.json` / `.jsonl` annotation file or a directory that contains multiple annotation files. Video resolution is heuristic and recursive: if the annotation does not include a usable video filename, WRR will search `--video-root` for files such as `sample_12.mp4`.

## Training the Controller Gate

Train the lightweight learned gate on a prepared manifest:

```bash
python -m wrr.cli train-gate \
  --config configs/llava_next_video.json \
  --manifest data/ovo_manifest.jsonl \
  --output outputs/gates/wrr_gate.pt \
  --epochs 5 \
  --hidden-dim 32 \
  --learning-rate 1e-3
```

Then evaluate with the learned controller config:

```bash
python -m wrr.cli run-manifest \
  --config configs/llava_next_video_learned.json \
  --manifest data/ovo_manifest.jsonl \
  --score-mode auto \
  --output outputs/ovo_eval_learned.json
```

## Standalone Scoring

If you already have a `run-manifest` output JSON and only want to re-score it, use:

```bash
python -m wrr.cli score-ovo \
  --manifest data/ovo_manifest.jsonl \
  --run-output outputs/ovo_eval.json
```

```bash
python -m wrr.cli score-river \
  --manifest data/river_manifest.jsonl \
  --run-output outputs/river_eval.json
```

```bash
python -m wrr.cli score-streamingbench \
  --manifest data/streamingbench_manifest.jsonl \
  --run-output outputs/streamingbench_eval.json
```

`run-manifest --score-mode auto` will automatically print OVO-style, RIVER-style, or StreamingBench-style reports when the manifest metadata contains a consistent benchmark name.

## Recommended Server Workflow

1. Install the package with `python -m pip install -e .`
2. Prepare a benchmark manifest with `prepare-ovo`, `prepare-river`, or `prepare-streamingbench`
3. Run the heuristic system first with `run-manifest --score-mode auto`
4. Train the lightweight gate with `train-gate`
5. Re-run evaluation with `llava_next_video_learned.json`

## Current Status

This is a **usable research codebase**, not yet a final benchmark submission package. The main advantage is that the code structure already matches the paper logic:

- `watch`: recent-window state management
- `retrieve`: event memory lookup
- `respond`: evidence-control policy

That makes it straightforward to iterate from heuristic control to learned gating or to add benchmark-specific scoring without rewriting the whole project.
