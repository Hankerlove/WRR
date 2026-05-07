from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CacheConfig:
    max_events: int = 32
    top_k: int = 4
    write_threshold: float = 0.30
    novelty_threshold: float = 0.18
    merge_threshold: float = 0.82


@dataclass(slots=True)
class ControllerConfig:
    policy: str = "heuristic"
    checkpoint_path: str | None = None
    retrieve_threshold: float = 0.28
    respond_threshold: float = 0.48
    proactive_threshold: float = 0.56
    improvement_margin: float = 0.04
    stability_confidence: float = 0.40
    min_stable_steps: int = 1
    max_wait_steps: int = 12


@dataclass(slots=True)
class BackendConfig:
    type: str = "mock"
    model_name: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    device_map: str = "auto"
    max_memory: dict[str, str] | None = None
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "float16"
    num_video_frames: int = 8
    max_new_tokens: int = 48
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False
    summary_prompt: str = "Summarize the latest visual content in one short sentence."
    answer_system_prompt: str = "You are a precise video understanding assistant. Answer using only the evidence visible in the provided video frames and context."


@dataclass(slots=True)
class AgentConfig:
    project_title: str
    embedding_dim: int = 64
    window_size: int = 3
    debug: bool = True
    sampling_fps: float = 1.0
    max_frames_per_video: int | None = None
    cache: CacheConfig = field(default_factory=CacheConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)


def load_config(path: str | Path) -> AgentConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    cache = CacheConfig(**raw.get("cache", {}))
    controller = ControllerConfig(**raw.get("controller", {}))
    backend = BackendConfig(**raw.get("backend", {}))
    return AgentConfig(
        project_title=raw.get("project_title", "WRR"),
        embedding_dim=raw.get("embedding_dim", 64),
        window_size=raw.get("window_size", 3),
        debug=raw.get("debug", True),
        sampling_fps=raw.get("sampling_fps", 1.0),
        max_frames_per_video=raw.get("max_frames_per_video"),
        cache=cache,
        controller=controller,
        backend=backend,
    )
