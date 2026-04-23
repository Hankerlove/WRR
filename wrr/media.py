from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .types import FrameObservation


def sample_video_frames(
    video_path: str | Path,
    sampling_fps: float = 1.0,
    max_frames: int | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> list[FrameObservation]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Sampling videos requires opencv-python. Install it before loading real benchmark videos."
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    native_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if native_fps <= 0:
        native_fps = max(sampling_fps, 1.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / native_fps if native_fps > 0 else 0.0
    effective_end = duration if end_time is None else min(end_time, duration)
    frame_step = max(int(round(native_fps / max(sampling_fps, 1e-6))), 1)
    start_index = int(start_time * native_fps)
    end_index = int(effective_end * native_fps) if effective_end > 0 else total_frames

    observations: list[FrameObservation] = []
    frame_index = 0
    sampled = 0
    while frame_index < total_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index < start_index:
            frame_index += 1
            continue
        if frame_index > end_index:
            break
        if (frame_index - start_index) % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_index / native_fps
            observations.append(
                FrameObservation(
                    timestamp=timestamp,
                    caption="",
                    frame_index=frame_index,
                    source_video=str(path),
                    metadata={"image_array": rgb},
                )
            )
            sampled += 1
            if max_frames is not None and sampled >= max_frames:
                break
        frame_index += 1

    capture.release()
    return observations


def sample_frame_paths(frame_paths: Sequence[str | Path], timestamps: Sequence[float] | None = None) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    for idx, frame_path in enumerate(frame_paths):
        ts = float(timestamps[idx]) if timestamps is not None else float(idx)
        observations.append(
            FrameObservation(
                timestamp=ts,
                image_path=str(frame_path),
                frame_index=idx,
            )
        )
    return observations


def frame_to_rgb_array(frame: FrameObservation) -> np.ndarray:
    if "image_array" in frame.metadata:
        return np.asarray(frame.metadata["image_array"])
    if frame.image_path is None:
        raise ValueError("FrameObservation does not contain image data or an image path.")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Loading image frames requires Pillow.") from exc
    image = Image.open(frame.image_path).convert("RGB")
    return np.asarray(image)


def frames_to_video_clip(frames: Sequence[FrameObservation], num_frames: int | None = None) -> np.ndarray:
    selected = _select_uniform_frames(frames, num_frames)
    if not selected:
        return np.zeros((0, 0, 0, 0), dtype=np.uint8)
    arrays = [frame_to_rgb_array(frame) for frame in selected]
    return np.stack(arrays, axis=0)


def _select_uniform_frames(frames: Sequence[FrameObservation], num_frames: int | None) -> list[FrameObservation]:
    if not frames:
        return []
    if num_frames is None or len(frames) <= num_frames:
        return list(frames)
    positions = np.linspace(0, len(frames) - 1, num_frames).astype(int).tolist()
    return [frames[pos] for pos in positions]
