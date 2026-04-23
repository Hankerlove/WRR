from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .config import AgentConfig, BackendConfig
from .media import frames_to_video_clip
from .types import AnswerProposal, FrameObservation, Query, RetrievalHit, WindowState
from .vector_ops import average, cosine_similarity, keyword_overlap, text_to_embedding


class BaseVideoBackend(ABC):
    @abstractmethod
    def encode_window(self, frames: Sequence[FrameObservation]) -> WindowState:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: Query) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def propose_answer(
        self,
        query: Query,
        current_state: WindowState,
        retrieved_hits: Sequence[RetrievalHit] = (),
    ) -> AnswerProposal:
        raise NotImplementedError


class MockVideoLanguageModel(BaseVideoBackend):
    """Deterministic mock backend used to make the pipeline runnable."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim = embedding_dim

    def encode_window(self, frames: Sequence[FrameObservation]) -> WindowState:
        captions = [frame.caption for frame in frames if frame.caption]
        summary = " | ".join(captions[-2:]) if captions else ""
        embedding = average([text_to_embedding(caption, self.embedding_dim) for caption in captions])
        if not embedding:
            embedding = [0.0] * self.embedding_dim
        timestamp = frames[-1].timestamp if frames else 0.0
        return WindowState(
            timestamp=timestamp,
            summary=summary,
            embedding=embedding,
            frame_count=len(frames),
            recent_captions=captions[-2:],
            frames=list(frames),
        )

    def embed_query(self, query: Query) -> list[float]:
        return text_to_embedding(query.text, self.embedding_dim)

    def propose_answer(
        self,
        query: Query,
        current_state: WindowState,
        retrieved_hits: Sequence[RetrievalHit] = (),
    ) -> AnswerProposal:
        candidates: list[tuple[str, str, float]] = []
        current_score = self._score_candidate(query.text, current_state.summary, current_state.embedding)
        if current_state.summary:
            candidates.append(("current", current_state.summary, current_score))
        for hit in retrieved_hits:
            score = self._score_candidate(query.text, hit.event.summary, hit.event.embedding)
            score = max(score, 0.55 * hit.score + 0.45 * score)
            candidates.append((hit.event.event_id, hit.event.summary, score))
        if not candidates:
            return AnswerProposal(answer=None, confidence=0.0, source="none")
        source, answer, score = max(candidates, key=lambda item: item[2])
        if score < 0.12:
            return AnswerProposal(answer=None, confidence=score, source=source)
        return AnswerProposal(answer=answer, confidence=score, source=source)

    def _score_candidate(self, query_text: str, candidate_text: str, embedding: list[float]) -> float:
        if not candidate_text:
            return 0.0
        query_embedding = text_to_embedding(query_text, self.embedding_dim)
        overlap = keyword_overlap(query_text, candidate_text)
        similarity = (cosine_similarity(query_embedding, embedding) + 1.0) / 2.0
        return 0.60 * overlap + 0.40 * similarity


class LlavaNextVideoBackend(BaseVideoBackend):
    """Open-source VLM backend built on top of Hugging Face LLaVA-NeXT-Video."""

    def __init__(self, config: BackendConfig, embedding_dim: int) -> None:
        self.config = config
        self.embedding_dim = embedding_dim
        self.model_name = config.model_name
        self._model = None
        self._processor = None

    def encode_window(self, frames: Sequence[FrameObservation]) -> WindowState:
        frame_list = list(frames)
        timestamp = frame_list[-1].timestamp if frame_list else 0.0
        captions = [frame.caption for frame in frame_list if frame.caption]

        if captions:
            summary = " | ".join(captions[-2:])
        elif frame_list:
            summary, _ = self._generate_text_from_frames(
                frame_list,
                user_prompt=self.config.summary_prompt,
                max_new_tokens=max(24, min(self.config.max_new_tokens, 48)),
            )
        else:
            summary = ""

        embedding = text_to_embedding(summary, self.embedding_dim) if summary else [0.0] * self.embedding_dim
        return WindowState(
            timestamp=timestamp,
            summary=summary,
            embedding=embedding,
            frame_count=len(frame_list),
            recent_captions=captions[-2:],
            frames=frame_list,
        )

    def embed_query(self, query: Query) -> list[float]:
        return text_to_embedding(query.text, self.embedding_dim)

    def propose_answer(
        self,
        query: Query,
        current_state: WindowState,
        retrieved_hits: Sequence[RetrievalHit] = (),
    ) -> AnswerProposal:
        if current_state.frames:
            answer_prompt = self._build_answer_prompt(query, retrieved_hits)
            answer, confidence = self._generate_text_from_frames(
                current_state.frames,
                user_prompt=answer_prompt,
                max_new_tokens=self.config.max_new_tokens,
            )
            answer = self._clean_text(answer)
            if not answer:
                return AnswerProposal(answer=None, confidence=confidence, source="llava_next_video")
            return AnswerProposal(answer=answer, confidence=confidence, source="llava_next_video")

        candidates: list[tuple[str, str, float]] = []
        if current_state.summary:
            current_score = self._score_candidate(query.text, current_state.summary)
            candidates.append(("current", current_state.summary, current_score))
        for hit in retrieved_hits:
            score = max(self._score_candidate(query.text, hit.event.summary), hit.score)
            candidates.append((hit.event.event_id, hit.event.summary, score))
        if not candidates:
            return AnswerProposal(answer=None, confidence=0.0, source="fallback")
        source, answer, score = max(candidates, key=lambda item: item[2])
        return AnswerProposal(answer=answer, confidence=score, source=source)

    def _build_answer_prompt(self, query: Query, retrieved_hits: Sequence[RetrievalHit]) -> str:
        lines = [self.config.answer_system_prompt]
        if retrieved_hits:
            lines.append("Retrieved historical events:")
            for idx, hit in enumerate(retrieved_hits, start=1):
                lines.append(f"{idx}. [{hit.event.start_time:.1f}s-{hit.event.end_time:.1f}s] {hit.event.summary}")
        lines.append(f"Question: {query.text}")
        lines.append("Answer briefly and directly.")
        return "\n".join(lines)

    def _generate_text_from_frames(
        self,
        frames: Sequence[FrameObservation],
        user_prompt: str,
        max_new_tokens: int,
    ) -> tuple[str, float]:
        if not frames:
            return "", 0.0
        self._ensure_loaded()

        import torch

        clip = frames_to_video_clip(frames, num_frames=self.config.num_video_frames)
        if clip.size == 0:
            return "", 0.0

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "video"},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(
            text=prompt,
            videos=clip,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self._model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.config.do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.config.do_sample:
            generate_kwargs["temperature"] = self.config.temperature
            generate_kwargs["top_p"] = self.config.top_p

        outputs = self._model.generate(
            **inputs,
            **generate_kwargs,
        )
        generated_tokens = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        confidence = 0.0
        if outputs.scores:
            transition_scores = self._model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )
            generated_scores = transition_scores[0, -generated_tokens.shape[1] :]
            if generated_scores.numel() > 0:
                confidence = float(torch.exp(generated_scores.mean()).detach().cpu().item())
        confidence = max(0.0, min(confidence, 1.0))
        return decoded.strip(), confidence

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        except ImportError as exc:
            raise ImportError(
                "The LLaVA-NeXT-Video backend requires torch and transformers>=4.49.0."
            ) from exc

        torch_dtype = getattr(torch, self.config.torch_dtype, torch.float16)
        self._model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model.eval()
        self._processor = LlavaNextVideoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._processor.tokenizer.padding_side = "left"
        if getattr(self._processor, "patch_size", None) is None and hasattr(self._model.config, "vision_config"):
            self._processor.patch_size = getattr(self._model.config.vision_config, "patch_size", None)
            self._processor.vision_feature_select_strategy = getattr(
                self._model.config,
                "vision_feature_select_strategy",
                None,
            )
            self._processor.num_additional_image_tokens = 1

    def _score_candidate(self, query_text: str, candidate_text: str) -> float:
        if not candidate_text:
            return 0.0
        overlap = keyword_overlap(query_text, candidate_text)
        similarity = (cosine_similarity(text_to_embedding(query_text, self.embedding_dim), text_to_embedding(candidate_text, self.embedding_dim)) + 1.0) / 2.0
        return 0.55 * overlap + 0.45 * similarity

    def _clean_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if cleaned.startswith("ASSISTANT:"):
            cleaned = cleaned.split("ASSISTANT:", 1)[1].strip()
        return cleaned


def build_backend(config: AgentConfig) -> BaseVideoBackend:
    backend_type = config.backend.type.lower()
    if backend_type == "mock":
        return MockVideoLanguageModel(config.embedding_dim)
    if backend_type == "llava_next_video":
        return LlavaNextVideoBackend(config.backend, config.embedding_dim)
    raise ValueError(f"Unsupported backend type: {config.backend.type}")
