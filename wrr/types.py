from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Vector = list[float]


class QueryType(str, Enum):
    RETRO = "retro"
    LIVE = "live"
    PROACTIVE = "proactive"


class Action(str, Enum):
    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    RESPOND = "RESPOND"


@dataclass(slots=True)
class FrameObservation:
    timestamp: float
    caption: str = ""
    image_path: str | None = None
    frame_index: int | None = None
    source_video: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Query:
    query_id: str
    text: str
    timestamp: float
    query_type: QueryType = QueryType.RETRO
    target_answer: str | None = None
    response_window: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Episode:
    episode_id: str
    frames: list[FrameObservation]
    queries: list[Query]
    video_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WindowState:
    timestamp: float
    summary: str
    embedding: Vector
    frame_count: int
    recent_captions: list[str]
    frames: list[FrameObservation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventSlot:
    event_id: str
    summary: str
    embedding: Vector
    start_time: float
    end_time: float
    salience: float
    support_count: int = 1


@dataclass(slots=True)
class RetrievalHit:
    event: EventSlot
    score: float


@dataclass(slots=True)
class AnswerProposal:
    answer: str | None
    confidence: float
    source: str


@dataclass(slots=True)
class AgentDecision:
    action: Action
    answer: str | None
    confidence: float
    used_retrieval: bool
    reason: str
    retrieved_event_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DecisionRecord:
    episode_id: str
    query_id: str
    timestamp: float
    action: Action
    answer: str | None
    confidence: float
    used_retrieval: bool
    reason: str
    retrieved_event_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryRuntimeState:
    query: Query
    wait_steps: int = 0
    last_answer: str | None = None
    consecutive_stable_steps: int = 0
    answered: bool = False
    answer_time: float | None = None


@dataclass(slots=True)
class EpisodeResult:
    episode_id: str
    decisions: list[DecisionRecord]
