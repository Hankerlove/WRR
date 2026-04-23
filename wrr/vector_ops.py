from __future__ import annotations

import hashlib
import math
import re

from .types import Vector

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize(vec: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0:
        return [0.0 for _ in vec]
    return [value / norm for value in vec]


def average(vectors: list[Vector]) -> Vector:
    if not vectors:
        return []
    width = len(vectors[0])
    out = [0.0] * width
    for vec in vectors:
        for idx, value in enumerate(vec):
            out[idx] += value
    return [value / len(vectors) for value in out]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(normalize(left), normalize(right)))


def l2_distance(left: Vector, right: Vector) -> float:
    if not left or not right:
        return 0.0
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right)))


def keyword_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def text_to_embedding(text: str, dim: int = 64) -> Vector:
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    return normalize(vec)
