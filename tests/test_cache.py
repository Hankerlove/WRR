from __future__ import annotations

import unittest

from wrr.config import CacheConfig
from wrr.memory import EventCache
from wrr.types import WindowState


class EventCacheTest(unittest.TestCase):
    def test_event_cache_writes_and_retrieves(self) -> None:
        cache = EventCache(
            CacheConfig(
                max_events=4,
                top_k=2,
                write_threshold=0.01,
                novelty_threshold=0.01,
                merge_threshold=0.95,
            )
        )
        first = WindowState(
            timestamp=0.0,
            summary="red cup on the table",
            embedding=[1.0, 0.0, 0.0],
            frame_count=1,
            recent_captions=["red cup on the table"],
        )
        second = WindowState(
            timestamp=1.0,
            summary="person opens the fridge",
            embedding=[0.0, 1.0, 0.0],
            frame_count=1,
            recent_captions=["person opens the fridge"],
        )
        cache.consider_write(first)
        cache.consider_write(second)
        hits = cache.retrieve([1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].event.summary, "red cup on the table")


if __name__ == "__main__":
    unittest.main()
