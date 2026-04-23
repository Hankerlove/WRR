from __future__ import annotations

import unittest
from pathlib import Path

from wrr.agent import WatchRetrieveRespondAgent
from wrr.backend import MockVideoLanguageModel
from wrr.config import load_config
from wrr.data import load_episode
from wrr.eval import evaluate_episode
from wrr.types import Action


class AgentIntegrationTest(unittest.TestCase):
    def test_demo_episode_produces_responses(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "demo.json")
        episode = load_episode(root / "examples" / "demo_episode.json")
        agent = WatchRetrieveRespondAgent(MockVideoLanguageModel(config.embedding_dim), config)
        result = agent.run_episode(episode)
        responses = [decision for decision in result.decisions if decision.action == Action.RESPOND]
        self.assertGreaterEqual(len(responses), 2)
        summary = evaluate_episode(episode, result.decisions)
        self.assertEqual(summary.total_queries, 3)
        self.assertGreater(summary.answered_queries, 0)


if __name__ == "__main__":
    unittest.main()
