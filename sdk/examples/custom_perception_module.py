"""
Example: Custom Perception Module

Shows how to implement the PerceptionModule protocol with a simple
keyword-based fallback (no LLM required).

In production, the default PerceptionModule calls an LLM to produce
a rich structured encoding. This example uses keyword heuristics
to demonstrate the interface contract without any dependencies.

Usage::

    from examples.custom_perception_module import KeywordPerceptionModule
    from archotec_sdk.types import Observation, CycleContext

    module = KeywordPerceptionModule()
    obs = Observation(raw="I need help urgently, I'm in crisis")
    ctx = CycleContext(observation=obs)

    import asyncio
    output = asyncio.run(module.perceive(obs, ctx))
    print(output.emotional_state)     # {"distress": 0.9, "valence": -0.7, ...}
    print(output.epistemic_uncertainty)  # float
"""

from __future__ import annotations

import re
from typing import Dict, List

# SDK imports only — no private src/ imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from archotec_types import Observation, CycleContext, PerceptionOutput


# ---------------------------------------------------------------------------
# Keyword signal tables
# ---------------------------------------------------------------------------

_DISTRESS_WORDS = {
    "crisis", "urgent", "help", "panic", "emergency", "scared",
    "desperate", "pain", "suffering", "danger", "hurt", "can't",
}
_POSITIVE_WORDS = {
    "great", "wonderful", "happy", "excited", "love", "amazing",
    "fantastic", "excellent", "perfect", "joy", "celebrate",
}
_QUESTION_WORDS = {"what", "how", "why", "when", "where", "who", "which", "?"}
_TRADE_WORDS = {"bitcoin", "eth", "trade", "price", "buy", "sell", "binance", "crypto"}
_SEARCH_WORDS = {"search", "find", "news", "latest", "current", "today"}
_SYSTEM_WORDS = {"cpu", "ram", "disk", "memory", "process", "system", "computer"}


class KeywordPerceptionModule:
    """
    Simple keyword-based perception module.

    Implements the PerceptionModule protocol without requiring an LLM.
    Useful for testing, offline mode, or as a fallback.

    This is deliberately simple — the production implementation uses an
    LLM to produce far richer structured representations.
    """

    def __init__(self, distress_threshold: float = 0.3):
        """
        Args:
            distress_threshold: Minimum keyword density to flag distress.
        """
        self.distress_threshold = distress_threshold
        self._learn_count = 0
        self._distress_history: List[float] = []

    async def perceive(
        self,
        observation: Observation,
        ctx: CycleContext,
    ) -> PerceptionOutput:
        """
        Encode raw observation into structured PerceptionOutput.

        Uses keyword matching to estimate distress, valence, intent, and concepts.
        """
        text = observation.raw.lower()
        words = set(re.findall(r"\w+", text))

        # ── Emotional state ──────────────────────────────────────────────────
        distress_hits = len(words & _DISTRESS_WORDS)
        positive_hits = len(words & _POSITIVE_WORDS)

        distress = min(1.0, distress_hits * 0.3)
        valence  = min(1.0, positive_hits * 0.25) - distress * 0.5
        arousal  = min(1.0, (distress_hits + positive_hits) * 0.2)

        emotional_state = {
            "distress": round(distress, 3),
            "valence":  round(max(-1.0, min(1.0, valence)), 3),
            "arousal":  round(arousal, 3),
        }

        # ── Intent distribution ───────────────────────────────────────────────
        intent_scores: Dict[str, float] = {
            "information_request": 0.1,
            "task_execution":      0.1,
            "emotional_support":   0.1,
            "social_interaction":  0.1,
        }

        if words & _QUESTION_WORDS:
            intent_scores["information_request"] += 0.5
        if words & (_TRADE_WORDS | _SEARCH_WORDS | _SYSTEM_WORDS):
            intent_scores["task_execution"] += 0.5
        if distress > 0.3:
            intent_scores["emotional_support"] += 0.5
        if any(w in text for w in ["hello", "hi", "hey", "how are you"]):
            intent_scores["social_interaction"] += 0.5

        total = sum(intent_scores.values())
        intent_distribution = {k: v / total for k, v in intent_scores.items()}

        # ── Concepts ──────────────────────────────────────────────────────────
        concepts: List[str] = []
        if words & _TRADE_WORDS:
            concepts.append("trading")
        if words & _SEARCH_WORDS:
            concepts.append("web_search")
        if words & _SYSTEM_WORDS:
            concepts.append("computer_control")
        if words & _DISTRESS_WORDS:
            concepts.append("distress")

        # ── Epistemic uncertainty ──────────────────────────────────────────────
        # Short or ambiguous inputs → high uncertainty
        word_count = len(observation.raw.split())
        base_uncertainty = max(0.3, 1.0 - min(word_count / 20.0, 0.7))
        # If all intent scores are low → even more uncertain
        max_intent = max(intent_distribution.values())
        epistemic_uncertainty = base_uncertainty + (1.0 - max_intent) * 0.2
        epistemic_uncertainty = min(1.0, epistemic_uncertainty)

        return PerceptionOutput(
            observation=observation.raw,
            emotional_state=emotional_state,
            intent_distribution=intent_distribution,
            intent_uncertainty=round(1.0 - max_intent, 3),
            concepts=concepts,
            summary=f"[keyword] words={word_count}, distress={distress:.2f}",
            epistemic_uncertainty=round(epistemic_uncertainty, 3),
        )

    async def learn_from_outcome(
        self,
        observation: str,
        actual_distress: float,
        confidence: float,
        episode_id: str,
    ) -> None:
        """
        Update internal state from outcome feedback.

        In this simple implementation, we just track history.
        A real implementation would adjust keyword weights.
        """
        self._learn_count += 1
        self._distress_history.append(actual_distress)
        # Trim history
        if len(self._distress_history) > 100:
            self._distress_history = self._distress_history[-100:]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    module = KeywordPerceptionModule()

    test_cases = [
        "I need help urgently, I'm in crisis",
        "What is the current Bitcoin price?",
        "Show me CPU and RAM usage",
        "Hello! How are you doing today?",
        "?",
    ]

    async def demo():
        print("KeywordPerceptionModule Demo\n" + "=" * 40)
        for text in test_cases:
            obs = Observation(raw=text)
            ctx = CycleContext(observation=obs)
            out = await module.perceive(obs, ctx)
            print(f"\nInput:   {text!r}")
            print(f"Distress: {out.emotional_state.get('distress', 0):.2f}")
            print(f"Intent:   {max(out.intent_distribution, key=out.intent_distribution.get)}")
            print(f"Concepts: {out.concepts}")
            print(f"Uncertainty: {out.epistemic_uncertainty:.2f}")

    asyncio.run(demo())
