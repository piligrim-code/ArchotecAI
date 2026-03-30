"""
Example: Custom Policy Module (EFE-based)

Shows how to implement the PolicyModule protocol with explicit
Expected Free Energy computation and stochastic sampling.

The key invariant: NEVER use argmax. NEVER use if/else on belief state.
Actions must emerge from the stochastic softmax distribution.

EFE(a) = pragmatic_value(a) + epistemic_value(a)

where:
    pragmatic_value(a) = sum(goal_weight × expected_goal_satisfaction(a))
    epistemic_value(a) = expected_uncertainty_reduction(a)

Selection: softmax(−EFE / temperature) → sample

Usage::

    from examples.custom_policy_module import EFEPolicyModule
    from archotec_sdk.types import BeliefSnapshot, CycleContext, Observation

    policy = EFEPolicyModule(temperature=1.0)
    belief = BeliefSnapshot(
        latent_state="exploration",
        epistemic_uncertainty=0.7,
        emotional_state={"distress": 0.1},
    )
    goals = {"EXPLORE_UNKNOWN": 0.8, "PROVIDE_INFORMATION": 0.5}
    ctx = CycleContext(observation=Observation(raw="What can you do?"))

    action, distribution = policy.select_action(belief, goals, ctx)
    print(action.action_type)        # stochastically sampled
    print(distribution.efe_values)   # EFE per action
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from archotec_types import (
    ActionCandidate,
    ActionDistributionSnapshot,
    BeliefSnapshot,
    CycleContext,
    Observation,
)


# ---------------------------------------------------------------------------
# Action catalogue (minimal public subset)
# ---------------------------------------------------------------------------

# action_type → (expected_epistemic_gain, goal_relevance)
# goal_relevance: which goals this action serves
_ACTION_CATALOGUE: Dict[str, Dict[str, float]] = {
    "respond":                  {"epistemic_gain": 0.1, "goal_info": 0.8, "goal_rapport": 0.6},
    "execute_capability":       {"epistemic_gain": 0.3, "goal_task": 0.9, "goal_explore": 0.4},
    "request_clarification":    {"epistemic_gain": 0.5, "goal_info": 0.5, "goal_uncertainty": 0.7},
    "explore_capability":       {"epistemic_gain": 0.6, "goal_explore": 0.9, "goal_learn": 0.7},
    "reflect_and_synthesize":   {"epistemic_gain": 0.2, "goal_info": 0.6, "goal_identity": 0.5},
    "delegate_subtask":         {"epistemic_gain": 0.3, "goal_task": 0.8, "goal_learn": 0.3},
}

# Map goal names to catalogue keys
_GOAL_TO_CATALOGUE_KEY: Dict[str, str] = {
    "PROVIDE_INFORMATION":   "goal_info",
    "EXPLORE_UNKNOWN":       "goal_explore",
    "LEARN_AND_ADAPT":       "goal_learn",
    "REDUCE_UNCERTAINTY":    "goal_uncertainty",
    "EXPAND_CAPABILITIES":   "goal_task",
    "BUILD_RAPPORT":         "goal_rapport",
    "MAINTAIN_IDENTITY":     "goal_identity",
}


class EFEPolicyModule:
    """
    Policy module that selects actions by minimizing Expected Free Energy.

    EFE(a) = pragmatic_value(a) + epistemic_value(a)

    Pragmatic value captures goal satisfaction.
    Epistemic value captures uncertainty reduction.

    Selection is stochastic: softmax(−EFE / temperature) → sample.
    This is NOT argmax. Higher temperature = more exploration.
    """

    def __init__(self, temperature: float = 1.0, exploration_boost: float = 0.1):
        """
        Args:
            temperature:       Softmax sampling temperature.
                               Low  (0.1) → near-deterministic (exploit)
                               High (2.0) → uniform exploration
            exploration_boost: Added epistemic value bonus (encourages exploration).
        """
        self.temperature = temperature
        self.exploration_boost = exploration_boost
        self._select_count = 0

    def select_action(
        self,
        belief: BeliefSnapshot,
        goals: Dict[str, float],
        ctx: CycleContext,
    ) -> Tuple[ActionCandidate, ActionDistributionSnapshot]:
        """
        Select action via EFE minimization and stochastic sampling.

        Returns:
            (selected_action, full_distribution)
        """
        self._select_count += 1

        # ── Compute EFE for each action ───────────────────────────────────────
        efe_values: Dict[str, float] = {}

        for action_type, catalogue in _ACTION_CATALOGUE.items():
            pragmatic = self._pragmatic_value(action_type, catalogue, goals, belief)
            epistemic  = self._epistemic_value(action_type, catalogue, belief)
            efe_values[action_type] = -(pragmatic + epistemic)  # negate: lower EFE = better

        # ── Softmax sampling (NOT argmax) ─────────────────────────────────────
        probabilities = self._softmax(efe_values, temperature=self.temperature)

        # Stochastic sample
        selected_type = self._sample(probabilities)
        selected_params = self._build_params(selected_type, belief, ctx)

        selected_action = ActionCandidate(
            action_type=selected_type,
            parameters=selected_params,
            expected_effects=self._expected_effects(selected_type, belief),
            source_module="EFEPolicyModule",
        )

        distribution = ActionDistributionSnapshot(
            probabilities=probabilities,
            temperature=self.temperature,
            efe_values=efe_values,
        )

        return selected_action, distribution

    # ── Internal computation ────────────────────────────────────────────────

    def _pragmatic_value(
        self,
        action_type: str,
        catalogue: Dict[str, float],
        goals: Dict[str, float],
        belief: BeliefSnapshot,
    ) -> float:
        """Pragmatic value = expected goal satisfaction."""
        value = 0.0
        for goal_name, goal_weight in goals.items():
            cat_key = _GOAL_TO_CATALOGUE_KEY.get(goal_name)
            if cat_key and cat_key in catalogue:
                value += goal_weight * catalogue[cat_key]

        # Distress modulation: under high distress, prefer direct responses
        distress = belief.emotional_state.get("distress", 0.0)
        if action_type == "respond" and distress > 0.5:
            value += distress * 0.5

        return value

    def _epistemic_value(
        self,
        action_type: str,
        catalogue: Dict[str, float],
        belief: BeliefSnapshot,
    ) -> float:
        """Epistemic value = expected uncertainty reduction."""
        base_gain = catalogue.get("epistemic_gain", 0.0)

        # Scale by current uncertainty: more uncertain → more value in resolving it
        uncertainty_weight = belief.epistemic_uncertainty
        value = base_gain * uncertainty_weight + self.exploration_boost

        return value

    def _expected_effects(
        self,
        action_type: str,
        belief: BeliefSnapshot,
    ) -> Dict[str, float]:
        """Predict what belief state changes this action will cause."""
        current_uncertainty = belief.epistemic_uncertainty
        return {
            "epistemic_uncertainty": max(0.0, current_uncertainty - 0.1),
            "distress":              max(0.0, belief.emotional_state.get("distress", 0) - 0.05),
        }

    def _build_params(
        self,
        action_type: str,
        belief: BeliefSnapshot,
        ctx: CycleContext,
    ) -> Dict:
        """Build action parameters based on context."""
        params: Dict = {}
        if action_type == "execute_capability":
            # Would normally come from capability assessment
            params["capability"] = "unknown"
            params["observation"] = ctx.observation.raw
        return params

    @staticmethod
    def _softmax(efe_values: Dict[str, float], temperature: float) -> Dict[str, float]:
        """
        Compute softmax distribution over negative EFE values.

        Lower EFE → higher probability (we negate before softmax).
        Temperature controls exploration vs exploitation.
        """
        # efe_values are already negated (higher = better)
        scores = {k: v / max(temperature, 1e-8) for k, v in efe_values.items()}
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}

    @staticmethod
    def _sample(probabilities: Dict[str, float]) -> str:
        """Sample action from probability distribution (NOT argmax)."""
        actions = list(probabilities.keys())
        weights = [probabilities[a] for a in actions]
        return random.choices(actions, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("EFEPolicyModule Demo\n" + "=" * 40)

    policy = EFEPolicyModule(temperature=1.0)

    scenarios = [
        {
            "name": "High distress, support needed",
            "belief": BeliefSnapshot(
                latent_state="crisis",
                epistemic_uncertainty=0.3,
                emotional_state={"distress": 0.9},
            ),
            "goals": {
                "REDUCE_USER_DISTRESS": 1.0,
                "BUILD_RAPPORT": 0.8,
                "MAINTAIN_IDENTITY": 1.0,
            },
        },
        {
            "name": "Exploration, low uncertainty",
            "belief": BeliefSnapshot(
                latent_state="exploration",
                epistemic_uncertainty=0.7,
                emotional_state={"distress": 0.05},
            ),
            "goals": {
                "EXPLORE_UNKNOWN": 0.9,
                "EXPAND_CAPABILITIES": 0.7,
                "MAINTAIN_IDENTITY": 1.0,
            },
        },
        {
            "name": "Task execution request",
            "belief": BeliefSnapshot(
                latent_state="instrumental",
                epistemic_uncertainty=0.2,
                emotional_state={"distress": 0.0},
            ),
            "goals": {
                "PROVIDE_INFORMATION": 0.9,
                "EXPAND_CAPABILITIES": 0.8,
                "MAINTAIN_IDENTITY": 1.0,
            },
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        ctx = CycleContext(observation=Observation(raw="test"))
        action, dist = policy.select_action(
            scenario["belief"], scenario["goals"], ctx
        )
        print(f"  Selected: {action.action_type}")
        top3 = sorted(dist.probabilities.items(), key=lambda x: -x[1])[:3]
        for act, prob in top3:
            efe = dist.efe_values[act]
            print(f"  {act:<28} p={prob:.3f}  efe={efe:+.3f}")
