"""
Archotec SDK — Module protocol contracts.

Every cognitive module in Archotec implements one or more of these protocols.
Using typing.Protocol (PEP 544) means existing classes conform structurally —
no need to modify their inheritance tree.

These protocols define the interface contracts at each phase of the cognitive
cycle. Implement them to plug custom modules into the architecture.

INVARIANT: These protocol definitions never change at runtime.
           The runtime evolution system generates new classes that implement these protocols.

Example — implementing a custom perception module::

    from archotec_sdk.protocols import PerceptionModule
    from archotec_sdk.types import Observation, CycleContext, PerceptionOutput

    class KeywordPerception:
        async def perceive(
            self,
            observation: Observation,
            ctx: CycleContext,
        ) -> PerceptionOutput:
            distress = 1.0 if any(w in observation.raw.lower()
                                  for w in ["help", "urgent", "crisis"]) else 0.0
            return PerceptionOutput(
                observation=observation.raw,
                emotional_state={"distress": distress},
                epistemic_uncertainty=0.5,
            )

        async def learn_from_outcome(self, observation, actual_distress,
                                     confidence, episode_id):
            pass  # stateless in this example
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .types import (
    ActionCandidate,
    ActionDistributionSnapshot,
    ActionOutcomeRecord,
    BeliefSnapshot,
    CycleContext,
    Observation,
    PerceptionOutput,
)


# ---------------------------------------------------------------------------
# Core cognitive module protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class PerceptionModule(Protocol):
    """
    Phase 3 — converts raw observations into structured perception.

    The perception module is one of three phases that calls the LLM.
    It encodes the raw observation into intent, emotion, and uncertainty
    representations that the belief module can consume.

    The LLM output is a structured encoding tool — not a decision maker.
    """

    async def perceive(
        self,
        observation: Observation,
        ctx: CycleContext,
    ) -> PerceptionOutput:
        """
        Process raw observation into structured perception output.

        Args:
            observation: The raw input (text, sensor reading, internal trigger).
            ctx:         Current cycle context (read-only in this phase).

        Returns:
            PerceptionOutput with emotional_state, intent_distribution,
            epistemic_uncertainty, and optional embedding.
        """
        ...

    async def learn_from_outcome(
        self,
        observation: str,
        actual_distress: float,
        confidence: float,
        episode_id: str,
    ) -> None:
        """
        Update perception parameters from execution outcome.

        Called at the end of each cycle to close the prediction-error loop.
        The module should update any internal weights or thresholds.
        """
        ...


@runtime_checkable
class BeliefModule(Protocol):
    """
    Phase 4 — maintains and updates the agent's Bayesian belief state.

    The belief module performs Bayesian inference:
        P(s|o) ∝ P(o|s) × P(s)

    where s is one of 18 latent world modes (or dynamically discovered modes)
    and o is the perception output.

    Key behaviour:
    - Updates the posterior over latent states each cycle
    - Triggers epistemic reset when Jensen-Shannon divergence > threshold
    - Tracks uncertainty as a continuous value, not a discrete flag
    """

    def update(
        self,
        perception: PerceptionOutput,
        ctx: CycleContext,
    ) -> BeliefSnapshot:
        """
        Update belief state from perception and return new snapshot.

        Args:
            perception: Structured output from perception module.
            ctx:        Cycle context (read belief_before from here).

        Returns:
            Updated BeliefSnapshot with new latent_state_distribution.
        """
        ...

    def get_current(self) -> BeliefSnapshot:
        """Return current belief snapshot without modifying state."""
        ...

    def snapshot(self) -> BeliefSnapshot:
        """Return an immutable snapshot of current belief state."""
        ...


@runtime_checkable
class GoalModule(Protocol):
    """
    Phase 8 — computes goal priorities from current belief state.

    The goal system maintains 9 autonomous goals with dynamic priorities.
    Priorities are modulated by belief state every cycle — not hardcoded.

    Standard goals:
        MAINTAIN_IDENTITY (fixed 1.0), LEARN_AND_ADAPT, EXPAND_CAPABILITIES,
        EXPLORE_UNKNOWN, PROVIDE_INFORMATION, REDUCE_UNCERTAINTY,
        REDUCE_USER_DISTRESS, ENSURE_SAFETY, BUILD_RAPPORT

    Autonomous drives (curiosity, consolidation, adaptation, exploration,
    self_assessment) inject stochastic goal activations via sigmoid dynamics.
    """

    def update_goals(
        self,
        belief: BeliefSnapshot,
        ctx: CycleContext,
    ) -> Dict[str, float]:
        """
        Update and return goal priorities based on current belief.

        Args:
            belief: Current belief snapshot.
            ctx:    Cycle context.

        Returns:
            Dict of goal_name → priority weight [0, 1].
            MAINTAIN_IDENTITY should always be 1.0.
        """
        ...


@runtime_checkable
class PolicyModule(Protocol):
    """
    Phase 9 — selects actions by minimizing Expected Free Energy.

    EFE(a) = pragmatic_value(a) + epistemic_value(a) + causal_bonus(a)

    where:
        pragmatic_value  = expected goal satisfaction
        epistemic_value  = expected information gain (uncertainty reduction)
        causal_bonus     = reward for actions that causally improve outcomes

    Action selection: softmax(−EFE / temperature) → stochastic sample.

    CRITICAL ARCHITECTURAL INVARIANT:
        Never use argmax. Never use if/else dispatch on belief states.
        Actions must emerge from the stochastic distribution.
    """

    def select_action(
        self,
        belief: BeliefSnapshot,
        goals: Dict[str, float],
        ctx: CycleContext,
    ) -> Tuple[ActionCandidate, ActionDistributionSnapshot]:
        """
        Select an action via stochastic EFE-minimizing policy.

        Args:
            belief: Current belief snapshot.
            goals:  Current goal priorities from goal module.
            ctx:    Cycle context.

        Returns:
            (selected_action, full_distribution)
            — the action sampled from the EFE distribution,
              plus the full distribution for meta-learning.
        """
        ...


@runtime_checkable
class ExecutionModule(Protocol):
    """
    Phase 11 — executes selected actions and returns outcomes.

    The execution module dispatches the selected action to the appropriate
    capability adapter (computer_control, trading, web_search, etc.) or
    triggers language generation if the action is text-based.

    Execution is one of three phases that may call the LLM (for rendering).
    The LLM here is a language renderer, not a decision maker.
    """

    async def execute(
        self,
        action: ActionCandidate,
        belief: BeliefSnapshot,
        ctx: CycleContext,
    ) -> ActionOutcomeRecord:
        """
        Execute action and return outcome record.

        Args:
            action: The action selected by policy.
            belief: Current belief snapshot (for context).
            ctx:    Cycle context.

        Returns:
            ActionOutcomeRecord with generated_text, actual_effects,
            prediction_error, and reward.
        """
        ...


@runtime_checkable
class FeedbackModule(Protocol):
    """
    Phase 12 — computes intrinsic reward from execution outcome and belief change.

    Intrinsic reward is computed from three sources:
    1. Prediction error:  |expected_effects − actual_effects|  (lower = better)
    2. Belief change:     KL(belief_after || belief_before)     (goal-directed)
    3. Goal satisfaction: dot(goal_priorities, actual_effects)

    This reward feeds into all four levels of meta-learning (L1-L4).
    """

    def compute_reward(
        self,
        outcome: ActionOutcomeRecord,
        belief_before: BeliefSnapshot,
        belief_after: BeliefSnapshot,
        goals: Dict[str, float],
        ctx: CycleContext,
    ) -> float:
        """
        Compute scalar intrinsic reward.

        Args:
            outcome:        Result of action execution.
            belief_before:  Belief state before execution.
            belief_after:   Belief state after execution.
            goals:          Current goal priorities.
            ctx:            Cycle context.

        Returns:
            Scalar reward value. Typical range: [−1, 1].
        """
        ...


@runtime_checkable
class MemoryModule(Protocol):
    """
    Phase 7 (retrieval) + Phase 13 (storage) — stores and retrieves experiences.

    Memory operates at two levels:
    - Episodic: individual experiences (observation, action, outcome, reward)
    - Semantic: compressed patterns extracted from episodic memory

    Consolidation converts episodic into semantic on a rolling basis.
    """

    def store_experience(self, ctx: CycleContext) -> None:
        """
        Store completed cycle as an episodic memory.

        Called at the end of Phase 13 with the full CycleContext.
        """
        ...

    def retrieve_relevant(
        self,
        query: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve experiences relevant to a query.

        Called at Phase 7 (before action) to inject context.
        Query typically contains: {'observation': str, 'latent_state': str}

        Returns:
            List of memory dicts, most relevant first.
        """
        ...

    def consolidate(self) -> None:
        """
        Run memory consolidation pass (episodic → semantic compression).

        May be called asynchronously at low load, or after every N cycles.
        """
        ...


@runtime_checkable
class LearningModule(Protocol):
    """
    Phase 14 — updates internal parameters from this cycle's experience.

    Implements the 4-level meta-learning hierarchy:

        L1: Gradient Descent    — scalar reward signal, all parameters
        L2: Causal Tracking     — per-parameter Pearson correlation, auto-rollback
        L3: Discovery           — scans module health for new learnable dimensions
        L4: Self-Referential    — monitors L1-L3 health, escalates if broken

    Escalation is stochastic: P(escalate) = sigmoid(distress × 6 − 3)
    This is NOT an if/else threshold — it is a smooth sigmoid transition.
    """

    def learn(self, ctx: CycleContext) -> Dict[str, Any]:
        """
        Run all learning updates for this cycle.

        Args:
            ctx: Completed cycle context with outcome and belief states.

        Returns:
            Dict describing what was updated, e.g.:
            {'l1_updates': 3, 'l2_rollbacks': 0, 'l3_discovered': 0}
        """
        ...


# ---------------------------------------------------------------------------
# Capability assessment and acquisition
# ---------------------------------------------------------------------------

@runtime_checkable
class CapabilityModule(Protocol):
    """
    Phase 5 — assesses whether the agent has a capability, and acquires missing ones.

    Capability beliefs are maintained as a Bayesian distribution:
    P(can_do | observation) updated from success/failure outcomes.

    When can_do = False, the acquisition pipeline activates:
    discovers what tools are available, selects a strategy, generates
    and validates an adapter, and registers it for the next cycle.
    """

    async def assess_capability(
        self,
        observation: str,
        ctx: CycleContext,
    ) -> Dict[str, Any]:
        """
        Assess whether the agent has the capability to handle this observation.

        Returns:
            Dict with at minimum:
            {'can_do': bool, 'capability': str | None, 'confidence': float}
        """
        ...

    async def acquire_capability(
        self,
        capability_name: str,
        observation: str,
        ctx: CycleContext,
    ) -> Dict[str, Any]:
        """
        Attempt to acquire a missing capability.

        Returns:
            Dict with acquisition result:
            {'success': bool, 'capability': str, 'adapter': str | None}
        """
        ...


@runtime_checkable
class DelegationModule(Protocol):
    """
    Optional — delegates subtasks to subagents.

    When a task exceeds the current agent's capabilities or cognitive load,
    the delegation module spawns a subagent with a scoped context.
    """

    async def delegate(
        self,
        task: str,
        context: Dict[str, Any],
        ctx: CycleContext,
    ) -> Dict[str, Any]:
        """
        Delegate a task to a subagent.

        Args:
            task:    Natural language task description.
            context: Relevant context for the subagent.
            ctx:     Current cycle context.

        Returns:
            Subagent result dict with 'response', 'success', and 'metadata'.
        """
        ...


# ---------------------------------------------------------------------------
# EvolvableModule — marker for hot-swappable modules
# ---------------------------------------------------------------------------

@runtime_checkable
class EvolvableModule(Protocol):
    """
    Marker protocol for modules that can be hot-swapped at runtime.

    All module implementations should implement this in addition to their
    functional protocol (PerceptionModule, PolicyModule, etc.).

    State Transfer Contract
    -----------------------
    When the evolution system swaps a module via registry.register(), the
    registry automatically calls::

        state = old_module.get_state()
        new_module.load_state(state)

    This preserves continuity of accumulated runtime state:
    - Performance counters (learn_count, error_count, reward_count, ...)
    - Rolling reward/error histories used by the evolution gate
    - Any other statistics that inform health metrics

    Rules:
    - get_state() MUST be idempotent and non-destructive
    - load_state() MUST silently ignore unknown keys (forward-compatibility)
    - Both MUST be safe to call at any time
    - State dict values MUST be JSON-serializable primitives
    """

    @property
    def module_id(self) -> str:
        """Unique identifier for this module implementation."""
        ...

    @property
    def module_version(self) -> str:
        """Version string for this module implementation."""
        ...

    def get_health_metrics(self) -> Dict[str, float]:
        """
        Return health metrics for evolution gate decisions.

        Example::

            {
                'prediction_error': 0.3,
                'avg_latency_s': 0.1,
                'success_rate': 0.9,
                'reward_trend': 0.05,
            }
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """
        Return serializable runtime state for transfer on hot-swap.

        Only accumulated counters and histories — NOT references to
        underlying core objects (those are re-injected on creation).

        Returns {} for modules with no meaningful accumulated state.
        """
        ...

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore accumulated runtime state from a previous module instance.

        Called immediately after registry.register() if the previous
        occupant implemented get_state(). Unknown keys must be ignored.
        """
        ...
