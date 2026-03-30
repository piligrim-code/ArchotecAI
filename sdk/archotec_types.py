"""
Archotec SDK — Core data types.

These are the types that cross module boundaries in the cognitive cycle.
Every module speaks this language; internal representations stay private.

INVARIANT: Schema is stable. Field names and types do not change at runtime.
CycleContext is mutable (phases accumulate state into it), but the schema is frozen.

Usage::

    from archotec_sdk.types import Observation, CycleContext, BeliefSnapshot

    obs = Observation(raw="What is the Bitcoin price?", source="external")
    ctx = CycleContext(observation=obs, episode_id="ep_001")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Observation — raw input entering the cognitive cycle
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    Raw input entering the cognitive cycle.

    Created once per cycle at the boundary between the environment and the agent.
    The perception module converts this into a PerceptionOutput.

    Attributes:
        raw:       The raw text or observation string.
        source:    Where the observation came from.
                   "external" — user or environment message
                   "internal" — autonomous drive / self-generated
                   "sensor"   — sensor / tool reading
        timestamp: When the observation was received.
        metadata:  Arbitrary extra data (e.g. user_id, channel, priority).
    """
    raw: str
    source: str = "external"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PerceptionOutput — structured result of perception phase
# ---------------------------------------------------------------------------

@dataclass
class PerceptionOutput:
    """
    Structured perception result produced by the PerceptionModule.

    The perception phase encodes the raw observation into a structured
    representation that the belief module can use for Bayesian inference.

    Key concepts:
    - emotional_state: Continuous-valued affective dimensions [0, 1]
    - intent_distribution: Probability distribution over possible user intents
    - epistemic_uncertainty: How uncertain the perception is about what was observed
    - embedding: Optional dense vector representation (if sentence-transformers available)

    Attributes:
        observation:          The original observation text.
        emotional_state:      Dict of affect dimensions → scalar [0, 1].
                              Standard keys: distress, valence, arousal, dominance.
        intent_distribution:  Dict of intent_name → probability.
        intent_uncertainty:   Overall uncertainty about the intent [0, 1].
        concepts:             Key concepts extracted from the observation.
        relations:            (subject, predicate, object) triples.
        summary:              One-sentence summary of the observation.
        embedding:            Dense vector (numpy array) if available.
        epistemic_uncertainty: How uncertain the perceptual encoding is [0, 1].
    """
    observation: str

    emotional_state: Dict[str, float] = field(default_factory=dict)
    intent_distribution: Dict[str, float] = field(default_factory=dict)
    intent_uncertainty: float = 0.5

    concepts: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    summary: str = ""
    embedding: Optional[Any] = None  # np.ndarray when numpy is available

    epistemic_uncertainty: float = 0.5

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BeliefSnapshot — agent's current belief about the world
# ---------------------------------------------------------------------------

@dataclass
class BeliefSnapshot:
    """
    Snapshot of the agent's belief state at a point in time.

    The belief module maintains a continuous probability distribution over
    18 latent world modes (plus dynamically discovered modes). This snapshot
    captures the full belief state at one moment in the cognitive cycle.

    Key concepts:
    - latent_state_distribution: Posterior P(s|o) over 18 latent modes
    - epistemic_uncertainty: How uncertain the agent is about the world state
    - emotional_state: The agent's own emotional/regulatory state
    - inferred_goals: What goals the agent infers from current context

    Standard latent states:
        NEUTRAL, CRISIS, SUPPORT, EXPLORATION, INSTRUMENTAL, SOCIAL,
        REFLECTIVE, COLLABORATIVE, UNCERTAIN, TEACHING,
        QUESTION_ABOUT_USER, QUESTION_ABOUT_AGENT, CASUAL_DIALOGUE,
        SEEKING_ADVICE, GREETING, PLAYFUL, EMOTIONAL_EXPRESSION,
        STATEMENT_SHARING

    Attributes:
        emotional_state:          Agent's affective state dimensions [0, 1].
        epistemic_uncertainty:    Uncertainty about world state [0, 1].
        aleatoric_uncertainty:    Irreducible environmental uncertainty [0, 1].
        identity_certainty:       Agent's sense of identity stability [0, 1].
        ontological_clarity:      Clarity about what the world contains [0, 1].
        self_awareness_level:     Self-monitoring depth [0, 1].
        latent_state:             MAP estimate of current world mode (string).
        latent_state_distribution: Full posterior distribution over all modes.
        inferred_goals:           List of (goal_name, priority) tuples.
        action_history:           Count of each action type taken so far.
        capability_performance:   Per-capability success metrics.
    """
    emotional_state: Dict[str, float] = field(default_factory=dict)

    epistemic_uncertainty: float = 0.5
    aleatoric_uncertainty: float = 0.3

    identity_certainty: float = 0.5
    ontological_clarity: float = 0.5
    self_awareness_level: float = 0.5

    latent_state: str = "neutral"
    latent_state_distribution: Dict[str, float] = field(default_factory=dict)
    dynamic_state: Optional[str] = None

    relational_dimensions: Dict[str, float] = field(default_factory=dict)
    inferred_goals: List[Tuple[str, float]] = field(default_factory=list)
    action_history: Dict[str, int] = field(default_factory=dict)
    capability_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    learning_trigger: Optional[Dict[str, Any]] = None

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

@dataclass
class ActionCandidate:
    """
    A candidate action proposed by the policy module.

    The policy module computes Expected Free Energy (EFE) for each candidate
    and samples stochastically. The sampled candidate becomes the selected action.

    EFE(a) = pragmatic_value(a) + epistemic_value(a) + causal_bonus(a)

    Selection: softmax(−EFE / temperature), then sample — never argmax.

    Attributes:
        action_type:      Name of the action (e.g. "execute_capability", "respond").
        parameters:       Action-specific parameters (e.g. capability name, target).
        expected_effects: Predicted belief state changes after execution.
        source_module:    Which module proposed this candidate.
    """
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_effects: Dict[str, float] = field(default_factory=dict)
    source_module: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDistributionSnapshot:
    """
    Full probability distribution over candidate actions at policy selection time.

    Captures the stochastic policy at one moment — useful for:
    - Meta-learning (what was the policy entropy?)
    - Auditing (why was this action chosen over others?)
    - Visualization (policy distribution over time)

    Attributes:
        probabilities: action_type → softmax probability.
        temperature:   Sampling temperature used.
        efe_values:    Raw EFE score per action (lower = preferred).
    """
    probabilities: Dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0
    efe_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionOutcomeRecord:
    """
    Result of executing a selected action.

    The feedback module uses this to compute intrinsic reward:
    reward ∝ reduction in epistemic_uncertainty + goal_satisfaction
             − prediction_error

    Attributes:
        action_type:       Which action was executed.
        generated_text:    Text produced (if any).
        expected_effects:  What effects the policy predicted.
        actual_effects:    What effects actually occurred (from belief diff).
        reward:            Computed intrinsic reward scalar.
        prediction_error:  |expected_effects − actual_effects|.
        success:           Whether execution succeeded without error.
    """
    action_type: str
    generated_text: str = ""
    expected_effects: Dict[str, float] = field(default_factory=dict)
    actual_effects: Dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    prediction_error: float = 0.0
    execution_time: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CycleContext — accumulates state across one cognitive cycle
# ---------------------------------------------------------------------------

@dataclass
class CycleContext:
    """
    Full mutable context for one cognitive cycle.

    Created at cycle start, passed through all 15 phases, returned at end.
    Each module reads what it needs and writes its output back into context.

    Phase pipeline (simplified):
        Observation → Perception → Belief → Goals → Policy →
        Planning → Execution → Feedback → Memory → Learning

    Attributes:
        observation:        The raw observation triggering this cycle.
        episode_id:         Unique ID for this cycle/episode.
        perception:         Output of perception phase.
        belief_before:      Belief snapshot before update.
        belief_after:       Belief snapshot after update.
        goal_priorities:    Dict of goal_name → priority weight.
        selected_action:    Action chosen by policy.
        action_distribution: Full distribution policy computed.
        outcome:            Result of executing the selected action.
        relevant_memories:  Memories retrieved before action (for context).
        cognitive_trace:    List of phase trace entries (for debugging).
    """
    observation: Observation
    episode_id: str = ""

    perception: Optional[PerceptionOutput] = None

    belief_before: Optional[BeliefSnapshot] = None
    belief_after: Optional[BeliefSnapshot] = None

    goal_priorities: Dict[str, float] = field(default_factory=dict)

    selected_action: Optional[ActionCandidate] = None
    action_distribution: Optional[ActionDistributionSnapshot] = None

    outcome: Optional[ActionOutcomeRecord] = None

    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)

    cognitive_trace: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def processing_time(self) -> float:
        """Total cycle processing time in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0

    def trace(self, phase: str, **kwargs: Any) -> None:
        """Append a trace entry for the given phase."""
        entry: Dict[str, Any] = {"phase": phase}
        entry.update(kwargs)
        self.cognitive_trace.append(entry)
