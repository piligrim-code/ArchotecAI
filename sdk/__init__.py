"""
Archotec AI SDK — Public interfaces for the Archotec cognitive architecture.

This package exposes the protocol interfaces and data types that define
the contracts between cognitive modules. Use these to:

  - Implement custom perception, belief, policy, or memory modules
  - Build adapters for external tools and APIs
  - Run the benchmark suite against your implementations
  - Understand the architectural contracts without the private implementation

Example::

    from archotec_sdk.protocols import PerceptionModule, BeliefModule
    from archotec_sdk.types import Observation, CycleContext

    class MyPerception(PerceptionModule):
        async def perceive(self, observation, ctx):
            ...

Docs: https://github.com/archotec-ai/archotec/tree/main/docs
"""

__version__ = "0.3.0"
__author__ = "Archotec AI"
__license__ = "Apache-2.0"

from .protocols import (
    BeliefModule,
    CapabilityModule,
    DelegationModule,
    EvolvableModule,
    ExecutionModule,
    FeedbackModule,
    GoalModule,
    LearningModule,
    MemoryModule,
    PerceptionModule,
    PolicyModule,
)
from .types import (
    ActionCandidate,
    ActionDistributionSnapshot,
    ActionOutcomeRecord,
    BeliefSnapshot,
    CycleContext,
    Observation,
    PerceptionOutput,
)

__all__ = [
    # Protocols
    "PerceptionModule",
    "BeliefModule",
    "GoalModule",
    "PolicyModule",
    "ExecutionModule",
    "FeedbackModule",
    "MemoryModule",
    "LearningModule",
    "CapabilityModule",
    "DelegationModule",
    "EvolvableModule",
    # Types
    "Observation",
    "PerceptionOutput",
    "BeliefSnapshot",
    "CycleContext",
    "ActionCandidate",
    "ActionDistributionSnapshot",
    "ActionOutcomeRecord",
]
