# Architecture

```
Observation → Perception → Belief → Goals → Policy → Execution → Feedback → Learning
     ↑_______________________________________________________________|
```

**Kernel** — non-swappable. Cognitive cycle orchestration, module registry, safety constraints.

**Core** — hot-swappable modules. Each implements a typed protocol and can be replaced at runtime.

**Infrastructure** — LLM providers, REST API, external adapters.

## Key Properties

- **Belief state**: continuous probability distribution over world modes, updated Bayesian each cycle
- **Policy**: minimizes Expected Free Energy; selection is stochastic (softmax), never argmax
- **Meta-learning**: parameters update online; the system modifies itself within safety boundaries
- **Safety**: 11 independent layers; the kernel tier is non-swappable by design

## Module Protocols

All modules implement typed protocols from [`sdk/protocols.py`](sdk/protocols.py).

```
PerceptionModule  → observation → structured encoding
BeliefModule      → update posterior distribution
GoalModule        → compute priority vector
PolicyModule      → EFE minimization → stochastic sample
ExecutionModule   → dispatch action, render response
FeedbackModule    → compute intrinsic reward
MemoryModule      → retrieve + store episodes
LearningModule    → update parameters
CapabilityModule  → assess + acquire capabilities
```
