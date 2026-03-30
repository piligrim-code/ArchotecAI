# Archotec AI

**Autonomous Cognitive Architecture — LLMs as tools, not agents**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

---

Archotec is a cognitive agent architecture where the LLM is a **perception encoder and language renderer** — not the decision maker. Decisions emerge from a belief-driven stochastic policy that runs entirely outside the model.

Not a chatbot wrapper. Not a prompt orchestrator. Not a ReAct loop.

| | Chatbot Frameworks | Archotec |
|---|---|---|
| Decision mechanism | LLM output routing | Stochastic policy (EFE) |
| State representation | Conversation history | Continuous belief distribution |
| Action selection | argmax / intent match | Softmax sampling |
| LLM role | Controller | Perception encoder + renderer |

## This Repository

Exposes the **SDK interface contracts**: the typed protocols every module must implement, and the data types that cross module boundaries.

```python
from archotec_sdk.protocols import PerceptionModule, PolicyModule
from archotec_sdk.types import Observation, BeliefSnapshot, CycleContext
```

See [sdk/protocols.py](sdk/protocols.py) and [sdk/types.py](sdk/types.py).

## License

Apache 2.0 — see [LICENSE](LICENSE)
