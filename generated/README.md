# Generated — raw output of COGOS's own capability builder

This directory is not a demo. It's a small gallery of code COGOS wrote itself, unedited, kept
exactly as it came out of the capability builder — no orchestration wrapper, no "here's how to
run the app" packaging on top. Each file is a real function COGOS was asked to produce as part of
a larger blueprint; some passed full auto-validation, some were staged (real code, but the
auto-test pass was flaky) — the per-directory notes below say which is which.

| | Blueprint COGOS was asked for | Organs generated |
|---|---|---|
| [`rag_bot_self_build/`](rag_bot_self_build/) | A RAG bot | `similarity_searcher.py` |
| [`portfolio_analytics/`](portfolio_analytics/) | A portfolio risk-analytics system | `volatility_calculator.py`, `covariance_matrix_generator.py`, `drawdown_analyzer.py`, `min_variance_optimizer.py` |

Neither blueprint fully auto-assembled into a finished application — the orchestration layer
(wiring the organs together, handling I/O, the parts that turn "some functions" into "a program")
isn't included here, because that part was hand-written, not generated. Publishing it alongside
the generated organs would blur which is which; this directory stays limited to what's actually
COGOS's own output.
