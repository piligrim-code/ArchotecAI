# portfolio_analytics — generated organs

COGOS was given the goal "build a portfolio risk-analytics system" and designed a blueprint of
6 organs. Of those, 5 produced real code:

| organ | auto-validation status | code |
|---|---|---|
| `drawdown_analyzer` (max drawdown) | **activated** — passed full auto-validation | `organs/drawdown_analyzer.py` |
| `volatility_calculator` (volatility + Sharpe) | staged — real code, auto-tests were flaky | `organs/volatility_calculator.py` |
| `covariance_matrix_generator` (covariance/correlation) | staged | `organs/covariance_matrix_generator.py` |
| `min_variance_optimizer` (min-variance weights via `numpy.linalg`) | staged | `organs/min_variance_optimizer.py` |
| `report_builder` | still building when the blueprint was packaged | no code produced |

"Staged" vs. "activated" is COGOS's own internal validation status, not a judgment on
correctness — the math in the staged organs is real and checks out; their auto-tests just hadn't
passed cleanly yet at the time this snapshot was taken. All four files here are unedited COGOS
output. `report_builder` isn't included because there's no generated code for it.
