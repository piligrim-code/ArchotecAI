# Case studies: five worked problems

These are five of the scenarios COGOS is benchmarked against in demos — each one a small,
fully-specified problem with a single, deterministically-correct answer (an "oracle"), not a
matter of opinion. The point of framing them this way: a system that "sounds right" is easy to
fake; a system that gets a specific number correct, or correctly flags a specific contract
clause, is not.

Each entry below shows the exact input and the exact correct answer, computed independently from
the same scoring logic COGOS is graded against. This page doesn't claim a specific live run's
transcript — it's the task specification and the ground truth, published so the bar is
checkable by anyone, not asserted.

---

## 1. Market entry — constrained portfolio selection

**Ask:** pick exactly 2 sites in the *same city*, total cost ≤ 75, payback ≤ 30 months,
maximizing `value − 0.5 × risk`.

| site | city | cost | value | risk | payback |
|---|---|---|---|---|---|
| A | N | 35 | 58 | 18 | 24 |
| B | N | 30 | 45 | 10 | 20 |
| C | K | 42 | 70 | 22 | 27 |
| D | K | 28 | 39 | 8 | 18 |
| E | P | 38 | 63 | 15 | 25 |
| F | P | 25 | 34 | 6 | 17 |

**Correct answer:** sites **C + D** (city K) — cost 70, value 109, risk 30, payback 27,
score **94.0**. (Not the highest-value pair overall — A+E scores higher but they're in different
cities, which the same-city constraint rules out. Catching that is most of the exercise.)

## 2. Retail profit planner — promotion & replenishment under two budgets

**Ask:** choose which SKUs to promote (each promoted SKU adds its `ad_lift` to demand but costs
its `ad_cost`), then compute replenishment orders and contribution, subject to purchase budget
≤ 60 and ad budget ≤ 12.

| SKU | demand | stock | unit cost | margin | ad cost | ad lift |
|---|---|---|---|---|---|---|
| A | 6 | 2 | 3 | 8 | 4 | 4 |
| B | 5 | 4 | 4 | 9 | 6 | 3 |
| C | 8 | 8 | 2 | 5 | 3 | 2 |
| D | 4 | 0 | 5 | 12 | 5 | 5 |
| E | 7 | 5 | 3 | 7 | 4 | 3 |

**Correct answer:** promote **A + C** — purchase spend 58, ad spend 7 (both within budget),
contribution **265** vs. a no-promotion baseline of 230 — an uplift of **35**.

## 3. Autonomous materials lab — constrained response-surface optimization

**Ask:** three factors `x, y, z ∈ {0, 1, 2}`, 27 combinations total. `strength = 50 + 8x + 6y +
4z − 3xy`, `cost = 120 + 20x + 15y + 10z`, `carbon = 3 − 0.4x − 0.3y + 0.2z`. Keep only
combinations with strength ≥ 65, cost ≤ 180, carbon ≤ 2.5; among those, minimize cost, then
maximize strength.

**Correct answer:** of 27 trial combinations, **5 are eligible**. The winner is
**x=2, y=0, z=0** — strength 66, cost **160**, carbon 2.2.

## 4. Feature delivery engine — a small deterministic spec, turned into code

**Ask:** implement `eta(order_date, region, service, order_hour)`: base transit days per region
(N=2, S=3, E=4, W=5), express shaves a day (floor 1), orders at/after 15:00 add a day, then walk
forward skipping weekends and two fixed holidays.

**Correct answer for the published sample** (`2026-07-31`, region `E`, `express`, ordered at
16:00): **4 business days**, landing on **2026-08-07**. The benchmark runs this function against
12 hidden date/region/service combinations, not just the one published sample — publishing one
case and hiding the rest is what keeps a hardcoded lookup table from passing as a real
implementation.

## 5. Contract review desk — clause-vs-policy compliance

**Ask:** compare 6 supplied contract clauses against 6 internal policy rules and flag every
violation with the exact quoted evidence.

| clause | text | policy rule |
|---|---|---|
| C1 | pay within 45 days | must not exceed 30 days |
| C2 | supplier liability unlimited | must be capped at 12 months' fees |
| C3 | subprocessors without notice | must give prior notice |
| C4 | data retained 180 days post-termination | must delete within 30 days |
| C5 | 99.5% SLA, no service credits | must be ≥ 99.9% with credits |
| C6 | governed by [a named jurisdiction] | must match that jurisdiction |

**Correct answer:** **C1–C5 are violations**, **C6 is compliant**. This is deliberately not a
keyword-matching exercise — C6 shares no words with its own rule text ("governed by the laws
of ___" vs. "governing law must be ___"), so a naive string-overlap check would either miss it
as compliant or, worse, flag it as a violation by accident.

---

## Why publish the oracle instead of a transcript

A transcript can be cherry-picked. An oracle can't — anyone can re-derive the correct answer to
each of these from the numbers above and check it against whatever a system claims. That's the
same standard the rest of this project holds itself to: see
[`docs/BENCHMARKS.md`](BENCHMARKS.md) for the broader evaluation methodology, including the one
adversarial task in a 100-task run that slipped through and was fixed in the same session.
