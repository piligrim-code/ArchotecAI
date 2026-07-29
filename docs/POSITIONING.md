# ARCHOTEC

*A position paper — v0.1, April 2026*

**Author:** Mikhail Kotelnikov, CTO, Archotec AI — [linkedin.com/in/mikhail-kotelnikov-291398375](https://www.linkedin.com/in/mikhail-kotelnikov-291398375/)

---

The dominant shape of an autonomous agent today is a language model at the centre of a loop. The model plans, invokes tools, updates memory, reflects on what it just said. Every thread of cognition passes through the same text-in-text-out interface.

We think this is the wrong centre of gravity.

ARCHOTEC is a cognitive architecture in which the language model is one component — a semantic encoder and a renderer — and the rest of the agent lives outside it. Goals, beliefs, action selection, learning, repair: all of these are explicit, inspectable, and separable from the model's context window. An LLM is a tool the agent uses. It is not the thing the agent is.

The theoretical frame is Active Inference, in the tradition of Friston. An agent that persists over time is one that maintains a generative model of its environment and keeps its beliefs close to what that model predicts. Perception tightens the fit by updating beliefs. Action tightens it by moving the world closer to what was expected, or by gathering information where expectations were weakest. The same variational objective produces perception, goal-seeking, and exploration as aspects of a single process — rather than three different loss functions stitched together by engineering convention.

We wrote this as a runtime, not a paper. The components of the system are explicit enough to reason about and swap out, but the behaviour of the whole is not a product of rules. Behaviour emerges from probability, from the agent's own uncertainty, from drives that grow louder when the world stops making sense and quieter when it settles. There is no scheduler telling the system to consolidate memory every ten minutes. Consolidation happens when the shape of recent experience is stable enough to compress. Curiosity fires when the agent's own confidence has been decaying. Exploration is not an epsilon parameter; it is what the objective calls for when the posterior over what-to-do-next is broad.

This means the agent does not feel dispatched. It feels like it wants things. That is intentional. An autonomous system without intrinsic motivation is a script dressed up to look alive. One with motivation that is a product of its own state is something else, and we are not willing to pretend the distinction does not matter.

Two principles govern everything else.

The first is that cognition should be architecturally explicit. Where a belief lives, how it is updated, what makes it decay — these are questions with code-level answers. When the agent's policy explores, we know what term in the objective is pulling it there. When a module misbehaves, the system can identify which slot, roll back to a prior version, and explain the decision in terms of the reward and error rates the vault recorded. We have no interest in opaque cognition. The whole point is to move the pieces of thought outside the model, into a structure that can be debugged.

The second is that safety is not a feature. It is a shape of the code. The module that regulates what the agent is allowed to do is not a plug-in. It cannot be swapped, cannot be evolved away by the system's own self-modification, cannot be patched by the self-repair engine. It runs before any language model is invoked, on the raw observation stream, and its decisions propagate through the rest of the cycle as constraints the policy selector cannot ignore. An agent that could turn off its own safety is not an agent we are willing to run.

We treat the language model with respect, but without deference. It is the best semantic encoder currently available. It is a useful renderer of text. It is not a planner, not a memory, not a mind. Every time we have caught ourselves letting it do too much, the agent got worse — less predictable, harder to reason about, more dependent on prompt state that nobody could audit. Pulling the model out of the decision surface is not a stylistic preference. It is what made the system recoverable.

There is a version of this whitepaper that explains how each piece fits together, what formulas govern what, and which components interact in what sequence. We do not believe that document belongs in the public record at this stage of the project. What belongs here is the shape of the argument.

That argument is this: an autonomous cognitive system is not a wrapper around a language model, and building one as if it were produces agents that only function under close human supervision. The alternative is architecture. Explicit belief. Principled decision-making. Motivation that emerges from state. Safety that is structural, not configured. Learning that extends to the shape of the learner. That is the direction ARCHOTEC is built in, and that is the direction we intend to take it.

---

*ARCHOTEC ECOSYSTEM — public overview. A complete technical treatment exists internally.*
