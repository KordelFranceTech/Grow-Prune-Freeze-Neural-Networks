# Decision Log

A running record of non-obvious architectural choices. Each entry captures **what was decided, when, why, and what would change the decision**. The goal is to make the project debuggable months later, when "why is the rollout length 10?" is a real question and the answer is no longer in anyone's head.

## How to add an entry

Append to the bottom of this file. Do not edit past entries; if a decision is reversed, add a new entry that references the old one.

Entry template:

```
## YYYY-MM-DD: Short title

**Decision:** What was chosen.

**Context:** What problem this was solving; what alternatives were considered.

**Reasoning:** Why this option won.

**Reversal trigger:** What evidence would cause us to revisit this.

**Affected components:** Which directories / SKILL.md files this touches.
```

Keep entries short — a paragraph each, not an essay. If a decision needs more than a paragraph, that's a sign it should also be documented in the relevant SKILL.md.

---

## Seed entries

These document the choices already baked into the initial project setup.

## 2025-XX-XX: Option 1 architecture (world model + separate planner)

**Decision:** Use a learned world model for plume prediction with a separate hand-designed planner that picks actions by scoring rollouts. Not end-to-end RL; not a Decision Transformer; not a shared-trunk world-model-plus-policy.

**Context:** Three architectures were considered for the model-based agent: (1) world model + separate policy, (2) Decision Transformer with direct action output, (3) shared transformer trunk with both world-model and policy heads.

**Reasoning:** Option 1 keeps the components independently testable, lets the world model be trained with pure supervised next-token prediction (no RL needed for the hard part), and matches the original hypothesis that explicit simulation of futures should drive decisions. Options 2 and 3 either skip the rollout step entirely (defeating the hypothesis) or tangle two training problems together.

**Reversal trigger:** End-to-end navigation success fails to beat the reactive baseline even after the world model passes its level-2 validation. At that point, the planning machinery is not earning its compute and a Decision Transformer fallback should be considered.

**Affected components:** all of `/world_model`, `/planner`, `CLAUDE.md`.

## 2025-XX-XX: Multi-stream factored tokenization (not flat joint vocabulary)

**Decision:** Each timestep emits separate tokens for left concentration, right concentration, and wind, with embeddings summed at each position. Not a single flattened joint vocabulary of (L × R × W) combinations.

**Context:** Two tokenization strategies were on the table: a flat joint vocabulary of ~170 tokens, or factored streams with per-stream embeddings summed at each position.

**Reasoning:** Factored streams are more sample-efficient — the model doesn't have to relearn that two combinations sharing a left-concentration value share concentration structure. Per-stream loss is also independently inspectable, which makes diagnosis easier. The parameter cost is lower.

**Reversal trigger:** Per-stream losses suggest the streams interact strongly in ways the summed-embedding bottleneck can't represent. In that case, late fusion (concatenated embeddings projected to d_model) is the next thing to try.

**Affected components:** `/tokenization`, `/world_model`.

## 2025-XX-XX: Filament-based simulator, not LLM-generated training data

**Decision:** Generate all training trajectories from a filament-based plume simulator (Farrell-Murlis style). Do not use a larger language model to synthesize plume data.

**Context:** Original proposal considered LLM-generated training data. Plume physics has well-understood statistical structure (intermittency, filament lifetimes, log-normal whiff distributions) that an LLM has no privileged access to.

**Reasoning:** An LLM would produce plausible-looking sequences with wrong statistics, training the world model to imitate hallucinated plumes. Filament simulators are the standard for plume-tracking research, run fast enough to generate 10⁶+ timesteps, and reproduce the right statistics.

**Reversal trigger:** None expected. If the filament simulator turns out to be too slow or too crude, the next step is LES (Large Eddy Simulation) of plumes, not LLM data.

**Affected components:** `/simulator`, `CLAUDE.md`.

## 2025-XX-XX: Hand-designed scoring before learned value function

**Decision:** Phase 1 of the planner uses a hand-designed scoring rule (whiff count, upwind progress, blank penalty, concentration trend). A learned value head is a Phase 2 option, only after the hand-designed version is shown to work.

**Context:** Could train a value function end-to-end with RL from the start, or start with an interpretable scoring rule.

**Reasoning:** The hand-designed rule requires no training, is debuggable by inspection, and provides a baseline that a learned function must beat. Starting with RL adds a second training problem on top of the world model and makes failures hard to localize.

**Reversal trigger:** Hand-designed scoring plateaus below the reactive-policy baseline despite a validated world model. At that point, train a value head on top of the frozen world model.

**Affected components:** `/planner`.

---

## Active decisions

(append new entries below this line)

---

## 2026-05-07: Reflecting domain boundaries instead of OOB termination

**Decision:** When the agent's position exceeds the domain bounds, clip it to the boundary (reflect) rather than terminating the episode. The `done` signal no longer includes `not in_bounds`.

**Context:** During autoresearch (GPF optimization), the reactive baseline scored 0% across 200 episodes — not because the task is impossible but because the heuristic policy (turn upwind, cast on blank) has no boundary awareness and repeatedly walks off the domain edge within 100–300 steps. The RL agent also terminates prematurely for the same reason, receiving no learning signal about *why* the episode ended.

**Reasoning:** The agent's observation (left_bin, right_bin, wind_octant) contains no boundary proximity information. Terminating on OOB creates an unexplainable reward cliff: the agent gets a negative outcome (episode ends) with no input signal that distinguishes "near boundary" from "not near boundary". This makes boundary avoidance impossible to learn directly and turns every training episode into a coin-flip on whether the random walk happened to stay in bounds. Reflecting boundaries let episodes run to max_steps or success, giving the agent 10× more gradient signal per episode and making the reactive baseline non-trivially comparable.

**Trade-off:** A real robot would have some boundary sensing (LIDAR, GPS fence, etc.) or truly fail by going out of range. Reflecting walls are an abstraction, but they are strictly dominated by adding a boundary sensor to the state. Adding the sensor is the right long-term fix (Phase 2); reflecting boundaries fix the training dynamics now without breaking the existing tokenization contract.

**Reversal trigger:** Add domain boundary distance as a new state token (fourth stream in the observation). At that point, restore OOB termination so the agent must learn active boundary avoidance rather than being bailed out by clipping.

**Affected components:** `simulator/filament_sim.py` (`step()`), all RL training loops.
