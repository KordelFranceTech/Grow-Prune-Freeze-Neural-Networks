# Planner Skill

## Purpose

At each control step, choose the next action by simulating short probable futures with the world model and scoring them. The planner is the **decision-maker**; the world model is only the imagination engine it queries.

## When to use this skill

Use when:
- Implementing the action selection loop
- Designing or modifying the scoring function
- Tuning rollout count, length, or sampling strategy
- Adding a learned value function on top of the hand-designed scoring rule

Do not use this skill for: tokenization, world model architecture, or simulator code.

## Control loop

At each 10 Hz control step:

1. Append the current observation token(s) to the history buffer.
2. For each candidate action `a` in the action set:
   a. Query the world model: condition on (history + a), sample N rollouts of length L observation tokens.
   b. Score each rollout with the value function.
   c. Take the mean (or a robust aggregate) over the N rollouts to get an estimated value V(a).
3. Pick `a* = argmax V(a)`.
4. Execute `a*`. Append `a*` to the history buffer. Loop.

The history buffer should hold the last `context_length` timesteps; older history is dropped. This is the same context length the world model was trained with.

## Latency budget

Rough cost per control step:

```
forward_passes_per_step ≈ num_actions × num_rollouts × rollout_length
```

With `num_actions = 6`, `num_rollouts = 8`, `rollout_length = 10`: ~480 forward passes per control step at 10 Hz. With KV caching, each forward pass is one transformer step, not a full sequence pass. Whether this closes on target hardware depends on model size and inference speed; measure early.

If the budget doesn't close, in priority order: (1) reduce `num_rollouts`, (2) reduce `rollout_length`, (3) reduce action set or use coarser actions, (4) shrink the model.

## Scoring function

Start with a hand-designed rule. Do not start with a learned value function. The hand-designed rule is interpretable, debuggable, requires no training, and gives a baseline that any learned function must beat.

**Starting scoring rule for a rollout:**

```
score(rollout) =
    w1 * expected_whiff_count
  + w2 * upwind_progress_estimate
  - w3 * blank_duration_penalty
  + w4 * concentration_increase_trend
```

Where:
- `expected_whiff_count`: number of non-BLANK concentration tokens in the rollout (either antenna)
- `upwind_progress_estimate`: dot product of cumulative agent displacement during the rollout with the estimated upwind direction. Requires tracking implied agent position from the action sequence; the rollout itself is plume tokens, but the actions taken during the rollout are known.
- `blank_duration_penalty`: penalize rollouts that go a long time without any whiff (the agent is wandering out of the plume)
- `concentration_increase_trend`: positive if mean whiff concentration in the second half of the rollout exceeds the first half (suggests approaching the source)

Initial weights: try `w1=1.0, w2=2.0, w3=0.5, w4=1.0`. Tune on simulator success rate, not on world model loss.

**Note:** rollouts beyond the immediate next action require choosing what action the agent takes during the rollout. Three options:
- Assume the agent repeats the candidate action (simplest, least realistic)
- Recursively plan inside the rollout (most accurate, expensive)
- Use a cheap default policy during rollouts (e.g., "continue forward unless blank for >0.5s, then cast")

Start with option 1; move to option 3 if the planner underperforms a reactive baseline.

## Aggregation across rollouts

`V(a) = mean(scores)` is the simple choice. Consider also:
- `V(a) = mean - λ * std` (risk-averse: prefer actions whose futures are consistently good)
- `V(a) = quantile(scores, 0.25)` (pessimistic: how good is the 25th-percentile future?)

For source-localization, slight risk-aversion tends to help — the agent prefers actions that reliably keep it in the plume over actions with high-variance outcomes.

## Optional: learned value function (Phase 2)

Once the hand-designed planner works, optionally train a learned value head:
- Freeze the world model
- Add a small MLP that maps the world model's final hidden state to a scalar value
- Train with TD learning or Monte Carlo returns from simulator rollouts where the agent uses the hand-designed planner as the behavior policy
- Replace the hand-designed scoring rule with the learned head only if it beats the baseline on held-out source-localization tasks

Do not jump to this phase before the hand-designed version works.

## Validation

The planner is validated end-to-end: success rate at locating the source within a fixed time budget across many simulator seeds, varied wind conditions, and varied initial agent positions. Targets and protocol live in `/validation/SKILL.md`.

## Common pitfalls

- **Replanning too often.** If the rollout cost is high, consider replanning every K control steps and committing to the chosen action for K steps. K=1 (every step) is the default; K=3 or K=5 is often fine and cuts compute.
- **Off-policy rollouts.** The world model was trained on simulator trajectories where the agent followed some policy. If the planner's actions take it far outside that distribution, world model predictions degrade. Mitigate by including diverse policies (random, reactive, infotaxis) in the training data — see `/simulator/SKILL.md`.
- **Hand-designed scoring rule overfitting to one wind regime.** Test the planner on wind conditions outside the training distribution. If it fails, the scoring rule is doing too much work that the world model should be doing.
- **Optimizing scoring weights on world model loss instead of navigation success.** These are not the same thing. Always tune on closed-loop success rate.

## Files in this directory

- `planner.py` — main control loop
- `scoring.py` — scoring functions (start with hand-designed rule)
- `rollout.py` — rollout sampling utilities (wraps world model inference)
- `value_head.py` — (Phase 2) learned value function
