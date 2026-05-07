# Validation Skill

## Purpose

Test each component in isolation before integration, and test the full system end-to-end. Validation is what keeps the three-component architecture from collapsing into an untestable tangle.

## When to use this skill

Use when:
- Adding a new validation test
- Running the validation suite before merging changes
- Diagnosing why end-to-end navigation success has dropped

## Three levels of validation

### 1. Tokenization fidelity (run before retraining the world model)

Tests live in `/validation/tokenization/`. See `/tokenization/SKILL.md` for the specific tests:
- Round-trip statistics match (whiff rate, blank duration, autocorrelation)
- Mutual information lower bound (≥1 bit between 1-second token window and source bearing)
- Bilateral asymmetry preserved

Failure here means the tokenizer is throwing away signal. Fix before training.

### 2. World model accuracy (run before plugging a checkpoint into the planner)

Tests live in `/validation/world_model/`:

- **Held-out next-token accuracy.** Per-stream cross-entropy on trajectories from simulator seeds not used in training. Track over time; a sudden regression means a training bug.

- **Calibration.** For each predicted distribution, the predicted probability of the realized token should match its empirical frequency. Plot a reliability diagram per stream. Miscalibrated models give the planner wrong rollout statistics.

- **Rollout statistics match simulator.** Generate long rollouts from the world model conditioned on a real trajectory's history, then check that the rollouts' aggregate statistics match those of real simulator continuations from the same starting state. Specifically:
  - Whiff rate over the rollout
  - Blank duration distribution
  - Concentration distribution within whiffs
  - Bilateral correlation
  Match within ~15% on each. This is the most important world model test — it directly measures whether imagined futures resemble real futures.

- **Action-conditioning sanity check.** Condition on the same history but different next actions; the rollout statistics should differ in physically sensible ways (e.g., "turn upwind" should produce more whiffs in rollouts than "turn downwind" when the agent is inside a plume).

### 3. End-to-end navigation success (the only metric that matters in the end)

Tests live in `/validation/navigation/`:

- Run the full agent (tokenizer + world model + planner) on the evaluation scenario suite from `/simulator/SKILL.md`.
- Report per-scenario: success rate (source found within time budget), median time-to-source, distance-to-source at timeout.
- Compare against baselines:
  - Random walk (sanity floor)
  - Pure reactive policy (turn upwind on whiff, cast on blank) — this is the bar to beat
  - Infotaxis (if implementable in your setup) — strong classical baseline

If the full agent doesn't beat the reactive policy, the planning machinery is not earning its compute cost. Diagnose: is the world model accurate (level 2)? Is the scoring rule sensible (try hand-tracing rollouts)?

## Diagnostic flow when end-to-end success drops

1. Did the tokenization change? Run level 1.
2. Did the world model change? Run level 2.
3. Did the scoring rule or rollout config change? Compare against last known good config.
4. Did the simulator change? Regenerate a small held-out set and verify previous checkpoint's level 2 numbers reproduce.

Most regressions are caught at level 1 or 2 if the tests are run in order. Skipping levels and going straight to end-to-end debugging wastes time.

## Files in this directory

- `tokenization/` — tokenization fidelity tests
- `world_model/` — world model accuracy and calibration tests
- `navigation/` — end-to-end navigation tests
- `baselines/` — reactive policy, infotaxis, random walk implementations
- `run_all.py` — runs the full suite, reports pass/fail per test
