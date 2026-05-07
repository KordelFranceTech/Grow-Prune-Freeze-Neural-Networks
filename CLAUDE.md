# Plume Navigation Agent — Project Guide

## What this project is

A lightweight reinforcement learning agent for chemical plume navigation: locating an odor source by following a turbulent plume back to its origin. The agent runs onboard a real robot with a bilateral chemical sensor pair (two antennae) and a wind direction estimate.

The defining architectural choice is **model-based planning with a learned world model**. At each control step the agent imagines several short probable futures of the plume conditioned on each candidate action, scores those futures, and picks the best action. This is closer to MuZero / Dreamer in spirit than to a pure model-free policy.

## Architecture (Option 1: World Model + Separate Policy)

There are three components and they are kept **strictly separate** in the code:

1. **Plume simulator** (`/simulator`) — generates training trajectories. Filament-based physics simulator (Farrell-Murlis style). Not used at inference time. This is the source of all training data; do not generate training data with an LLM.

2. **World model** (`/world_model`) — a small decoder-only transformer trained via next-token prediction on tokenized simulator trajectories. Inputs: tokenized observation history interleaved with action tokens. Outputs: probability distribution over next observation token(s). It does **not** output actions. Its only job is imagination.

3. **Planner / policy** (`/planner`) — at each control step, for each candidate action, samples N short rollouts from the world model, scores them with a value function, and picks the highest-scoring action. The scoring function starts as a hand-designed rule (interpretable, no RL training needed) and may later be replaced with a learned value head trained via RL on top of the frozen world model.

These three components have separate training, separate tests, and separate failure modes. Do not merge them into a single end-to-end model unless explicitly instructed.

## Tokenization (the contract between simulator, world model, and planner)

All three components agree on a tokenization scheme defined in `/tokenization`. This is the project's interface boundary; if you change tokenization, you retrain everything.

**Multi-stream factored tokens.** At each timestep (10 Hz), the observation is encoded as multiple parallel tokens whose embeddings are summed before entering the transformer:

- Left antenna concentration: 7 levels (BLANK + 6 log-spaced concentration bins above noise floor)
- Right antenna concentration: 7 levels (same scheme as left)
- Wind direction in agent frame: 8 octants
- Action: ~6 discrete actions (forward, turn ±15°, turn around, cast left, cast right)
- Special: PAD, BOS, EOS, RESET

Concentration bin boundaries are log-spaced multiples of the sensor noise floor; exact values are calibrated from simulator histograms (see `/tokenization/SKILL.md`).

**Sequence layout.** Observations and actions are interleaved:

```
[BOS] [obs_t-k] [act_t-k] [obs_t-k+1] [act_t-k+1] ... [obs_t] [act_t] [obs_t+1_predicted] ...
```

This is what makes the world model action-conditioned — it can predict "what happens if I take action a now" rather than only "what happens if I keep going."

## Real-time constraints

The end goal is onboard inference on resource-constrained hardware, ultimately implemented in pure C. This shapes several decisions:

- The world model must be small. Target: a few hundred thousand parameters, not millions. Layers, heads, and embedding dimension should be tuned to fit a latency budget, not to maximize validation loss.
- At each 10 Hz control step the planner does roughly (num_actions × num_rollouts × rollout_length) forward passes. Back-of-envelope this against the target hardware before committing to a model size.
- All Python training code should produce weights in a format that's trivial to export to a C inference implementation. No exotic operators; stick to standard attention, layer norm, and GELU/ReLU.
- Keep the architecture's hyperparameters (n_layers, n_heads, d_model, context_length) in a single config file. The "adaptive complexity" research direction depends on being able to swap these cleanly.

## What this project is NOT

- Not a language model project. The transformer is a sequence model over a tiny domain-specific vocabulary (~30 tokens total across all streams). Do not import GPT-2 weights, tokenizers, or BPE machinery.
- Not end-to-end RL. The world model is trained by supervised next-token prediction on simulator data. RL, if used at all, only enters at the planner / value-function stage on top of a frozen world model.
- Not a project where training data comes from an LLM. All training trajectories come from the physics simulator.

## Repository layout

```
/simulator/       Filament-based plume simulator + trajectory logging
/tokenization/    Tokenizer, detokenizer, bin calibration, vocabulary definitions
/world_model/     Transformer architecture, training loop, checkpoints
/planner/         Rollout-based planner, scoring functions, action selection
/validation/      Statistical tests for tokenization fidelity, world model accuracy,
                  end-to-end navigation success rate
/configs/         Hyperparameter configs (one per experiment)
/c_inference/     (Future) C implementation of transformer inference
```

Each top-level directory has its own `SKILL.md` describing what belongs there and how to build/test it.

## Conventions

- **Determinism.** All training and evaluation uses seeded RNGs. Simulator trajectories are reproducible from a seed + config.
- **Validation before integration.** Before plugging a new world model checkpoint into the planner, run the standalone world model validation suite (see `/validation/SKILL.md`). Before plugging a new tokenizer into training, run tokenization fidelity checks.
- **Config over code.** Hyperparameters live in YAML configs in `/configs`, not as defaults in function signatures. This makes the "adaptive complexity" experiments tractable.
- **No premature C optimization.** Build and validate everything in Python first. The C port is a downstream concern, not a constraint on the Python implementation's clarity.

## Decision log

When making non-obvious architectural choices (e.g., choosing rollout length, picking a scoring function, changing bin boundaries), append a short entry to `/DECISIONS.md` with the date, the choice, and the reasoning. This keeps the project debuggable months later.
