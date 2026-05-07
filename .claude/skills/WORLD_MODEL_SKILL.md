# World Model Skill

## Purpose

Train and serve a small decoder-only transformer that predicts probability distributions over future plume tokens, conditioned on a history of (observation, action) pairs. The world model is **only an imagination engine** — it does not output actions.

## When to use this skill

Use when:
- Designing or modifying the transformer architecture
- Implementing the training loop
- Running inference for the planner's rollouts
- Saving/loading checkpoints

Do not use this skill for: tokenization, action selection, or simulator code.

## Architectural constraints

This is a small transformer for a small vocabulary. Anchoring values for a starting point:

- `d_model`: 64–128
- `n_layers`: 2–4
- `n_heads`: 2–4
- `context_length`: 32–64 timesteps (each timestep = 2 positions: obs + action), so 64–128 token positions
- Vocabulary: ~32 tokens total across streams
- Total parameters target: a few hundred thousand, not millions

These are starting points. The "adaptive complexity" research direction will sweep over them.

**Architecture details:**
- Decoder-only, causal (masked) self-attention
- Multi-stream input embedding: at each observation position, embeddings from left-concentration, right-concentration, and wind streams are summed. Action positions use a single action embedding.
- Standard positional embeddings (learned or sinusoidal — either is fine; learned is simpler to port to C)
- Pre-norm layer ordering (more stable for small models)
- GELU or ReLU activation (ReLU is simpler in C; pick one and stick with it)
- Output head: separate logits projection per stream (left conc, right conc, wind). The model predicts the next observation as three parallel distributions, not as a single joint distribution.
- No dropout at inference; light dropout (0.1) during training

**Constraints from the future C port:**
- Only standard ops: matmul, layer norm, softmax, GELU/ReLU, embedding lookup
- No flash attention, no fused kernels — vanilla attention is what gets ported
- Weights exportable as raw float32 arrays with a documented layout

## Training

**Data.** Tokenized trajectories from the filament simulator (see `/simulator/SKILL.md`). Each trajectory is a sequence of (observation, action) pairs at 10 Hz. Aim for at least 10⁶ timesteps total across diverse wind conditions and source positions.

**Objective.** Standard next-token cross-entropy, summed across the three observation streams. At each observation position, the model predicts the next observation's left-conc, right-conc, and wind tokens; loss is the sum of three cross-entropies. Action positions are conditioning context — the model does not predict actions, so action-position outputs are masked from the loss.

**Optimizer.** AdamW with cosine schedule, warmup over the first ~1% of steps. Starting LR around 3e-4 for a model this size.

**Batch shape.** `(batch, n_timesteps, n_streams)` where `n_streams = 4` (3 obs streams + 1 action stream that's null-padded at observation positions and active at action positions). Internally flatten to `(batch, 2 * n_timesteps)` token positions before feeding the transformer.

**Validation loss.** Per-stream cross-entropy on a held-out set of trajectories generated with simulator seeds not used in training. Track all three stream losses separately — if one stream's loss is much higher than the others, something is wrong with that stream's tokenization or embedding.

## Inference (for the planner)

The planner calls the world model in two modes:

1. **Encode history:** given the last k timesteps of (obs, action), run a forward pass to produce the cached attention state at the current position.
2. **Roll out futures:** given a candidate next action and a sample budget N, sample N continuations of length L. Each continuation is a sequence of predicted observation tokens (with the planner choosing actions between them, or with a fixed action assumed for the rollout — see `/planner/SKILL.md`).

Implement KV caching from day one — without it, rollouts are quadratic in length and won't meet the latency budget.

Sampling: temperature 1.0 by default. Top-k or nucleus sampling can be used to reduce variance in rollouts; document choice in config.

## Checkpoints

Save every N steps with full config snapshot. Filename convention: `wm_{config_hash}_{step}.pt`. A checkpoint is not "valid" until it has passed the validation suite (see `/validation/SKILL.md`); record validation results in a sidecar JSON.

## Common pitfalls

- **Predicting joint observation tokens instead of factored streams.** Use three separate output heads, not one giant joint vocabulary. Factored is more sample-efficient and gives interpretable per-stream losses.
- **Forgetting to mask action positions in the loss.** The model is not an action policy; loss on action positions corrupts training.
- **Context length too short.** Plume blanks can last seconds. A context shorter than the typical blank duration means the model can't condition on "I just lost the plume after a long whiff" — a navigation-critical state.
- **Using flash attention or other ops not portable to C.** Build for the deployment target from day one.

## Files in this directory

- `model.py` — transformer architecture
- `train.py` — training loop
- `infer.py` — inference utilities (forward pass, KV cache, sampling)
- `export.py` — checkpoint → C-friendly weight format
