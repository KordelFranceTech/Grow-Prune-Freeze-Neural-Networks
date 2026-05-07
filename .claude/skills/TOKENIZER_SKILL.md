# Tokenization Skill

## Purpose

Define and implement the discrete vocabulary that the simulator emits, the world model consumes, and the planner reasons over. Tokenization is the project's interface boundary — every other component depends on it being stable and statistically faithful.

## When to use this skill

Use when:
- Implementing or modifying the tokenizer / detokenizer
- Calibrating bin boundaries from simulator data
- Adding a new observation stream (e.g., an additional sensor)
- Validating that tokenization preserves navigation-relevant statistics

Do not use this skill for: world model architecture, training loops, or planner logic.

## Vocabulary

Multi-stream factored tokens. Each timestep emits one token from each stream; embeddings are summed before entering the transformer.

| Stream             | Vocab size | Tokens                                                              |
|--------------------|------------|---------------------------------------------------------------------|
| Left concentration | 7          | `L_BLANK`, `L_C0` … `L_C5`                                          |
| Right concentration| 7          | `R_BLANK`, `R_C0` … `R_C5`                                          |
| Wind (agent frame) | 8          | `W_0` … `W_7` (octants, 0 = directly ahead, increasing clockwise)   |
| Action             | 6          | `A_FWD`, `A_LEFT15`, `A_RIGHT15`, `A_TURN180`, `A_CAST_L`, `A_CAST_R` |
| Special            | 4          | `PAD`, `BOS`, `EOS`, `RESET`                                        |

Total embedding parameters: 7 + 7 + 8 + 6 + 4 = 32 entries × d_model.

## Concentration bin boundaries

Bins are log-spaced multiples of the sensor noise floor σ:

| Token   | Range                           |
|---------|---------------------------------|
| BLANK   | c < 3σ                          |
| C0      | 3σ ≤ c < 6σ                     |
| C1      | 6σ ≤ c < 15σ                    |
| C2      | 15σ ≤ c < 45σ                   |
| C3      | 45σ ≤ c < 150σ                  |
| C4      | 150σ ≤ c < 600σ                 |
| C5      | c ≥ 600σ                        |

These multipliers are starting values. **Calibrate them from simulator data** by collecting log(c) histograms over many trajectories and choosing boundaries at equal-mass quantiles within whiffs (i.e., excluding blanks). Save calibrated boundaries in `/tokenization/bins.yaml` and version them.

## Wind tokenization

Quantize wind direction relative to agent heading into 8 octants (45° each). `W_0` is wind coming directly from in front of the agent; angles increase clockwise. Wind below a small magnitude threshold maps to a configurable default (typically `W_0`); document this choice in the config.

## Action tokenization

Discrete action set is fixed for now. If new actions are added, append to the end of the vocabulary — never reorder existing tokens, since this would invalidate trained checkpoints.

## Sequence layout

Observations and actions are interleaved at every timestep:

```
[BOS] [obs_t0] [act_t0] [obs_t1] [act_t1] ... [obs_tn] [act_tn] [EOS]
```

Each `[obs_ti]` expands to four parallel tokens (left conc, right conc, wind) — three streams whose embeddings are summed at that position. Each `[act_ti]` is a single token. So one timestep occupies two positions in the sequence: an observation position and an action position. Context length should be expressed in *timesteps*, not raw token positions, in configs.

## Validation tests (run before any retraining)

These live in `/validation/tokenization/` and must pass before a tokenizer change is merged.

1. **Round-trip statistics.** Tokenize a long simulator trajectory, detokenize back to bin-center values, and compare aggregate statistics to the original:
   - Whiff rate (fraction of timesteps with detection): match within 5%
   - Mean blank duration: match within 10%
   - Mean whiff concentration (in log space): match within 0.2 dex
   - Autocorrelation of detection signal at 100ms, 500ms, 1s lags: match within 0.05

2. **Mutual information lower bound.** Compute MI between a 1-second window of tokenized observations and the true direction-to-source bearing in the simulator. If MI < 1 bit, the tokenization is throwing away navigation-relevant signal. Investigate: usually too few concentration bins or too coarse a wind grid.

3. **Bilateral information preservation.** Check that the tokenized left/right streams preserve the asymmetry signal: condition on "source is to the agent's left" vs "source is to the agent's right" and verify the distribution over (L_token, R_token) pairs differs significantly. If bilateral asymmetry is lost in tokenization, bilateral sensing is wasted.

## Common pitfalls

- **Uniform binning of concentration.** Plume concentrations are heavy-tailed; uniform bins put 95% of mass in the lowest bin. Always log-space.
- **Treating BLANK as just "low concentration."** Blanks are qualitatively different and the model needs a dedicated symbol for them. Do not merge BLANK with C0.
- **Tokenizing wind in world frame instead of agent frame.** The agent's heading changes; wind direction must be rotated into the agent frame before tokenization, otherwise the same physical situation produces different tokens depending on heading.
- **Reordering tokens.** Token IDs are part of the trained model's frozen weights. Append, never reorder.

## Files in this directory

- `vocabulary.py` — token ID definitions, single source of truth
- `tokenizer.py` — observation tuple → token IDs
- `detokenizer.py` — token IDs → bin-center values (used for validation only)
- `bins.yaml` — calibrated concentration bin boundaries
- `calibrate.py` — script to recompute `bins.yaml` from a simulator trajectory dataset
