# research/ — Experiment runner and log

## What belongs here

- `run_experiment.py` — trains one GPF agent from a config, prints a structured result line, and appends to `experiments_log.md`
- `experiments_log.md` — auto-generated table of every experiment run (best SR, final SR, grow/prune events, learning curve, key hyperparameters)
- `configs/` — one YAML per experiment; naming convention `expN_short_description.yaml`; `exp_best.yaml` is a stable alias for the current best config

## Running an experiment

```bash
# from project root
python3 research/run_experiment.py --config research/configs/exp_best.yaml
python3 research/run_experiment.py --config research/configs/expN_foo.yaml --tag my_tag
```

Results are appended to `research/experiments_log.md` automatically. Background (nohup) pattern:

```bash
nohup python3 -u research/run_experiment.py --config research/configs/expN.yaml \
  --tag expN > /tmp/expN_out.txt 2>&1 &
echo $!   # save PID
tail -f /tmp/expN_out.txt
```

## Config structure

All configs follow this schema (see `exp_best.yaml` for a fully annotated reference):

```
name / seed
experiment:   n_episodes, eval_every, notes
simulator:    dt, episode_length, domain_size, source_*, filament, wind, sensors, agent, success_radius
tokenization: vocabulary_version, concentration_bin_edges, wind_*, actions
rl_agent:     algorithm, network (hidden_dim, lr), gamma, epsilon_*, n_episodes, max_steps_per_episode,
              eval_every_episodes, output_dir, reward (source_found, time_penalty, whiff_bonus, ...)
gpf_agent:    output_dir, grow (eps_add, M_add, cooldown, min_episodes, max_hidden_layers, ema_beta,
              use_eval_trigger_only), belief (tau_belief_*), prune (tau_prune, prune_accum_steps,
              max_prune_events), freeze (tau_freeze_delta, tau_freeze_frac, freeze_check_interval,
              freeze_patience, min_episodes_before_freeze)
```

## Key hyperparameters and their effects (as of 2026-05-11)

| Parameter | Location | Effect |
|-----------|----------|--------|
| `episode_length` | `simulator` | **Most impactful.** 100s >> 50s. Eliminates timeout failures on hard starts. |
| `max_steps_per_episode` | `rl_agent` | Must match `episode_length / dt` (e.g. 100/0.1 = 1000; use 1200 for margin). |
| `eps_add` | `gpf_agent.grow` | Stagnation threshold for grow trigger. 0.05 = fire if <5% improvement. |
| `cooldown_episodes` / `min_episodes_before_grow` | `gpf_agent.grow` | Both set to 400 in best config — prevents premature grows. |
| `tau_prune` | `gpf_agent.prune` | OBD saliency threshold. 1e-6 keeps ~48% on first prune. Lower = more kept. |
| `max_prune_events` | `gpf_agent.prune` | Caps cascading prune destruction. Default 999 (no cap). Set to 1 to only prune after 1st grow. **Untested at episode_length=100s — next thing to try.** |
| `max_hidden_layers` | `gpf_agent.grow` | Hard depth cap. 4 in best config. |
| `min_episodes_before_freeze` | `gpf_agent.freeze` | Set to 99999 to disable freeze entirely (freeze is not yet beneficial). |

## Naming the next experiment

Check the last `expN` in `research/configs/` and `research/experiments_log.md`, then use N+1. As of 2026-05-11, the last experiment is exp13; the next should be `exp14_...`.

## Suggested next experiments

1. **exp14_prune_once_long_ep** — `exp_best.yaml` + `max_prune_events=1`. Tests whether capping prune events at 1 allows 3-layer and 4-layer networks to converge cleanly at 100s episode length, potentially exceeding 98.0%.

2. **exp15_wider_network** — increase `hidden_dim` from 64 to 128. Check if wider layers improve the 1-layer baseline ceiling (currently 95.5% at ep1000).

3. **exp16_target_network** — add a target network to `agent/gpf_expected_sarsa.py` for Q-value stability in longer runs (needed if experiments exceed ~6000 episodes).
