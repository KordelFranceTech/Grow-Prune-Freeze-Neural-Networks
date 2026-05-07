# GPF Autoresearch Experiments Log

Tracking GPF Expected SARSA hyperparameter optimization.
Each entry represents one fixed-budget training run.
Goal: monotonically improving best success rate across commits.

Metric: **best navigation success rate** over 3 000 training episodes.
Baseline: reactive chemotaxis policy (whiff-surge/cast).


---

## exp1_baseline_measurement  —  2026-05-07 12:08

| Field | Value |
|---|---|
| **Best success rate** | **13.50%** |
| Final success rate | 0.00% |
| Reactive baseline | 0.00% |
| Episodes (budget) | 3000 |
| Seed | 42 |
| Elapsed | 1617s |
| Grow events | 5 |
| Prune events | 5 |
| Freeze events | 5 |

**Learning curve:** ep500:13.5%  ep1000:2.0%  ep1500:0.5%  ep2000:11.5%  ep2500:6.5%  ep3000:0.0%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.005  M_add=50  cooldown=200  warm_up=300
- tau_prune=1e-08  prune_accum=500
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** Baseline measurement. Current default.yaml GPF settings, no code changes. Establishes the floor for subsequent experiments.

