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


---

## exp1_baseline_measurement  —  2026-05-07 12:27

| Field | Value |
|---|---|
| **Best success rate** | **13.50%** |
| Final success rate | 0.00% |
| Reactive baseline | 0.00% |
| Episodes (budget) | 3000 |
| Seed | 42 |
| Elapsed | 1681s |
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


---

## exp2_no_grow_single_layer  —  2026-05-07 12:33

| Field | Value |
|---|---|
| **Best success rate** | **6.50%** |
| Final success rate | 0.00% |
| Reactive baseline | 0.00% |
| Episodes (budget) | 3000 |
| Seed | 42 |
| Elapsed | 992s |
| Grow events | 0 |
| Prune events | 0 |
| Freeze events | 0 |

**Learning curve:** ep500:6.5%  ep1000:1.5%  ep1500:2.5%  ep2000:0.5%  ep2500:0.0%  ep3000:0.0%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.005  M_add=50  cooldown=200  warm_up=99999
- tau_prune=1e-06  prune_accum=1000
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** No GPF grows. Single-layer 22→64→6 network trains for full 3000 episodes without depth changes. Tests whether stable single-layer convergence outperforms the grow-collapse pattern of Exp 1 (best=13.5%).


---

## exp2_env_fix_reward_shaping  —  2026-05-07 13:59

| Field | Value |
|---|---|
| **Best success rate** | **91.00%** |
| Final success rate | 87.00% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 3000 |
| Seed | 42 |
| Elapsed | 897s |
| Grow events | 0 |
| Prune events | 0 |
| Freeze events | 0 |

**Learning curve:** ep500:7.0%  ep1000:9.0%  ep1500:37.0%  ep2000:84.5%  ep2500:91.0%  ep3000:87.0%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.005  M_add=50  cooldown=200  warm_up=99999
- tau_prune=1e-06  prune_accum=1000
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** Environmental fixes: reflecting walls, 5x denser plume, distance shaping, stronger whiff reward. No GPF grows. Goal: demonstrate stable single-layer convergence above Exp 1 best (13.5% on broken environment).

