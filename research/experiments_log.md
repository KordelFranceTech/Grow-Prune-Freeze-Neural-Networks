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


---

## exp3_single_grow  —  2026-05-07 14:35

| Field | Value |
|---|---|
| **Best success rate** | **61.50%** |
| Final success rate | 50.00% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 3000 |
| Seed | 42 |
| Elapsed | 992s |
| Grow events | 1 |
| Prune events | 1 |
| Freeze events | 1 |

**Learning curve:** ep500:7.0%  ep1000:9.0%  ep1500:37.0%  ep2000:59.0%  ep2500:61.5%  ep3000:50.0%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.03  M_add=6  cooldown=2000  warm_up=2000
- tau_prune=1e-06  prune_accum=1000
- tau_freeze_delta=0.005  tau_freeze_frac=0.95  patience=5

**Notes:** One GPF grow at ep ~2000 (eval-triggered). Same env fixes as exp2. Tests whether depth increase after single-layer convergence improves results.


---

## exp4_wider_no_grow  —  2026-05-07 14:56

| Field | Value |
|---|---|
| **Best success rate** | **85.50%** |
| Final success rate | 83.00% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 3000 |
| Seed | 42 |
| Elapsed | 972s |
| Grow events | 0 |
| Prune events | 0 |
| Freeze events | 0 |

**Learning curve:** ep500:30.0%  ep1000:29.0%  ep1500:85.5%  ep2000:80.5%  ep2500:51.5%  ep3000:83.0%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.03  M_add=6  cooldown=500  warm_up=99999
- tau_prune=1e-06  prune_accum=1200
- tau_freeze_delta=0.005  tau_freeze_frac=0.95  patience=5

**Notes:** 128-wide network, no grows (min_episodes_before_grow=99999). Tests whether wider base network alone beats exp2's 91% best. Same env as exp2 (source_strength=5, distance_shaping=2.0, reflecting walls).


---

## exp5_lower_lr_longer  —  2026-05-07 16:34

| Field | Value |
|---|---|
| **Best success rate** | **58.00%** |
| Final success rate | 58.00% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 4000 |
| Seed | 42 |
| Elapsed | 5519s |
| Grow events | 0 |
| Prune events | 0 |
| Freeze events | 0 |

**Learning curve:** ep500:11.5%  ep1000:37.0%  ep1500:16.5%  ep2000:39.5%  ep2500:36.5%  ep3000:49.0%  ep3500:41.5%  ep4000:58.0%

**Key hyperparameters:**
- lr=0.0005  γ=0.99  ε_decay=0.9995
- eps_add=0.03  M_add=6  cooldown=500  warm_up=99999
- tau_prune=1e-06  prune_accum=1200
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** 64-wide, lr=5e-4 (half of exp2), 4000 episodes. Same env as exp2. Tests whether lower lr stabilizes the policy at/above 91% for longer, enabling the extra 1000 episodes to push best success rate above 91%.


---

## exp6_early_grow  —  2026-05-07 19:07

| Field | Value |
|---|---|
| **Best success rate** | **90.50%** |
| Final success rate | 90.50% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 4000 |
| Seed | 42 |
| Elapsed | 1889s |
| Grow events | 1 |
| Prune events | 1 |
| Freeze events | 0 |

**Learning curve:** ep500:11.5%  ep1000:37.0%  ep1500:16.5%  ep2000:39.5%  ep2500:9.0%  ep3000:66.0%  ep3500:88.5%  ep4000:90.5%

**Key hyperparameters:**
- lr=0.0005  γ=0.99  ε_decay=0.9995
- eps_add=0.03  M_add=6  cooldown=500  warm_up=500
- tau_prune=1e-06  prune_accum=1200
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** 64-wide, lr=5e-4, one eval-triggered grow at ep ~1000 (early stagnation), 4000 episodes total. use_eval_trigger_only prevents TD-error trigger. Tests: does early depth increase push beyond exp5's single-layer ceiling?


---

## exp7_extended_single_grow  —  2026-05-07 22:54

| Field | Value |
|---|---|
| **Best success rate** | **92.50%** |
| Final success rate | 46.00% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 5000 |
| Seed | 42 |
| Elapsed | 1669s |
| Grow events | 2 |
| Prune events | 2 |
| Freeze events | 0 |

**Learning curve:** ep500:7.0%  ep1000:9.0%  ep1500:24.0%  ep2000:89.5%  ep2500:91.0%  ep3000:77.0%  ep3500:92.5%  ep4000:90.0%  ep4500:90.0%  ep5000:46.0%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.03  M_add=6  cooldown=500  warm_up=500
- tau_prune=1e-06  prune_accum=1200
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** Same as exp6 (64-wide, lr=1e-3, eval-triggered grow at ep ~2000) but with 5000 episode budget. Exp6 showed 90.5% at ep4000 still climbing — 1000 more episodes expected to push above exp2's 91% ceiling. Same grow config: grow fires when success rate improvement < 3% over recent evals.


---

## exp8_single_grow_capped  —  2026-05-08 01:13

| Field | Value |
|---|---|
| **Best success rate** | **92.50%** |
| Final success rate | 88.50% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 5000 |
| Seed | 42 |
| Elapsed | 1655s |
| Grow events | 1 |
| Prune events | 1 |
| Freeze events | 0 |

**Learning curve:** ep500:7.0%  ep1000:9.0%  ep1500:24.0%  ep2000:89.5%  ep2500:91.0%  ep3000:77.0%  ep3500:92.5%  ep4000:90.0%  ep4500:90.0%  ep5000:88.5%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.03  M_add=6  cooldown=500  warm_up=500
- tau_prune=1e-06  prune_accum=1200
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** Same as exp7 (64-wide, lr=1e-3, eval-triggered grow at ep~1000) but max_hidden_layers=2 prevents the disruptive 2nd grow. 2-layer network gets full 4000 eps post-grow to converge. Target: >92.5%.


---

## Experiment 9 — exp9_wider_two_layer (KILLED at ep4000)

| Field | Value |
|---|---|
| Config | exp9_wider_two_layer.yaml |
| **Best success rate** | **90.50%** (ep2000) |
| Final success rate | 86.00% (ep4000, killed) |
| Reactive baseline | 0.50% |
| Episodes run | 4000 / 5000 |
| Seed | 42 |
| Grow events | 1 (ep1000, 30.0%→29.0%, sr_improve=0) |
| Prune events | 1 (kept 42.66% of weights) |
| Freeze events | 0 |

**Learning curve:** ep500:30.0%  ep1000:29.0%  ep1500:0.0%  ep2000:90.5%  ep2500:66.5%  ep3000:87.5%  ep3500:44.5%  ep4000:86.0%

**Key hyperparameters:**
- hidden_dim=128  lr=0.001  γ=0.99
- tau_prune=1e-06  prune_accum=1200  max_hidden_layers=2

**Notes:** 128-wide 2-layer. OBD pruning at step~566077 kept only 42.66% of weights — catastrophic collapse to 0% at ep1500. Network recovered to 90.5% at ep2000 but oscillated wildly (0%→90.5%→66.5%→87.5%→44.5%→86.0%). Root cause: tau_prune=1e-6 too aggressive for 128-wide (per-weight gradient smaller due to gradient clipping over more params). Corrected config sets tau_prune=1e-8, prune_accum_steps=3000. Does NOT beat 92.5% bar.


---

## exp10_stronger_reward  —  2026-05-10 01:21

| Field | Value |
|---|---|
| **Best success rate** | **88.00%** |
| Final success rate | 35.00% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 5000 |
| Seed | 42 |
| Elapsed | 1939s |
| Grow events | 1 |
| Prune events | 1 |
| Freeze events | 0 |

**Learning curve:** ep500:14.5%  ep1000:2.5%  ep1500:59.5%  ep2000:26.0%  ep2500:68.0%  ep3000:42.0%  ep3500:88.0%  ep4000:48.0%  ep4500:34.0%  ep5000:35.0%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.03  M_add=6  cooldown=500  warm_up=500
- tau_prune=1e-07  prune_accum=2000
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** 64-wide, lr=1e-3, eval-triggered grow at ep~1000, max_hidden_layers=2, source_found=150 (vs 100 in all prior exps). Tests whether stronger terminal reward breaks the 92.5% policy ceiling.


---

## exp11_deeper_sparse  —  2026-05-10 02:01

| Field | Value |
|---|---|
| **Best success rate** | **94.50%** |
| Final success rate | 88.50% |
| Reactive baseline | 0.50% |
| Episodes (budget) | 5000 |
| Seed | 42 |
| Elapsed | 1660s |
| Grow events | 3 |
| Prune events | 2 |
| Freeze events | 0 |

**Learning curve:** ep500:7.0%  ep1000:9.0%  ep1500:11.5%  ep2000:91.5%  ep2500:25.0%  ep3000:94.0%  ep3500:91.5%  ep4000:94.5%  ep4500:56.5%  ep5000:88.5%

**Key hyperparameters:**
- lr=0.001  γ=0.99  ε_decay=0.9995
- eps_add=0.05  M_add=6  cooldown=400  warm_up=400
- tau_prune=1e-06  prune_accum=1500
- tau_freeze_delta=0.01  tau_freeze_frac=0.9  patience=3

**Notes:** 64-wide, 4-layer max, early-chain grow (ep~1000/1500/2000), tau_prune=1e-6. Tests whether GPF's depth+sparsity beats the 2-layer dense ceiling of 92.5%.

