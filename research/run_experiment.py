"""
Fixed-budget GPF experiment runner (autoresearch pattern).

Each call trains a GPF agent for a fixed episode budget, evaluates it,
and appends a structured result to research/experiments_log.md.

Usage:
    python3 research/run_experiment.py --config research/configs/exp1.yaml
    python3 research/run_experiment.py --config research/configs/exp2.yaml --tag "optimizer_fix"
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.env import PlumeNavigationEnv
from agent.gpf_expected_sarsa import GPFExpectedSARSA
from validation.evaluate import evaluate_agent, evaluate_reactive_baseline
from train_gpf import build_gpf


LOG_PATH = os.path.join(os.path.dirname(__file__), 'experiments_log.md')


def run_experiment(cfg, tag='', seed=None):
    sim_cfg = cfg['simulator']
    tok_cfg = cfg['tokenization']
    rl_cfg  = cfg['rl_agent']
    gpf_cfg = cfg.get('gpf_agent', {})
    exp_cfg = cfg.get('experiment', {})

    n_episodes  = exp_cfg.get('n_episodes', rl_cfg.get('n_episodes', 3000))
    eval_every  = exp_cfg.get('eval_every', 500)
    save_dir    = gpf_cfg.get('output_dir', 'checkpoints/gpf/')
    os.makedirs(save_dir, exist_ok=True)

    effective_seed = seed if seed is not None else cfg.get('seed', 42)
    rng = np.random.default_rng(effective_seed + 999)

    print(f"\n{'='*60}")
    print(f"Experiment: {tag or 'unnamed'}  |  seed={effective_seed}  |  n_episodes={n_episodes}")
    print(f"{'='*60}")

    env   = PlumeNavigationEnv(sim_cfg, tok_cfg, rl_cfg, seed=effective_seed)
    agent = build_gpf(env, rl_cfg, gpf_cfg, seed=effective_seed)

    print("\nEvaluating reactive baseline...")
    reactive = evaluate_reactive_baseline(sim_cfg, tok_cfg, rl_cfg, n_episodes=200)
    reactive_sr = reactive['success_rate']
    print(f"  Reactive baseline: {reactive_sr:.2%}")

    best_sr = 0.0
    eval_points = []
    success_rates = []
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done  = False
        while not done:
            action = agent.select_action(state, rng)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()

        if ep % eval_every == 0:
            results = evaluate_agent(agent, env, n_episodes=200)
            sr = results['success_rate']
            eval_points.append(ep)
            success_rates.append(sr)

            layers_str = ''
            if hasattr(agent, 'network'):
                layers_str = f"  layers={agent.network.n_hidden}"

            elapsed = time.time() - t0
            print(f"  [ep {ep:>5}]  success={sr:.2%}  "
                  f"steps={results['mean_steps']:.0f}  "
                  f"ε={agent.epsilon:.3f}{layers_str}  "
                  f"({elapsed:.0f}s elapsed)")

            if sr > best_sr:
                best_sr = sr
                agent.save(os.path.join(save_dir, 'network_best_gpf'))

    elapsed_total = time.time() - t0
    final_sr = success_rates[-1] if success_rates else 0.0
    grow_events = agent.network.n_hidden - 1
    prune_events = len(agent.prune_events)
    freeze_events = len(agent.freeze_events)

    print(f"\nResult: best_success={best_sr:.2%}  final_success={final_sr:.2%}  "
          f"({elapsed_total:.0f}s)")
    print(f"  Network: {agent.network.n_hidden} layers  "
          f"grow={grow_events}  prune={prune_events}  freeze={freeze_events}")
    print(f"  Beats reactive ({reactive_sr:.2%}): {'YES' if best_sr > reactive_sr else 'NO'}")

    # -------------------------------------------------------
    # Append to log
    # -------------------------------------------------------
    _append_to_log(
        tag=tag,
        cfg=cfg,
        n_episodes=n_episodes,
        best_sr=best_sr,
        final_sr=final_sr,
        reactive_sr=reactive_sr,
        eval_points=eval_points,
        success_rates=success_rates,
        grow_events=grow_events,
        prune_events=prune_events,
        freeze_events=freeze_events,
        elapsed=elapsed_total,
        seed=effective_seed,
    )

    return {
        'best_sr': best_sr,
        'final_sr': final_sr,
        'reactive_sr': reactive_sr,
        'eval_points': eval_points,
        'success_rates': success_rates,
        'grow_events': grow_events,
        'prune_events': prune_events,
        'freeze_events': freeze_events,
    }


def _append_to_log(tag, cfg, n_episodes, best_sr, final_sr, reactive_sr,
                   eval_points, success_rates, grow_events, prune_events,
                   freeze_events, elapsed, seed):
    rl_cfg  = cfg['rl_agent']
    gpf_cfg = cfg.get('gpf_agent', {})
    grow    = gpf_cfg.get('grow', {})
    prune   = gpf_cfg.get('prune', {})
    freeze  = gpf_cfg.get('freeze', {})
    exp_cfg = cfg.get('experiment', {})

    curve_str = '  '.join(f"ep{e}:{s:.1%}" for e, s in zip(eval_points, success_rates))

    entry = f"""
---

## {tag or 'experiment'}  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}

| Field | Value |
|---|---|
| **Best success rate** | **{best_sr:.2%}** |
| Final success rate | {final_sr:.2%} |
| Reactive baseline | {reactive_sr:.2%} |
| Episodes (budget) | {n_episodes} |
| Seed | {seed} |
| Elapsed | {elapsed:.0f}s |
| Grow events | {grow_events} |
| Prune events | {prune_events} |
| Freeze events | {freeze_events} |

**Learning curve:** {curve_str}

**Key hyperparameters:**
- lr={rl_cfg['network']['learning_rate']}  γ={rl_cfg['gamma']}  ε_decay={rl_cfg['epsilon_decay']}
- eps_add={grow.get('eps_add')}  M_add={grow.get('M_add')}  cooldown={grow.get('cooldown_episodes')}  warm_up={grow.get('min_episodes_before_grow')}
- tau_prune={prune.get('tau_prune')}  prune_accum={prune.get('prune_accum_steps')}
- tau_freeze_delta={freeze.get('tau_freeze_delta')}  tau_freeze_frac={freeze.get('tau_freeze_frac')}  patience={freeze.get('freeze_patience')}

**Notes:** {exp_cfg.get('notes', '')}
"""

    with open(LOG_PATH, 'a') as f:
        f.write(entry)
    print(f"\nResults appended to {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Experiment config YAML')
    parser.add_argument('--tag',    default='',    help='Short label for this experiment')
    parser.add_argument('--seed',   type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg, tag=args.tag or cfg.get('name', ''), seed=args.seed)


if __name__ == '__main__':
    main()
