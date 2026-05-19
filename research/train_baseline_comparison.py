"""
Train a static single-hidden-layer Expected SARSA agent using the exp_best config.

Saves eval curve to checkpoints/gpf/baseline_eval_exp_best.json and best
checkpoint to checkpoints/gpf/network_best_baseline_exp_best.pt.

DOES NOT touch network_best_gpf.pt or any other GPF file.

Usage (from project root):
    python3 research/train_baseline_comparison.py
    python3 research/train_baseline_comparison.py --config research/configs/exp_best.yaml
"""

import argparse
import json
import os
import sys
import time
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.env import PlumeNavigationEnv
from agent.expected_sarsa import ExpectedSARSA
from validation.evaluate import evaluate_agent


EVAL_OUT  = 'checkpoints/gpf/baseline_eval_exp_best.json'
CKPT_OUT  = 'checkpoints/gpf/network_best_baseline_exp_best'


def train(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg['simulator']
    tok_cfg = cfg['tokenization']
    rl_cfg  = cfg['rl_agent']

    seed        = cfg.get('seed', 42)
    n_episodes  = rl_cfg['n_episodes']
    eval_every  = rl_cfg.get('eval_every_episodes', 500)
    save_dir    = rl_cfg.get('output_dir', 'checkpoints/gpf/')
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    env = PlumeNavigationEnv(sim_cfg, tok_cfg, rl_cfg, seed=seed)

    net_cfg = rl_cfg['network']
    agent = ExpectedSARSA(
        state_shape   = env.state_shape,
        n_actions     = env.n_actions,
        hidden_dim    = net_cfg['hidden_dim'],
        lr            = net_cfg['learning_rate'],
        gamma         = rl_cfg['gamma'],
        epsilon_start = rl_cfg['epsilon_start'],
        epsilon_end   = rl_cfg['epsilon_end'],
        epsilon_decay = rl_cfg['epsilon_decay'],
        seed          = seed,
    )

    print(f"\n{'='*60}")
    print(f"Baseline Expected SARSA — {n_episodes} episodes  (seed={seed})")
    print(f"Architecture: {sum(env.state_shape)} → {net_cfg['hidden_dim']} → {env.n_actions}  (static, no GPF)")
    print(f"{'='*60}")

    best_sr      = 0.0
    eval_points  = []
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
            sr      = results['success_rate']
            elapsed = time.time() - t0
            eval_points.append(ep)
            success_rates.append(sr)

            print(f"  [ep {ep:>5}]  success={sr:.2%}  "
                  f"steps={results['mean_steps']:.0f}  "
                  f"ε={agent.epsilon:.3f}  ({elapsed:.0f}s)")

            if sr > best_sr:
                best_sr = sr
                agent.save(os.path.join(save_dir, 'network_best_baseline_exp_best'))

    elapsed_total = time.time() - t0
    final_sr = success_rates[-1] if success_rates else 0.0

    print(f"\nBaseline complete: best={best_sr:.2%}  final={final_sr:.2%}  ({elapsed_total:.0f}s)")
    print(f"Checkpoint: {os.path.join(save_dir, 'network_best_baseline_exp_best.pt')}")

    # Save eval curve for the plot script
    out = {
        'eval_episodes':   eval_points,
        'success_rates':   [round(s * 100, 1) for s in success_rates],
        'best_sr':         round(best_sr * 100, 1),
        'final_sr':        round(final_sr * 100, 1),
        'n_episodes':      n_episodes,
        'hidden_dim':      net_cfg['hidden_dim'],
        'seed':            seed,
        'elapsed_s':       round(elapsed_total, 1),
    }
    with open(EVAL_OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Eval curve saved to {EVAL_OUT}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='research/configs/exp_best.yaml')
    args = parser.parse_args()
    train(args.config)


if __name__ == '__main__':
    main()
