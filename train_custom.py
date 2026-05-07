"""
Training script for the custom NumPy Expected SARSA agent (custom_agent.py).

Usage:
    python3 train_custom.py
    python3 train_custom.py --config configs/custom_agent_config.yaml --seed 7
    python3 train_custom.py --no-viz
"""

import argparse
import os
import time
import yaml
import numpy as np
from tqdm import tqdm

from agent.env import PlumeNavigationEnv
from agent.custom_agent import ExpectedSARSA
from validation.evaluate import evaluate_agent, evaluate_reactive_baseline
from visualization.plot_trajectory import record_episode, plot_trajectory


def build_agent(env, rl_cfg, seed=0):
    net_cfg = rl_cfg['network']
    return ExpectedSARSA(
        state_shape=env.state_shape,
        n_actions=env.n_actions,
        hidden_dim=net_cfg['hidden_dim'],
        lr=net_cfg['learning_rate'],
        gamma=rl_cfg['gamma'],
        epsilon_start=rl_cfg['epsilon_start'],
        epsilon_end=rl_cfg['epsilon_end'],
        epsilon_decay=rl_cfg['epsilon_decay'],
        seed=seed,
    )


def train(cfg, seed=None, visualize=True):
    sim_cfg = cfg['simulator']
    tok_cfg = cfg['tokenization']
    rl_cfg  = cfg['rl_agent']

    effective_seed = seed if seed is not None else cfg.get('seed', 42)
    rng = np.random.default_rng(effective_seed)

    env   = PlumeNavigationEnv(sim_cfg, tok_cfg, rl_cfg, seed=effective_seed)
    agent = build_agent(env, rl_cfg, seed=effective_seed)

    net_cfg    = rl_cfg['network']
    n_episodes = rl_cfg['n_episodes']
    eval_every = rl_cfg.get('eval_every_episodes', 500)
    save_dir   = rl_cfg.get('output_dir', 'checkpoints/custom/')
    os.makedirs(save_dir, exist_ok=True)

    input_dim = sum(env.state_shape)
    print(f"Training NumPy Expected SARSA  —  {n_episodes} episodes")
    print(f"  Network: {input_dim} → {net_cfg['hidden_dim']} → {env.n_actions}")
    print(f"  State space: {env.state_shape}  |  lr={net_cfg['learning_rate']}")

    print("\nEvaluating reactive baseline...")
    baseline = evaluate_reactive_baseline(sim_cfg, tok_cfg, rl_cfg, n_episodes=200)
    print(f"  Reactive baseline: {baseline['success_rate']:.2%}  "
          f"({baseline['mean_steps']:.0f} mean steps)")

    best_success_rate = 0.0
    t0 = time.time()

    for ep in tqdm(range(1, n_episodes + 1), desc="Training"):
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
            elapsed = time.time() - t0
            print(f"\n[ep {ep:>6}]  "
                  f"success={results['success_rate']:.2%}  "
                  f"mean_steps={results['mean_steps']:.0f}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"elapsed={elapsed:.0f}s")

            if results['success_rate'] > best_success_rate:
                best_success_rate = results['success_rate']
                agent.save(os.path.join(save_dir, 'network_best'))

    agent.save(os.path.join(save_dir, 'network_final'))

    print(f"\nTraining complete.")
    print(f"  Best success rate : {best_success_rate:.2%}")
    print(f"  Reactive baseline : {baseline['success_rate']:.2%}")
    beat = "YES" if best_success_rate > baseline['success_rate'] else "NO"
    print(f"  Beats baseline    : {beat}")

    if visualize:
        print("\nRecording best-policy episode for visualization...")
        best_path = os.path.join(save_dir, 'network_best.npz')
        if os.path.exists(best_path):
            agent.load(best_path)

        data = record_episode(agent, env, seed=0)
        plot_path = os.path.join(save_dir, 'trajectory.png')
        plot_trajectory(data,
                        filament_mass=sim_cfg['filament'].get('mass', 0.1),
                        save_path=plot_path,
                        show=False)

    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/custom_agent_config.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-viz', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, seed=args.seed, visualize=not args.no_viz)


if __name__ == '__main__':
    main()
