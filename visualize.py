"""
Standalone script: load a saved network and visualize one navigation episode.

Usage:
    python3 visualize.py                           # baseline PyTorch agent
    python3 visualize.py --agent gpf               # GPF agent
    python3 visualize.py --agent custom            # NumPy custom agent (no PyTorch)
    python3 visualize.py --agent custom --config configs/custom_agent_config.yaml
    python3 visualize.py --weights checkpoints/rl/network_final.pt --seed 5
    python3 visualize.py --save trajectory.png     # save instead of showing interactively
"""

import argparse
import os
import yaml

import numpy as np

from agent.env import PlumeNavigationEnv
from train import build_agent
from train_gpf import build_gpf
from train_custom import build_agent as build_custom_agent
from visualization.plot_trajectory import record_episode, plot_trajectory


_AGENT_DEFAULTS = {
    'baseline': ('checkpoints/rl/',     'network_best.pt'),
    'gpf':      ('checkpoints/gpf/',    'network_best_gpf.pt'),
    'custom':   ('checkpoints/custom/', 'network_best.npz'),
}

_TRAIN_SCRIPT = {
    'baseline': 'train.py',
    'gpf':      'train_gpf.py',
    'custom':   'train_custom.py',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='Config file. Defaults to configs/default.yaml for '
                             'baseline/gpf, configs/custom_agent_config.yaml for custom.')
    parser.add_argument('--agent', choices=['baseline', 'gpf', 'custom'],
                        default='baseline',
                        help='Which agent type to load (default: baseline)')
    parser.add_argument('--weights', default=None,
                        help='Explicit path to a weights file. '
                             'Defaults to the best checkpoint for the chosen agent.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Episode seed for reproducible visualization')
    parser.add_argument('--save', default=None,
                        help='Save path for the figure (e.g. trajectory.png). '
                             'If omitted the plot is shown interactively.')
    args = parser.parse_args()

    # Pick the right default config for each agent type
    if args.config is None:
        args.config = ('configs/custom_agent_config.yaml'
                       if args.agent == 'custom'
                       else 'configs/default.yaml')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg['simulator']
    tok_cfg = cfg['tokenization']
    rl_cfg  = cfg['rl_agent']
    gpf_cfg = cfg.get('gpf_agent', {})

    env = PlumeNavigationEnv(sim_cfg, tok_cfg, rl_cfg, seed=args.seed)

    if args.agent == 'gpf':
        agent = build_gpf(env, rl_cfg, gpf_cfg, seed=0)
    elif args.agent == 'custom':
        agent = build_custom_agent(env, rl_cfg, seed=0)
    else:
        agent = build_agent(env, rl_cfg, seed=0)

    default_dir, default_file = _AGENT_DEFAULTS[args.agent]
    weights_path = args.weights or os.path.join(default_dir, default_file)
    if not os.path.exists(weights_path):
        print(f"No weights file found at {weights_path}. "
              f"Run {_TRAIN_SCRIPT[args.agent]} first, or pass --weights.")
        return

    agent.load(weights_path)
    agent.epsilon = 0.0   # fully greedy for visualization
    print(f"Loaded {args.agent} weights from {weights_path}")
    if hasattr(agent, 'network') and hasattr(agent.network, 'n_hidden'):
        print(f"  Network depth: {agent.network.n_hidden} hidden layers")

    print(f"Recording episode (seed={args.seed}) ...")
    data = record_episode(agent, env, seed=args.seed)
    status = 'SUCCESS' if data['info']['success'] else 'TIMEOUT'
    print(f"Episode result: {status}  ({data['n_steps']} steps, "
          f"final dist={data['info']['dist_to_source']:.2f} m)")

    show = args.save is None
    plot_trajectory(
        data,
        filament_mass=sim_cfg['filament'].get('mass', 0.1),
        save_path=args.save,
        show=show,
    )


if __name__ == '__main__':
    main()
