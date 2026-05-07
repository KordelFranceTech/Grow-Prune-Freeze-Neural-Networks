"""
Side-by-side training: baseline Expected SARSA vs GPF Expected SARSA.

Usage:
    python3 train_gpf.py
    python3 train_gpf.py --config configs/default.yaml --seed 7
    python3 train_gpf.py --no-viz
    python3 train_gpf.py --gpf-only      # skip baseline, useful for fast iteration

Outputs (written to gpf_agent.output_dir, default checkpoints/gpf/):
    network_best_baseline.pt    best baseline checkpoint
    network_best_gpf.pt         best GPF checkpoint
    comparison.png              4-panel comparison figure
"""

import argparse
import os
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent.env import PlumeNavigationEnv
from agent.expected_sarsa import ExpectedSARSA
from agent.gpf_expected_sarsa import GPFExpectedSARSA
from validation.evaluate import evaluate_agent, evaluate_reactive_baseline


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def build_baseline(env, rl_cfg, seed=0):
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


def build_gpf(env, rl_cfg, gpf_cfg, seed=0):
    net_cfg = rl_cfg['network']
    grow = gpf_cfg.get('grow', {})
    belief = gpf_cfg.get('belief', {})
    prune = gpf_cfg.get('prune', {})
    freeze = gpf_cfg.get('freeze', {})
    # Estimate total training steps for φh = 1/kmax belief scaling
    n_total = rl_cfg.get('n_episodes', 10000) * rl_cfg.get('max_steps_per_episode', 600)
    return GPFExpectedSARSA(
        state_shape=env.state_shape,
        n_actions=env.n_actions,
        hidden_dim=net_cfg['hidden_dim'],
        lr=net_cfg['learning_rate'],
        gamma=rl_cfg['gamma'],
        epsilon_start=rl_cfg['epsilon_start'],
        epsilon_end=rl_cfg['epsilon_end'],
        epsilon_decay=rl_cfg['epsilon_decay'],
        eps_add=grow.get('eps_add', 0.01),
        M_add=grow.get('M_add', 20),
        cooldown_episodes=grow.get('cooldown_episodes', 10),
        min_episodes_before_grow=grow.get('min_episodes_before_grow', 30),
        max_hidden_layers=grow.get('max_hidden_layers', 6),
        ema_beta=grow.get('ema_beta', 0.9),
        tau_belief_harden=belief.get('tau_belief_harden', 0.02),
        tau_belief_prune=belief.get('tau_belief_prune', 0.3),
        n_total_steps=n_total,
        tau_prune=prune.get('tau_prune', 1e-8),
        prune_accum_steps=prune.get('prune_accum_steps', 500),
        tau_freeze_delta=freeze.get('tau_freeze_delta', 0.01),
        tau_freeze_frac=freeze.get('tau_freeze_frac', 0.9),
        freeze_check_interval=freeze.get('freeze_check_interval', 300),
        freeze_patience=freeze.get('freeze_patience', 3),
        min_episodes_before_freeze=freeze.get('min_episodes_before_freeze', 100),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Single-agent training loop
# ---------------------------------------------------------------------------

def _run_training_loop(agent, env, n_episodes, eval_every, save_dir,
                       save_prefix, rng, label):
    """Train *agent* for n_episodes, evaluate every eval_every episodes.

    Returns
    -------
    eval_points : list of int   episode numbers where eval was done
    success_rates : list of float
    td_errors_snapshot : list of float   rolling mean TD error (sampled at eval)
    """
    best_sr = 0.0
    eval_points = []
    success_rates = []
    td_snapshots = []

    for ep in tqdm(range(1, n_episodes + 1), desc=label, leave=False):
        state = env.reset()
        done = False
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

            # Rolling mean TD error over the last eval_every steps
            if hasattr(agent, 'td_error_history') and agent.td_error_history:
                window = agent.td_error_history[-eval_every:]
                td_snapshots.append(float(np.mean(window)))
            else:
                td_snapshots.append(float('nan'))

            # GPF-specific diagnostics
            n_hidden_str = ''
            if hasattr(agent, 'network') and hasattr(agent.network, 'n_hidden'):
                n_hidden_str = f"  layers={agent.network.n_hidden}"
                agent.layer_history.append(agent.network.n_hidden)

            print(f"\n[{label} ep {ep:>6}]  "
                  f"success={sr:.2%}  "
                  f"mean_steps={results['mean_steps']:.0f}  "
                  f"ε={agent.epsilon:.3f}"
                  f"{n_hidden_str}")

            if sr > best_sr:
                best_sr = sr
                agent.save(os.path.join(save_dir, f'network_best_{save_prefix}'))

    agent.save(os.path.join(save_dir, f'network_final_{save_prefix}'))
    return eval_points, success_rates, td_snapshots


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _plot_comparison(baseline_data, gpf_agent, save_path):
    """4-panel comparison figure."""
    bl_eps, bl_sr, bl_td = baseline_data
    gpf_eps, gpf_sr, gpf_td = gpf_agent['eval']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Baseline vs GPF Expected SARSA — Plume Navigation', fontsize=13)

    # Panel 1: Success rate comparison
    ax = axes[0, 0]
    ax.plot(bl_eps, [s * 100 for s in bl_sr], label='Baseline', color='steelblue')
    ax.plot(gpf_eps, [s * 100 for s in gpf_sr], label='GPF', color='darkorange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success rate (%)')
    ax.set_title('Navigation success rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Rolling mean TD error
    ax = axes[0, 1]
    valid_bl = [(e, t) for e, t in zip(bl_eps, bl_td) if not np.isnan(t)]
    valid_gpf = [(e, t) for e, t in zip(gpf_eps, gpf_td) if not np.isnan(t)]
    if valid_bl:
        ax.plot(*zip(*valid_bl), label='Baseline', color='steelblue')
    if valid_gpf:
        ax.plot(*zip(*valid_gpf), label='GPF', color='darkorange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean |TD error|')
    ax.set_title('TD error (lower = tighter value estimates)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: GPF layer depth over time
    ax = axes[1, 0]
    lh = gpf_agent['layer_history']
    if lh:
        lh_eps = gpf_eps[:len(lh)]
        ax.step(lh_eps, lh, where='post', color='darkorange')
        ax.set_ylim(0, max(lh) + 1)
    else:
        ax.text(0.5, 0.5, 'No layer changes', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Hidden layers')
    ax.set_title('GPF network depth over time')
    ax.grid(True, alpha=0.3)

    # Panel 4: GPF prune events (keep ratio bar chart)
    ax = axes[1, 1]
    pe = gpf_agent['prune_events']
    if pe:
        steps, ratios = zip(*pe)
        ax.bar(range(len(steps)), [r * 100 for r in ratios], color='darkorange', alpha=0.7)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f'{s}' for s in steps], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Weights kept (%)')
        ax.set_title(f'GPF prune events ({len(steps)} total)')
    else:
        ax.text(0.5, 0.5, 'No prune events', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('GPF prune events')
    ax.set_xlabel('Training step at prune')
    ax.grid(True, alpha=0.3, axis='y')

    # Freeze event annotations on panel 3
    fe = gpf_agent['freeze_events']
    if fe and lh:
        ax3 = axes[1, 0]
        freeze_steps = [s for s, _ in fe]
        # Map training steps to nearest eval episode for x-axis
        for s, idx in fe:
            ax3.axvline(x=s // max(gpf_eps[0], 1) * gpf_eps[0] if gpf_eps else s,
                        color='purple', alpha=0.4, linewidth=1, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(cfg, seed=None, run_baseline=True, visualize=True):
    sim_cfg = cfg['simulator']
    tok_cfg = cfg['tokenization']
    rl_cfg = cfg['rl_agent']
    gpf_cfg = cfg.get('gpf_agent', {})

    effective_seed = seed if seed is not None else cfg.get('seed', 42)
    rng_baseline = np.random.default_rng(effective_seed)
    rng_gpf = np.random.default_rng(effective_seed + 1)

    save_dir = gpf_cfg.get('output_dir', 'checkpoints/gpf/')
    os.makedirs(save_dir, exist_ok=True)

    n_episodes = rl_cfg['n_episodes']
    eval_every = rl_cfg.get('eval_every_episodes', 500)

    env_baseline = PlumeNavigationEnv(sim_cfg, tok_cfg, rl_cfg, seed=effective_seed)
    env_gpf = PlumeNavigationEnv(sim_cfg, tok_cfg, rl_cfg, seed=effective_seed + 1)

    print("\nEvaluating reactive baseline...")
    baseline_policy = evaluate_reactive_baseline(sim_cfg, tok_cfg, rl_cfg, n_episodes=200)
    print(f"  Reactive baseline: {baseline_policy['success_rate']:.2%}  "
          f"({baseline_policy['mean_steps']:.0f} mean steps)")

    # ------------------------------------------------------------------
    # Baseline training
    # ------------------------------------------------------------------
    bl_eval = ([], [], [])
    if run_baseline:
        print(f"\n{'='*60}")
        print(f"Training baseline Expected SARSA — {n_episodes} episodes")
        baseline_agent = build_baseline(env_baseline, rl_cfg, seed=effective_seed)
        t0 = time.time()
        bl_eval = _run_training_loop(
            baseline_agent, env_baseline, n_episodes, eval_every,
            save_dir, 'baseline', rng_baseline, 'Baseline'
        )
        elapsed = time.time() - t0
        best_bl = max(bl_eval[1]) if bl_eval[1] else 0.0
        print(f"\nBaseline complete: best success={best_bl:.2%}  ({elapsed:.0f}s)")
    else:
        print("\n[skipping baseline training, --gpf-only flag set]")

    # ------------------------------------------------------------------
    # GPF training
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training GPF Expected SARSA — {n_episodes} episodes")
    gpf_agent = build_gpf(env_gpf, rl_cfg, gpf_cfg, seed=effective_seed)

    input_dim = sum(env_gpf.state_shape)
    net_cfg = rl_cfg['network']
    print(f"  Initial network: {input_dim} → {net_cfg['hidden_dim']} → {env_gpf.n_actions}")
    print(f"  GPF grow/prune/freeze enabled")

    t0 = time.time()
    gpf_eval = _run_training_loop(
        gpf_agent, env_gpf, n_episodes, eval_every,
        save_dir, 'gpf', rng_gpf, 'GPF'
    )
    elapsed = time.time() - t0
    best_gpf = max(gpf_eval[1]) if gpf_eval[1] else 0.0
    print(f"\nGPF complete: best success={best_gpf:.2%}  ({elapsed:.0f}s)")
    print(f"  Final depth: {gpf_agent.network.n_hidden} hidden layers")
    print(f"  Grow events: {gpf_agent.network.n_hidden - 1}")
    print(f"  Prune events: {len(gpf_agent.prune_events)}")
    print(f"  Freeze events: {len(gpf_agent.freeze_events)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary")
    print(f"  Reactive baseline : {baseline_policy['success_rate']:.2%}")
    if run_baseline:
        beat = 'YES' if best_bl > baseline_policy['success_rate'] else 'NO'
        print(f"  Baseline best     : {best_bl:.2%}  (beats reactive: {beat})")
    beat_gpf = 'YES' if best_gpf > baseline_policy['success_rate'] else 'NO'
    print(f"  GPF best          : {best_gpf:.2%}  (beats reactive: {beat_gpf})")
    if run_baseline:
        delta = best_gpf - best_bl
        sign = '+' if delta >= 0 else ''
        print(f"  GPF vs baseline   : {sign}{delta:.2%}")

    # ------------------------------------------------------------------
    # Comparison plot
    # ------------------------------------------------------------------
    if visualize:
        plot_path = os.path.join(save_dir, 'comparison.png')
        _plot_comparison(
            baseline_data=bl_eval,
            gpf_agent={
                'eval': gpf_eval,
                'layer_history': gpf_agent.layer_history,
                'prune_events': gpf_agent.prune_events,
                'freeze_events': gpf_agent.freeze_events,
            },
            save_path=plot_path,
        )

    return baseline_agent if run_baseline else None, gpf_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip comparison plot at the end')
    parser.add_argument('--gpf-only', action='store_true',
                        help='Skip baseline training; only train GPF agent')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    compare(cfg,
            seed=args.seed,
            run_baseline=not args.gpf_only,
            visualize=not args.no_viz)


if __name__ == '__main__':
    main()
