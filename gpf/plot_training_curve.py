"""
Generate a publication-quality training curve for exp11b (the best GPF checkpoint).

Usage:
    python3 gpf/plot_training_curve.py
    python3 gpf/plot_training_curve.py --out gpf/fig_training_curve.pdf

Output: fig_training_curve.pdf (or .png) — suitable for direct inclusion in LaTeX
as \includegraphics{gpf/fig_training_curve.pdf}.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Exp11b training data (from research/experiments_log.md)
# ---------------------------------------------------------------------------

EVAL_EPISODES   = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
SUCCESS_RATES   = [24.5, 95.5, 40.5, 97.5, 91.5, 98.0, 95.5, 96.5, 95.5, 68.0]

# (episode, label, keep_ratio_or_None)
GROW_EVENTS  = [2000, 3000, 4000]          # episode at which grow fired
PRUNE_EVENTS = [                           # (episode_after, keep_pct)
    (2000, 47.9),
    (3000, 19.6),
    (4000,  9.9),
]
LAYER_COUNTS = {                           # layers active AFTER each eval
    500: 1, 1000: 1, 1500: 1,
    2000: 2, 2500: 2,
    3000: 3, 3500: 3,
    4000: 4, 4500: 4, 5000: 4,
}
BEST_EP   = 3000
BEST_SR   = 98.0


def plot(out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))

    # --- main curve ---
    ax.plot(EVAL_EPISODES, SUCCESS_RATES,
            color='#2166ac', linewidth=1.8, marker='o', markersize=5,
            zorder=3, label='Evaluation success rate')

    # --- best checkpoint marker ---
    ax.scatter([BEST_EP], [BEST_SR],
               color='#d73027', s=80, zorder=5,
               label=f'Best checkpoint ({BEST_SR:.0f}%)')

    # --- grow event verticals ---
    for i, ep in enumerate(GROW_EVENTS):
        ax.axvline(ep, color='#4dac26', linewidth=1.2, linestyle='--', alpha=0.8,
                   label='Grow event' if i == 0 else None)

    # --- prune annotations (keep ratio) ---
    for i, (ep, keep) in enumerate(PRUNE_EVENTS):
        ax.annotate(f'prune\n{keep:.0f}% kept',
                    xy=(ep, SUCCESS_RATES[EVAL_EPISODES.index(ep)]),
                    xytext=(ep + 120, SUCCESS_RATES[EVAL_EPISODES.index(ep)] - 12),
                    fontsize=7, color='#7b3294',
                    arrowprops=dict(arrowstyle='->', color='#7b3294', lw=0.8),
                    ha='left')

    # --- layer depth secondary annotation strip ---
    y_strip = -8
    prev_layers = None
    for ep in EVAL_EPISODES:
        n = LAYER_COUNTS[ep]
        if n != prev_layers:
            ax.text(ep, y_strip, f'L={n}',
                    fontsize=7, ha='center', color='#555555',
                    transform=ax.get_xaxis_transform() if False else
                    ax.transData)
        prev_layers = n

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success rate (%)', fontsize=11)
    ax.set_title('GPF Expected SARSA — Plume Navigation Training Curve',
                 fontsize=11)
    ax.set_xlim(0, 5200)
    ax.set_ylim(-5, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, alpha=0.3, linewidth=0.6)

    # Legend
    grow_patch = mpatches.Patch(color='#4dac26', label='Grow event')
    best_dot   = plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='#d73027', markersize=7,
                             label=f'Best: {BEST_SR:.0f}% (ep {BEST_EP})')
    ax.legend(handles=[
        plt.Line2D([0], [0], color='#2166ac', lw=1.8, label='Success rate'),
        grow_patch,
        best_dot,
    ], fontsize=8, loc='lower left')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='gpf/fig_training_curve.pdf',
                        help='Output path (.pdf or .png)')
    args = parser.parse_args()
    plot(args.out)


if __name__ == '__main__':
    main()
