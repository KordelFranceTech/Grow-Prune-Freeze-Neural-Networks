"""
Generate a publication-quality training curve comparing:
  - GPF Expected SARSA (exp11b / exp_best): adaptive depth, grow + prune events
  - Baseline Expected SARSA: static 22→64→6 network, same hyperparameters, no GPF

Usage:
    python3 gpf/plot_training_curve.py                          # → gpf/fig_training_curve.pdf
    python3 gpf/plot_training_curve.py --out gpf/fig_training_curve.png

When checkpoints/gpf/baseline_eval_exp_best.json exists (produced by
research/train_baseline_comparison.py), the baseline curve is overlaid
automatically.  Without it the GPF-only plot is generated.

Output is suitable for direct LaTeX inclusion:
    \\includegraphics{gpf/fig_training_curve}
"""

import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# GPF exp11b / exp_best training data (from research/experiments_log.md)
# ---------------------------------------------------------------------------

GPF_EPISODES     = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
GPF_SUCCESS      = [24.5, 95.5, 40.5, 97.5, 91.5, 98.0, 95.5, 96.5, 95.5, 68.0]

GROW_EVENTS      = [2000, 3000, 4000]   # episodes where grow fired
PRUNE_EVENTS     = [                    # (episode, keep_pct)
    (2000, 47.9),
    (3000, 19.6),
    (4000,  9.9),
]
LAYER_COUNTS     = {                    # hidden layers active after each eval
    500: 1, 1000: 1, 1500: 1,
    2000: 2, 2500: 2,
    3000: 3, 3500: 3,
    4000: 4, 4500: 4, 5000: 4,
}
GPF_BEST_EP      = 3000
GPF_BEST_SR      = 98.0

# Path written by research/train_baseline_comparison.py
BASELINE_JSON    = 'checkpoints/gpf/baseline_eval_exp_best.json'

# Colours
COL_GPF          = '#2166ac'   # blue
COL_GPF_BEST     = '#d73027'   # red dot
COL_GROW         = '#4dac26'   # green dashed
COL_PRUNE        = '#7b3294'   # purple annotation
COL_BASELINE     = '#b35900'   # orange-brown


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(out_path: str, max_ep: int = 5000):
    baseline = _load_baseline()

    # Truncate GPF data
    gpf_eps  = [e for e in GPF_EPISODES  if e <= max_ep]
    gpf_sr   = [s for e, s in zip(GPF_EPISODES, GPF_SUCCESS) if e <= max_ep]
    grows    = [e for e in GROW_EVENTS  if e <= max_ep]
    prunes   = [(e, k) for e, k in PRUNE_EVENTS if e <= max_ep]
    layer_eps = [e for e in gpf_eps if LAYER_COUNTS.get(e) != LAYER_COUNTS.get(
                 gpf_eps[gpf_eps.index(e) - 1] if gpf_eps.index(e) > 0 else None)]

    has_baseline = baseline is not None
    fig_w = 9 if has_baseline else 7
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    # ---- GPF curve ---------------------------------------------------------
    ax.plot(gpf_eps, gpf_sr,
            color=COL_GPF, linewidth=1.8, marker='o', markersize=5,
            zorder=3, label='GPF Expected SARSA (adaptive depth)')

    # best checkpoint marker (only if within range)
    if GPF_BEST_EP <= max_ep:
        ax.scatter([GPF_BEST_EP], [GPF_BEST_SR],
                   color=COL_GPF_BEST, s=90, zorder=5,
                   label=f'GPF best: {GPF_BEST_SR:.0f}% (ep {GPF_BEST_EP})')

    # grow event verticals
    for i, ep in enumerate(grows):
        ax.axvline(ep, color=COL_GROW, linewidth=1.1, linestyle='--', alpha=0.75,
                   label='Grow event' if i == 0 else None)

    # prune annotations
    for ep, keep in prunes:
        idx = GPF_EPISODES.index(ep)
        ax.annotate(f'prune\n{keep:.0f}% kept',
                    xy=(ep, GPF_SUCCESS[idx]),
                    xytext=(ep + 130, GPF_SUCCESS[idx] - 13),
                    fontsize=6.5, color=COL_PRUNE,
                    arrowprops=dict(arrowstyle='->', color=COL_PRUNE, lw=0.8),
                    ha='left')

    # layer-depth strip along bottom
    prev = None
    for ep in gpf_eps:
        n = LAYER_COUNTS[ep]
        if n != prev:
            ax.text(ep, -8, f'L={n}', fontsize=7, ha='center', color='#555555')
        prev = n

    # ---- Baseline curve (if available) -------------------------------------
    if has_baseline:
        bl_eps  = [e for e in baseline['eval_episodes'] if e <= max_ep]
        bl_sr   = [s for e, s in zip(baseline['eval_episodes'],
                                     baseline['success_rates']) if e <= max_ep]
        bl_best = max(bl_sr) if bl_sr else baseline['best_sr']

        ax.plot(bl_eps, bl_sr,
                color=COL_BASELINE, linewidth=1.8, marker='s', markersize=5,
                linestyle='--', zorder=3,
                label=f'Baseline Expected SARSA (static 22→64→6)')

        bl_best_ep = bl_eps[bl_sr.index(max(bl_sr))]
        ax.scatter([bl_best_ep], [bl_best],
                   color=COL_BASELINE, s=90, marker='D', zorder=5,
                   label=f'Baseline best: {bl_best:.0f}% (ep {bl_best_ep})')

    # ---- Axes & legend -----------------------------------------------------
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success rate (%)', fontsize=11)
    title = ('GPF vs. Baseline Expected SARSA — Plume Navigation'
             if has_baseline else
             'GPF Expected SARSA — Plume Navigation Training Curve')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, max_ep + 200)
    ax.set_ylim(-12, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, alpha=0.3, linewidth=0.6)

    # build legend manually so ordering is clean
    legend_handles = [
        plt.Line2D([0], [0], color=COL_GPF, lw=1.8, marker='o', markersize=5,
                   label='GPF Expected SARSA (adaptive depth)'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COL_GPF_BEST, markersize=8,
                   label=f'GPF best: {GPF_BEST_SR:.0f}% (ep {GPF_BEST_EP})'),
        mpatches.Patch(facecolor=COL_GROW, alpha=0.75, label='Grow event'),
    ]
    if has_baseline:
        legend_handles += [
            plt.Line2D([0], [0], color=COL_BASELINE, lw=1.8, marker='s',
                       markersize=5, linestyle='--',
                       label='Baseline Expected SARSA (static 22→64→6)'),
            plt.Line2D([0], [0], marker='D', color='w',
                       markerfacecolor=COL_BASELINE, markersize=8,
                       label=f'Baseline best: {baseline["best_sr"]:.0f}%'),
        ]

    ax.legend(handles=legend_handles, fontsize=8,
              loc='lower left', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}  (baseline overlay: {'yes' if has_baseline else 'no — run research/train_baseline_comparison.py first'})")
    plt.close(fig)


def _load_baseline():
    """Return baseline eval dict if JSON exists, else None."""
    if not os.path.exists(BASELINE_JSON):
        return None
    try:
        with open(BASELINE_JSON) as f:
            data = json.load(f)
        # Sanity check: needs at least one eval point
        if not data.get('eval_episodes'):
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='gpf/fig_training_curve.pdf',
                        help='Output path (.pdf or .png)')
    parser.add_argument('--max-ep', type=int, default=4500,
                        help='Truncate both curves at this episode (default: 4500)')
    args = parser.parse_args()
    plot(args.out, max_ep=args.max_ep)


if __name__ == '__main__':
    main()
