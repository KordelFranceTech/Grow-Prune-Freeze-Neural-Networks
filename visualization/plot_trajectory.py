"""
2D visualization of the agent navigating the plume.

record_episode()  — runs one greedy episode and captures trajectory + filament snapshots
plot_trajectory() — renders the captured data as a matplotlib figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Rectangle


def record_episode(agent, env, seed=0):
    """
    Run one greedy episode (ε=0) and return a dict of recorded data.

    The filament state is snapshotted once every 50 steps so the plume
    shape can be visualised without storing every frame.
    """
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rng = np.random.default_rng(seed)

    state = env.reset(seed=seed)

    positions = []
    concentrations = []
    headings = []
    filament_snapshots = []   # list of (step, [(x, y, sigma, age), ...])

    done = False
    step = 0
    while not done:
        obs = env._sim._get_obs()
        positions.append(obs['agent_pos'].copy())
        concentrations.append(max(obs['left_concentration'], obs['right_concentration']))
        headings.append(obs['agent_heading'])

        if step % 50 == 0:
            filament_snapshots.append((
                step,
                [(f['x'], f['y'], f['sigma'], f['age']) for f in env._sim._filaments],
                env._sim._wind_dir,
                env._sim._wind_speed,
            ))

        action = agent.select_action(state, rng)
        state, _, done, info = env.step(action)
        step += 1

    # Capture final position
    obs = env._sim._get_obs()
    positions.append(obs['agent_pos'].copy())
    concentrations.append(max(obs['left_concentration'], obs['right_concentration']))

    agent.epsilon = saved_epsilon

    return {
        'positions': np.array(positions),
        'concentrations': np.array(concentrations),
        'headings': np.array(headings),
        'filament_snapshots': filament_snapshots,
        'source_pos': env._sim.source_pos.copy(),
        'domain': np.array(env._sim.cfg['domain_size']),
        'info': info,
        'n_steps': step,
    }


def _plume_heatmap(filaments, domain, filament_mass, resolution=120):
    """Compute log-concentration on a 2-D grid from a filament list."""
    xs = np.linspace(0, domain[0], resolution)
    ys = np.linspace(0, domain[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    C = np.zeros_like(X)
    for (fx, fy, sigma, age) in filaments:
        dx = X - fx
        dy = Y - fy
        s2 = sigma ** 2
        dist2 = dx * dx + dy * dy
        mask = dist2 < 9.0 * s2
        C[mask] += filament_mass * np.exp(-dist2[mask] / (2.0 * s2)) / (2.0 * np.pi * s2)
    return X, Y, C


def plot_trajectory(data, filament_mass=0.1, save_path=None, show=True):
    """
    Render a 2-D top-down plot of the recorded episode.

    Parameters
    ----------
    data        : dict returned by record_episode()
    save_path   : if given, save the figure to this path instead of showing
    show        : call plt.show() when True
    """
    positions = data['positions']
    concs = data['concentrations']
    source = data['source_pos']
    domain = data['domain']
    info = data['info']
    snapshots = data['filament_snapshots']
    n_steps = data['n_steps']

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_facecolor('#f5f5f5')

    # ------------------------------------------------------------------
    # Plume heatmap — use the snapshot nearest the middle of the episode
    # ------------------------------------------------------------------
    if snapshots:
        mid = len(snapshots) // 2
        _, filaments_mid, wind_dir, wind_speed = snapshots[mid]
        if filaments_mid:
            X, Y, C = _plume_heatmap(filaments_mid, domain, filament_mass)
            C_log = np.log10(np.clip(C, 1e-6, None))
            ax.contourf(X, Y, C_log, levels=20, cmap='YlOrRd', alpha=0.55)
    else:
        wind_dir, wind_speed = 0.0, 1.0

    # ------------------------------------------------------------------
    # Domain boundary
    # ------------------------------------------------------------------
    ax.add_patch(Rectangle((0, 0), domain[0], domain[1],
                            fill=False, edgecolor='#333333', linewidth=2, zorder=3))

    # ------------------------------------------------------------------
    # Agent trajectory — line coloured by detected concentration
    # ------------------------------------------------------------------
    noise_floor = 0.001
    c_min = max(noise_floor, float(concs.min()))
    c_max = max(c_min * 10, float(concs.max()))
    norm = mcolors.LogNorm(vmin=c_min, vmax=c_max)
    traj_cmap = plt.cm.Blues_r

    for i in range(len(positions) - 1):
        col = traj_cmap(norm(max(concs[i], c_min)))
        ax.plot([positions[i, 0], positions[i + 1, 0]],
                [positions[i, 1], positions[i + 1, 1]],
                color=col, linewidth=1.8, alpha=0.85, zorder=4)

    # Direction arrows along path (every 30 steps)
    for i in range(0, len(data['headings']) - 1, 30):
        hdg = data['headings'][i]
        dx = np.cos(hdg) * 0.25
        dy = np.sin(hdg) * 0.25
        ax.annotate('', xy=positions[i] + np.array([dx, dy]), xytext=positions[i],
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0),
                    zorder=5)

    # Start and end markers
    ax.scatter(*positions[0], s=120, color='limegreen', edgecolors='k',
               linewidths=0.8, zorder=6, label='Start')
    if info['success']:
        ax.scatter(*positions[-1], s=250, color='gold', edgecolors='k',
                   linewidths=0.8, marker='*', zorder=6, label='Source reached')
    else:
        ax.scatter(*positions[-1], s=120, color='crimson', edgecolors='k',
                   linewidths=0.8, marker='X', zorder=6, label='Timed out')

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------
    ax.scatter(*source, s=350, color='crimson', edgecolors='k',
               linewidths=1.2, marker='*', zorder=7, label='Odour source')

    # ------------------------------------------------------------------
    # Wind direction indicator (bottom-left corner)
    # ------------------------------------------------------------------
    arrow_origin = np.array([domain[0] * 0.06, domain[1] * 0.06])
    arrow_len = domain[0] * 0.08
    adx = np.cos(wind_dir) * arrow_len
    ady = np.sin(wind_dir) * arrow_len
    ax.annotate('', xy=arrow_origin + np.array([adx, ady]), xytext=arrow_origin,
                arrowprops=dict(arrowstyle='->', color='dimgray', lw=2.0), zorder=6)
    ax.text(arrow_origin[0], arrow_origin[1] - domain[1] * 0.04,
            'wind', ha='center', va='top', color='dimgray', fontsize=9)

    # ------------------------------------------------------------------
    # Colourbar for trajectory concentration
    # ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=traj_cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Detected concentration (max antenna)', fontsize=10)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    status = 'Source found' if info['success'] else 'Timeout'
    ax.set_title(
        f'Plume Navigation — {status} in {n_steps} steps  '
        f'(dist to source: {info["dist_to_source"]:.2f} m)',
        fontsize=12,
    )
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_xlim(-0.5, domain[0] + 0.5)
    ax.set_ylim(-0.5, domain[1] + 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Trajectory plot saved → {save_path}')

    if show:
        import matplotlib
        if matplotlib.get_backend().lower() == 'agg':
            # Non-interactive backend: fall back to saving if a path was not given
            if not save_path:
                save_path = 'trajectory.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f'Non-interactive backend — saved to {save_path} instead of displaying.')
        else:
            plt.show()

    return fig, ax
