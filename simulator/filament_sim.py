import numpy as np
from .wind import make_wind
from .sensors import BilateralSensor

# Discrete action index → (heading_delta_deg, description)
_ACTION_DELTAS = {
    0: 0.0,    # forward
    1: 15.0,   # turn_left_15
    2: -15.0,  # turn_right_15
    3: 180.0,  # turn_around
    4: None,   # cast_left  — filled from config at runtime
    5: None,   # cast_right — filled from config at runtime
}


class FilamentSimulator:
    """
    Farrell-Murlis style filament-based plume simulator.

    Each episode:
      - Source emits discrete filaments at `source_strength` filaments/s (Poisson).
      - Filaments advect with wind and grow by diffusion (σ² = σ₀² + 2Dt).
      - Bilateral sensor samples Gaussian kernel sum at antenna positions.
      - Agent executes discrete actions that change heading then step forward.
    """

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self._wind = make_wind(cfg['wind'], rng)
        self._sensor = BilateralSensor(cfg['sensors'], rng)
        self._filament_mass = cfg['filament'].get('mass', 0.1)
        self._dt = cfg['dt']
        self._max_steps = cfg['episode_length']
        self._domain = np.array(cfg['domain_size'], dtype=float)
        self._cast_amp = float(cfg['agent']['cast_amplitude'])

        # Fill runtime cast deltas
        _ACTION_DELTAS[4] = self._cast_amp
        _ACTION_DELTAS[5] = -self._cast_amp

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        self._wind.reset()
        self._wind_speed = self._wind.speed
        self._wind_dir = self._wind.direction

        self._filaments = []
        self._step = 0

        self._place_source()
        self._place_agent()

        return self._get_obs()

    def step(self, action):
        # Wind update first so the agent sees fresh conditions
        self._wind_speed, self._wind_dir = self._wind.step(self._dt)

        # Emit new filaments, then advect existing ones
        self._emit_filaments()
        self._update_filaments()

        # Move agent
        self._apply_action(int(action))
        self._step += 1

        # Reflect position off domain walls instead of terminating.
        # The agent has no boundary signal in its observation, so terminating
        # on out-of-bounds means it can never learn to avoid walls.  Reflecting
        # boundaries allow longer episodes, more gradient signal, and make the
        # reactive baseline non-zero (so it can serve as a meaningful comparison).
        self.agent_pos = np.clip(self.agent_pos, 0.0, self._domain)

        obs = self._get_obs()
        dist = float(np.linalg.norm(self.agent_pos - self.source_pos))
        success_r = self.cfg.get('success_radius', 0.5)
        success = dist < success_r
        done = success or self._step >= self._max_steps

        info = {
            'success': success,
            'dist_to_source': dist,
            'step': self._step,
            'in_bounds': True,   # always in bounds with reflecting walls
        }
        return obs, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_source(self):
        src = self.cfg['source_position']
        if src == 'random':
            # Place source on the upwind side of the domain so filaments can
            # travel downwind through most of the domain before leaving.
            # upwind = direction wind comes FROM = opposite of wind vector.
            upwind = np.array([-np.cos(self._wind_dir), -np.sin(self._wind_dir)])
            cx, cy = self._domain / 2.0
            dx, dy = self._domain
            # Push center ~25% of domain size toward the upwind edge.
            base = np.array([cx, cy]) + upwind * 0.25 * self._domain
            jitter = self.rng.uniform(-0.15 * self._domain, 0.15 * self._domain)
            pos = base + jitter
            self.source_pos = np.clip(pos, 0.05 * self._domain, 0.95 * self._domain)
        else:
            self.source_pos = np.array(src, dtype=float)

    def _place_agent(self):
        # Agent starts DOWNWIND of the source (where filaments travel to).
        # wind_dir is the direction the wind blows TOWARD, so downwind = wind_dir.
        downwind_dir = self._wind_dir
        dist_range = (3.0, min(10.0, float(np.min(self._domain)) / 3.0))
        dist = self.rng.uniform(*dist_range)
        angle = downwind_dir + self.rng.uniform(-np.pi / 4, np.pi / 4)
        self.agent_pos = self.source_pos + dist * np.array([np.cos(angle), np.sin(angle)])
        self.agent_pos = np.clip(self.agent_pos, 0.0, self._domain)

        # Face upwind (toward source), with noise.
        # upwind = wind_dir + π
        upwind_dir = self._wind_dir + np.pi
        heading_noise = self.rng.normal(0.0, np.deg2rad(30.0))
        self.agent_heading = upwind_dir + heading_noise

    def _emit_filaments(self):
        expected = self.cfg['source_strength'] * self._dt
        n_new = self.rng.poisson(expected)
        for _ in range(n_new):
            jitter = self.rng.normal(0.0, self.cfg['filament']['initial_radius'], size=2)
            self._filaments.append({
                'x': float(self.source_pos[0] + jitter[0]),
                'y': float(self.source_pos[1] + jitter[1]),
                'age': 0.0,
                'sigma': self.cfg['filament']['initial_radius'],
            })

    def _update_filaments(self):
        fil_cfg = self.cfg['filament']
        dt = self._dt
        vx = self._wind_speed * np.cos(self._wind_dir)
        vy = self._wind_speed * np.sin(self._wind_dir)
        sigma0 = fil_cfg['initial_radius']
        D = fil_cfg['diffusion_rate']
        decay_time = fil_cfg['decay_time']
        dx, dy = self._domain

        keep = []
        for f in self._filaments:
            f['age'] += dt
            if f['age'] > decay_time:
                continue
            f['x'] += vx * dt
            f['y'] += vy * dt
            f['sigma'] = float(np.sqrt(sigma0 ** 2 + 2.0 * D * f['age']))
            if 0.0 <= f['x'] <= dx and 0.0 <= f['y'] <= dy:
                keep.append(f)
        self._filaments = keep

    def _apply_action(self, action):
        delta_deg = _ACTION_DELTAS[action]
        self.agent_heading += np.deg2rad(delta_deg)
        # Keep heading in (-π, π]
        self.agent_heading = (self.agent_heading + np.pi) % (2 * np.pi) - np.pi

        speed = self.cfg['agent']['forward_speed']
        self.agent_pos = self.agent_pos + speed * self._dt * np.array([
            np.cos(self.agent_heading),
            np.sin(self.agent_heading),
        ])

    def _get_obs(self):
        left_c, right_c = self._sensor.sample(
            self.agent_pos, self.agent_heading, self._filaments, self._filament_mass
        )
        return {
            'left_concentration': left_c,
            'right_concentration': right_c,
            'wind_speed': self._wind_speed,
            'wind_direction': self._wind_dir,      # absolute, radians
            'agent_heading': self.agent_heading,   # absolute, radians
            'agent_pos': self.agent_pos.copy(),
            'source_pos': self.source_pos.copy(),
        }
