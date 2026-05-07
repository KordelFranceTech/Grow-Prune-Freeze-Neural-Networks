import numpy as np


class BilateralSensor:
    """Bilateral antenna model: two point sensors at fixed offsets in agent-body frame."""

    def __init__(self, cfg, rng):
        sep = cfg['antenna_separation']
        fwd = cfg['antenna_forward_offset']
        # Body-frame offsets: x=forward, y=left
        self._left_local = np.array([fwd, sep / 2.0])
        self._right_local = np.array([fwd, -sep / 2.0])
        self._noise_sigma = cfg['noise_floor_sigma']
        self._saturation = cfg['saturation']
        self.rng = rng

    def antenna_positions(self, agent_pos, agent_heading):
        """World positions of left and right antennae."""
        c, s = np.cos(agent_heading), np.sin(agent_heading)
        R = np.array([[c, -s], [s, c]])
        return agent_pos + R @ self._left_local, agent_pos + R @ self._right_local

    def sample(self, agent_pos, agent_heading, filaments, filament_mass):
        """Return (left_c, right_c) with sensor noise and saturation clamping."""
        left_pos, right_pos = self.antenna_positions(agent_pos, agent_heading)
        left_c = self._concentration_at(left_pos, filaments, filament_mass)
        right_c = self._concentration_at(right_pos, filaments, filament_mass)
        left_c += self.rng.normal(0, self._noise_sigma)
        right_c += self.rng.normal(0, self._noise_sigma)
        left_c = float(np.clip(left_c, 0.0, self._saturation))
        right_c = float(np.clip(right_c, 0.0, self._saturation))
        return left_c, right_c

    def _concentration_at(self, pos, filaments, filament_mass):
        if not filaments:
            return 0.0
        xs = np.array([f['x'] for f in filaments])
        ys = np.array([f['y'] for f in filaments])
        sigma2 = np.array([f['sigma'] ** 2 for f in filaments])

        dx = pos[0] - xs
        dy = pos[1] - ys
        dist2 = dx * dx + dy * dy

        within = dist2 < 9.0 * sigma2
        if not np.any(within):
            return 0.0

        contrib = filament_mass * np.exp(-dist2[within] / (2.0 * sigma2[within])) \
                  / (2.0 * np.pi * sigma2[within])
        return float(np.sum(contrib))
