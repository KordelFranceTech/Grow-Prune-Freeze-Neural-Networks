import numpy as np
from .vocabulary import N_CONC_BINS, N_WIND_OCTANTS


class Tokenizer:
    """
    Maps raw sensor observations to discrete token indices.

    State tuple: (left_bin, right_bin, wind_octant)
      left_bin, right_bin : 0..6  (0=BLANK, 1=C0, ..., 6=C5)
      wind_octant         : 0..7  (0=headwind, increasing clockwise)
    """

    def __init__(self, tok_cfg, sensor_cfg):
        sigma = sensor_cfg['noise_floor_sigma']
        # Absolute bin edges from noise-floor multipliers
        self._edges = np.array(tok_cfg['concentration_bin_edges']) * sigma
        self._n_octants = tok_cfg['wind_n_octants']
        self._wind_zero_thresh = tok_cfg['wind_zero_threshold']

    @property
    def state_shape(self):
        return (N_CONC_BINS, N_CONC_BINS, N_WIND_OCTANTS)

    def tokenize(self, obs):
        """Return (left_bin, right_bin, wind_octant) as a tuple of ints."""
        left_bin = self._conc_bin(obs['left_concentration'])
        right_bin = self._conc_bin(obs['right_concentration'])
        wind_octant = self._wind_octant(obs['wind_direction'], obs['agent_heading'],
                                        obs['wind_speed'])
        return (left_bin, right_bin, wind_octant)

    def _conc_bin(self, c):
        """Concentration value → bin index 0..6."""
        bin_idx = int(np.searchsorted(self._edges, c, side='right'))
        return min(bin_idx, N_CONC_BINS - 1)

    def _wind_octant(self, wind_dir, agent_heading, wind_speed):
        """
        Wind direction relative to agent heading → octant index 0..7.

        Octant 0 = wind coming from directly ahead (headwind).
        Octants increase clockwise (0=front, 2=right, 4=rear, 6=left).
        """
        if wind_speed < self._wind_zero_thresh:
            return 0  # treat calm as headwind per config default

        # Wind direction tells us where the wind *goes*; reverse to get where it *comes from*
        wind_from = wind_dir + np.pi
        rel = wind_from - agent_heading
        # Normalize to [0, 2π)
        rel = rel % (2 * np.pi)
        octant = int(rel / (2 * np.pi) * self._n_octants) % self._n_octants
        return octant

    def bin_centers(self):
        """Return concentration bin center values for each bin index."""
        centers = []
        edges = [0.0] + list(self._edges)
        for i in range(N_CONC_BINS):
            lo = edges[i]
            hi = edges[i + 1] if i + 1 < len(edges) else edges[-1] * 4.0
            centers.append((lo + hi) / 2.0)
        return centers
