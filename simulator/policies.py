"""Behavior policies for training data generation — not the RL agent."""

import numpy as np
from tokenization.vocabulary import A_FWD, A_LEFT15, A_RIGHT15, A_CAST_L, A_CAST_R


class RandomWalk:
    """Uniform random action selection. Provides broad state-space coverage."""

    def __init__(self, n_actions, rng):
        self.n_actions = n_actions
        self.rng = rng

    def reset(self):
        pass

    def select_action(self, obs):
        return int(self.rng.integers(self.n_actions))


class Reactive:
    """
    Turn upwind on whiff; cast perpendicular to wind on blank.

    This is the standard reactive plume-tracking baseline. It works well in
    steady conditions but struggles with high turbulence.
    """

    def __init__(self, cfg, rng):
        self.noise_floor = cfg['sensors']['noise_floor_sigma']
        self.whiff_threshold = self.noise_floor * 3.0
        self.rng = rng
        self._cast_side = 1  # alternates between +1 (left) and -1 (right)
        self._blank_steps = 0
        self._blank_threshold_steps = int(1.0 / cfg['dt'])  # 1 second blank before casting

    def reset(self):
        self._cast_side = 1
        self._blank_steps = 0

    def select_action(self, obs):
        left_c = obs['left_concentration']
        right_c = obs['right_concentration']
        wind_dir = obs['wind_direction']
        heading = obs['agent_heading']

        if max(left_c, right_c) > self.whiff_threshold:
            self._blank_steps = 0
            # Steer upwind: turn toward the wind direction
            wind_rel = _angle_diff(wind_dir, heading)
            if wind_rel > 0:
                return A_LEFT15
            else:
                return A_RIGHT15
        else:
            self._blank_steps += 1
            if self._blank_steps >= self._blank_threshold_steps:
                # Cast perpendicular to wind
                self._blank_steps = 0
                self._cast_side *= -1
                return A_CAST_L if self._cast_side > 0 else A_CAST_R
            return A_FWD


class SurgeAndCast:
    """
    Surge upwind on whiff; cast in expanding arcs on blank.

    State machine: SURGE → CAST → SURGE → ...
    """

    def __init__(self, cfg, rng):
        self.noise_floor = cfg['sensors']['noise_floor_sigma']
        self.whiff_threshold = self.noise_floor * 3.0
        self.rng = rng
        self._state = 'surge'
        self._cast_side = 1
        self._blank_steps = 0
        self._surge_steps = 0
        self._blank_limit = int(0.5 / cfg['dt'])   # 0.5s blank triggers cast
        self._surge_limit = int(1.0 / cfg['dt'])   # 1s surge then check

    def reset(self):
        self._state = 'surge'
        self._cast_side = 1
        self._blank_steps = 0
        self._surge_steps = 0

    def select_action(self, obs):
        left_c = obs['left_concentration']
        right_c = obs['right_concentration']
        in_plume = max(left_c, right_c) > self.whiff_threshold

        if in_plume:
            self._state = 'surge'
            self._blank_steps = 0
            self._surge_steps += 1
            # Bias toward higher-concentration side
            if left_c > right_c:
                return A_LEFT15
            elif right_c > left_c:
                return A_RIGHT15
            return A_FWD
        else:
            self._blank_steps += 1
            self._surge_steps = 0
            if self._blank_steps >= self._blank_limit:
                self._blank_steps = 0
                self._cast_side *= -1
                return A_CAST_L if self._cast_side > 0 else A_CAST_R
            return A_FWD


def _angle_diff(a, b):
    """Signed difference a - b, wrapped to (-π, π]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi
