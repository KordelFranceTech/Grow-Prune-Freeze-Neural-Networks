import numpy as np
from simulator.filament_sim import FilamentSimulator
from tokenization.tokenizer import Tokenizer


class PlumeNavigationEnv:
    """
    Gym-style environment wrapping the filament simulator.

    Observations are returned as discrete state tuples (left_bin, right_bin, wind_octant)
    suitable for a tabular Q-learning agent.

    Reward shaping:
      +source_found      : reaching the source
      time_penalty       : every step (negative, encourages efficiency)
      +whiff_bonus       : any antenna detects concentration above threshold
      blank_penalty      : each step when both antennae blank for > blank_penalty_duration
      +distance_shaping  : (prev_dist - curr_dist) * shaping_coeff per step
                           Potential-based shaping: keeps optimal policy invariant.
                           Provides dense gradient toward the source even before
                           the agent discovers the terminal success reward.
    """

    def __init__(self, sim_cfg, tok_cfg, rl_cfg, seed=None):
        rng_seed = seed if seed is not None else sim_cfg.get('seed', 42)
        self._rng = np.random.default_rng(rng_seed)
        self._sim = FilamentSimulator(sim_cfg, self._rng)
        self._tok = Tokenizer(tok_cfg, sim_cfg['sensors'])
        self._rl_cfg = rl_cfg
        self._reward_cfg = rl_cfg['reward']
        self._noise_floor = sim_cfg['sensors']['noise_floor_sigma']
        self._whiff_thresh = self._noise_floor * self._reward_cfg['whiff_threshold_multiplier']
        self._blank_penalty_steps = int(
            self._reward_cfg['blank_penalty_duration'] / sim_cfg['dt']
        )
        self._max_steps = rl_cfg.get('max_steps_per_episode', sim_cfg['episode_length'])
        # Override simulator episode length so it matches RL budget
        self._sim.cfg = dict(self._sim.cfg)
        self._sim._max_steps = self._max_steps

        self._distance_shaping = float(self._reward_cfg.get('distance_shaping', 0.0))
        self._blank_steps = 0
        self._prev_dist = None
        self.state_shape = self._tok.state_shape
        self.n_actions = 6

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs = self._sim.reset()
        self._blank_steps = 0
        self._prev_dist = float(np.linalg.norm(
            self._sim.agent_pos - self._sim.source_pos
        ))
        return self._tok.tokenize(obs)

    def step(self, action):
        obs, done, info = self._sim.step(action)
        state = self._tok.tokenize(obs)
        reward = self._compute_reward(obs, info)
        return state, reward, done, info

    def _compute_reward(self, obs, info):
        r = self._reward_cfg['time_penalty']

        if info['success']:
            r += self._reward_cfg['source_found']
            self._prev_dist = 0.0
            return r

        left_c = obs['left_concentration']
        right_c = obs['right_concentration']
        if max(left_c, right_c) > self._whiff_thresh:
            r += self._reward_cfg['whiff_bonus']
            self._blank_steps = 0
        else:
            self._blank_steps += 1
            if self._blank_steps > self._blank_penalty_steps:
                r += self._reward_cfg['blank_penalty']

        # Potential-based distance shaping: reward for getting closer.
        # F(s,s') = γΦ(s') - Φ(s) ≈ Φ(s') - Φ(s) (γ≈1) where Φ = -λ·dist
        # → F = λ·(prev_dist - curr_dist)
        if self._distance_shaping > 0.0 and self._prev_dist is not None:
            curr_dist = info['dist_to_source']
            r += self._distance_shaping * (self._prev_dist - curr_dist)
            self._prev_dist = curr_dist

        return r
