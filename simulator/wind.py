import numpy as np


class ConstantWind:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def reset(self):
        self._speed = self.cfg['mean_speed']
        self._direction = np.deg2rad(self.cfg['mean_direction'])

    def step(self, dt):
        return self._speed, self._direction

    @property
    def speed(self):
        return self._speed

    @property
    def direction(self):
        return self._direction


class GaussianGustWind:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def reset(self):
        self._speed = self.cfg['mean_speed']
        self._direction = np.deg2rad(self.cfg['mean_direction'])

    def step(self, dt):
        cfg = self.cfg
        self._speed += self.rng.normal(0, cfg['speed_volatility'] * np.sqrt(dt))
        self._speed = max(0.1, self._speed)
        self._direction += self.rng.normal(0, np.deg2rad(cfg['direction_volatility']) * np.sqrt(dt))
        return self._speed, self._direction

    @property
    def speed(self):
        return self._speed

    @property
    def direction(self):
        return self._direction


class OrnsteinUhlenbeckWind:
    """Mean-reverting wind with separate OU processes for speed and direction."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def reset(self):
        self._speed = self.cfg['mean_speed']
        self._direction = np.deg2rad(self.cfg['mean_direction'])

    def step(self, dt):
        cfg = self.cfg
        tau = cfg['correlation_time']
        alpha = dt / tau

        mean_speed = cfg['mean_speed']
        self._speed += -alpha * (self._speed - mean_speed) + \
                       cfg['speed_volatility'] * np.sqrt(2 * alpha) * self.rng.normal()
        self._speed = max(0.1, self._speed)

        mean_dir = np.deg2rad(cfg['mean_direction'])
        dir_vol = np.deg2rad(cfg['direction_volatility'])
        self._direction += -alpha * (self._direction - mean_dir) + \
                           dir_vol * np.sqrt(2 * alpha) * self.rng.normal()

        return self._speed, self._direction

    @property
    def speed(self):
        return self._speed

    @property
    def direction(self):
        return self._direction


def make_wind(cfg, rng):
    wtype = cfg['type']
    if wtype == 'constant':
        return ConstantWind(cfg, rng)
    elif wtype == 'gaussian_gust':
        return GaussianGustWind(cfg, rng)
    elif wtype == 'ornstein_uhlenbeck':
        return OrnsteinUhlenbeckWind(cfg, rng)
    else:
        raise ValueError(f"Unknown wind type: {wtype}")
