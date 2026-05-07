"""Evaluation utilities for the Expected SARSA plume navigation agent."""

import numpy as np
from agent.env import PlumeNavigationEnv
from simulator.policies import Reactive


def evaluate_agent(agent, env, n_episodes, greedy=True):
    """
    Run `agent` for `n_episodes` episodes and return summary statistics.

    When `greedy=True` the agent acts greedily (ε=0); set False to use
    its current epsilon for exploration-inclusive evaluation.

    Returns a dict with:
      success_rate      : fraction of episodes where source was found
      mean_steps        : mean steps taken (across all episodes)
      mean_steps_success: mean steps taken (success episodes only)
      n_episodes        : total episodes run
    """
    saved_epsilon = agent.epsilon
    if greedy:
        agent.epsilon = 0.0

    rng = np.random.default_rng(0)
    successes = []
    steps_all = []
    steps_success = []

    for ep in range(n_episodes):
        state = env.reset(seed=ep)
        done = False
        ep_steps = 0
        while not done:
            action = agent.select_action(state, rng)
            state, _, done, info = env.step(action)
            ep_steps += 1
        successes.append(info['success'])
        steps_all.append(ep_steps)
        if info['success']:
            steps_success.append(ep_steps)

    agent.epsilon = saved_epsilon

    return {
        'success_rate': float(np.mean(successes)),
        'mean_steps': float(np.mean(steps_all)),
        'mean_steps_success': float(np.mean(steps_success)) if steps_success else float('nan'),
        'n_episodes': n_episodes,
    }


def evaluate_reactive_baseline(sim_cfg, tok_cfg, rl_cfg, n_episodes):
    """
    Run the hand-coded Reactive policy and return the same summary dict.
    Used to set the bar that the RL agent must beat.
    """
    env = PlumeNavigationEnv(sim_cfg, tok_cfg, rl_cfg)
    rng = np.random.default_rng(1)
    policy = Reactive(sim_cfg, rng)

    successes = []
    steps_all = []
    steps_success = []

    for ep in range(n_episodes):
        env.reset(seed=ep)
        obs = env._sim._get_obs()
        policy.reset()
        done = False
        ep_steps = 0
        while not done:
            action = policy.select_action(obs)
            _, _, done, info = env.step(action)
            obs = env._sim._get_obs()
            ep_steps += 1
        successes.append(info['success'])
        steps_all.append(ep_steps)
        if info['success']:
            steps_success.append(ep_steps)

    return {
        'success_rate': float(np.mean(successes)),
        'mean_steps': float(np.mean(steps_all)),
        'mean_steps_success': float(np.mean(steps_success)) if steps_success else float('nan'),
        'n_episodes': n_episodes,
    }
