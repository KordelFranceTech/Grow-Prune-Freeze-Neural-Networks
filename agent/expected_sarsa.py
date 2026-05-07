"""
Expected SARSA with a one-hidden-layer PyTorch Q-network.

Architecture:
    one-hot(state) [22-dim] → Linear → ReLU → Linear → Q-values [6-dim]

Training uses the semi-gradient TD update: the target y is computed inside
torch.no_grad() so gradients flow only through Q(s, a), not through Q(s', ·).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _QNetwork(nn.Module):
    """One-hidden-layer feedforward Q-network."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        # PyTorch default (Kaiming uniform) is appropriate for ReLU — no override needed.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def _encode(state: tuple, state_shape: tuple) -> torch.Tensor:
    """Concatenate one-hot vectors for each discrete state dimension."""
    x = np.zeros(sum(state_shape), dtype=np.float32)
    offset = 0
    for val, size in zip(state, state_shape):
        x[offset + val] = 1.0
        offset += size
    return torch.from_numpy(x)


class ExpectedSARSA:
    """
    Expected SARSA with PyTorch neural network function approximation.

    Public interface is identical to the previous tabular version so that
    train.py, evaluate.py, and visualize.py require no changes.
    """

    def __init__(self, state_shape: tuple, n_actions: int,
                 hidden_dim: int, lr: float,
                 gamma: float, epsilon_start: float,
                 epsilon_end: float, epsilon_decay: float,
                 seed: int = 0):
        torch.manual_seed(seed)
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        input_dim = sum(state_shape)
        self.network = _QNetwork(input_dim, hidden_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Forward (inference)
    # ------------------------------------------------------------------

    def q_values(self, state: tuple) -> np.ndarray:
        x = _encode(state, self.state_shape)
        with torch.no_grad():
            return self.network(x).numpy()

    # ------------------------------------------------------------------
    # Action selection (ε-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state: tuple, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    # ------------------------------------------------------------------
    # Semi-gradient Expected SARSA update
    # ------------------------------------------------------------------

    def update(self, state: tuple, action: int,
               reward: float, next_state: tuple, done: bool):
        # Compute TD target — no gradient through this path
        with torch.no_grad():
            if done:
                target = float(reward)
            else:
                next_q = self.network(_encode(next_state, self.state_shape)).numpy()
                best = int(np.argmax(next_q))
                pi = np.full(self.n_actions, self.epsilon / self.n_actions)
                pi[best] += 1.0 - self.epsilon
                expected_q = float(np.dot(pi, next_q))
                target = float(reward) + self.gamma * expected_q

        # Forward pass for the taken action (gradient flows here)
        q_all = self.network(_encode(state, self.state_shape))
        q_sa = q_all[action]
        loss = F.mse_loss(q_sa, torch.tensor(target, dtype=torch.float32))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # ε decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        p = str(path)
        if not p.endswith('.pt'):
            p += '.pt'
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, p)

    def load(self, path: str):
        p = str(path)
        if not p.endswith('.pt'):
            p += '.pt'
        ckpt = torch.load(p, map_location='cpu')
        self.network.load_state_dict(ckpt['network_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epsilon = float(ckpt.get('epsilon', self.epsilon))
