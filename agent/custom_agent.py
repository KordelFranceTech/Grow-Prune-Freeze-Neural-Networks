"""
Expected SARSA with a one-hidden-layer neural network Q-function.
Pure NumPy implementation — no PyTorch dependency.

Architecture:
    one-hot(state) [22-dim] → Linear(22→64) → ReLU → Linear(64→6) → Q-values

Training uses the semi-gradient TD update: the target y is computed with
the current weights but treated as a fixed constant during backprop, so
gradients flow only through Q(s, a), not through Q(s', ·).

Weights saved as .npz (NumPy archive).
"""

import numpy as np


def _one_hot(state, state_shape):
    """Concatenate one-hot vectors for each dimension of the discrete state."""
    x = np.zeros(sum(state_shape), dtype=np.float64)
    offset = 0
    for val, size in zip(state, state_shape):
        x[offset + val] = 1.0
        offset += size
    return x


class ExpectedSARSA:
    """
    Expected SARSA with neural network function approximation.

    State encoding: concatenated one-hot over (left_bin, right_bin, wind_octant).
    Update: semi-gradient MSE TD loss on Q(s, a) toward target y.
    Optimiser: SGD with global gradient-norm clipping to 1.0.
    """

    def __init__(self, state_shape, n_actions, hidden_dim, lr,
                 gamma, epsilon_start, epsilon_end, epsilon_decay, seed=0):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.lr = lr

        input_dim = sum(state_shape)   # 7+7+8 = 22 for default vocab
        rng = np.random.default_rng(seed)

        # He initialisation for ReLU layers
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (n_actions, hidden_dim))
        self.b2 = np.zeros(n_actions)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, x):
        h_pre = self.W1 @ x + self.b1
        h = np.maximum(0.0, h_pre)          # ReLU
        q = self.W2 @ h + self.b2
        return q, h, h_pre

    def q_values(self, state):
        x = _one_hot(state, self.state_shape)
        q, _, _ = self._forward(x)
        return q

    # ------------------------------------------------------------------
    # Action selection (ε-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    # ------------------------------------------------------------------
    # Semi-gradient TD update
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, done):
        x = _one_hot(state, self.state_shape)
        q, h, h_pre = self._forward(x)

        # Compute target (no gradient through this path)
        if done:
            target = float(reward)
        else:
            next_q = self.q_values(next_state)
            best = int(np.argmax(next_q))
            pi = np.full(self.n_actions, self.epsilon / self.n_actions)
            pi[best] += 1.0 - self.epsilon
            expected_q = float(np.dot(pi, next_q))
            target = float(reward) + self.gamma * expected_q

        # TD error for the taken action only
        td_error = q[action] - target

        # Backprop through output layer (only the neuron for the taken action)
        dW2 = np.zeros_like(self.W2)
        dW2[action] = td_error * h
        db2 = np.zeros_like(self.b2)
        db2[action] = td_error

        # Backprop through ReLU and input layer
        dh = td_error * self.W2[action]
        dh_pre = dh * (h_pre > 0.0)   # ReLU subgradient
        dW1 = np.outer(dh_pre, x)
        db1 = dh_pre

        # Global gradient-norm clip then SGD step
        gnorm = np.sqrt(
            np.sum(dW1 ** 2) + np.sum(db1 ** 2) +
            np.sum(dW2 ** 2) + np.sum(db2 ** 2)
        )
        scale = 1.0 / gnorm if gnorm > 1.0 else 1.0

        self.W1 -= self.lr * scale * dW1
        self.b1 -= self.lr * scale * db1
        self.W2 -= self.lr * scale * dW2
        self.b2 -= self.lr * scale * db2

    # ------------------------------------------------------------------
    # ε decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence (.npz format)
    # ------------------------------------------------------------------

    def save(self, path):
        p = str(path)
        if p.endswith('.npz'):
            p = p[:-4]          # np.savez adds .npz automatically
        np.savez(p, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        p = str(path)
        if not p.endswith('.npz'):
            p += '.npz'
        data = np.load(p)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
