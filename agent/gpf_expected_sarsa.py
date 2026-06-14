"""
GPF Expected SARSA — Grow-Prune-Freeze adaptive-depth Q-network.

Implements the dissertation's GPF framework (Chapter 7, Table 7.1):

  GROW  — adds a hidden layer when the EMA of episode-level TD-error
           improvement stagnates below eps_add, subject to a warm-up
           period (min_episodes_before_grow, εe) and per-grow cooldown
           (cooldown_episodes, εk).

  PRUNE — deferred after each grow event; fires once prune_accum_steps
           fresh gradients have been collected.  Uses OBD saliency
           (0.5 · w² · E[g²]) as primary criterion, but belief-hardened
           weights (h_ij ≥ tau_belief_prune, ϑp) are protected regardless
           of saliency — preserving sparse but important connections.

  FREEZE — layers whose weights have been stable (|Δw| < tau_freeze_delta,
           ϑf) across a fraction tau_freeze_frac of their parameters for
           freeze_patience (εf) consecutive stability checks are frozen.
           This follows Eq. 7.4 of the dissertation, not gradient norms.

Belief system (§7.1):
  Each parameter has a belief value h ∈ [0, 1).  At every update step,
  h_ij += φ_h · 1{|w_ij| > tau_belief_harden} where φ_h = 1/n_total_steps.
  Hardened (high-belief) weights are protected from pruning.

Key differences from v1:
  - EMA improvement is NOT reset after grow (v1 bug caused runaway grows)
  - Grow trigger operates on episode-level average TD error (smoother RL signal)
  - Per-weight belief system added (dissertation §7.1, Table 7.1 params ϑv/ϑp/φh)
  - Freeze uses weight-stability snapshots (Eq. 7.4) not gradient norm EMA

References
----------
- LeCun et al. (1989) Optimal Brain Damage
- GPFs_reference.pdf  §7.1, Table 7.1, Eq. 7.4
- train_gpt_gpf.py    AdaptiveGPT reference implementation
"""

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State encoding  (identical to expected_sarsa.py)
# ---------------------------------------------------------------------------

def _encode(state: tuple, state_shape: tuple) -> torch.Tensor:
    x = np.zeros(sum(state_shape), dtype=np.float32)
    offset = 0
    for val, size in zip(state, state_shape):
        x[offset + val] = 1.0
        offset += size
    return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Adaptive Q-network with a growable hidden-layer stack
# ---------------------------------------------------------------------------

class _GPFQNetwork(nn.Module):
    """
    MLP Q-network whose hidden stack grows at runtime.

    Layout
    ------
    hidden_layers[0]   : Linear(input_dim, hidden_dim)   — the seed layer
    hidden_layers[k≥1] : Linear(hidden_dim, hidden_dim)  — grown layers
    output_layer       : Linear(hidden_dim, n_actions)   — never frozen/pruned
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.output_layer = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

    def add_hidden_layer(self) -> nn.Module:
        """
        Append one H→H hidden layer before the output layer.

        Always uses near-identity initialization (I + small noise), preserving
        the learned Q-function via dynamical isometry regardless of grow count.
        Deepcopy of a previously pruned layer is intentionally avoided: a pruned
        layer may have ≤15% nonzero weights, which would cripple the new layer's
        starting capacity.
        """
        H = self.hidden_dim
        new_layer = nn.Linear(H, H)
        with torch.no_grad():
            new_layer.weight.data = torch.eye(H) + 0.01 * torch.randn(H, H)
            new_layer.bias.data = 0.01 * torch.randn(H)
        self.hidden_layers.append(new_layer)
        return new_layer

    @property
    def n_hidden(self) -> int:
        return len(self.hidden_layers)


# ---------------------------------------------------------------------------
# GPF Expected SARSA agent
# ---------------------------------------------------------------------------

class GPFExpectedSARSA:
    """
    Expected SARSA with Grow-Prune-Freeze adaptive network depth.

    Public interface matches ExpectedSARSA so that train_gpf.py can run
    both agents with identical training loops.
    """

    def __init__(
        self,
        state_shape: tuple,
        n_actions: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        # ---- GPF: grow (Table 7.1: εe, εk) ----
        eps_add: float = 0.01,
        M_add: int = 20,                    # episode window for improvement averaging
        cooldown_episodes: int = 10,        # εk: min episodes between grows
        min_episodes_before_grow: int = 30, # εe: warm-up before GPF activates
        max_hidden_layers: int = 6,
        ema_beta: float = 0.9,
        # ---- GPF: belief (Table 7.1: ϑv, ϑp, φh) ----
        tau_belief_harden: float = 0.02,   # ϑv: weight threshold for belief increment
        tau_belief_prune: float = 0.3,     # ϑp: belief below this → pruneable
        n_total_steps: int = 2_000_000,    # kmax estimate for φh = 1/kmax
        # ---- GPF: prune (Table 7.1: εp) ----
        tau_prune: float = 1e-8,           # OBD saliency threshold
        prune_accum_steps: int = 500,      # εp: steps to accumulate g² before pruning
        max_prune_events: int = 999,       # cap total prune events (prevents cascading)
        # ---- GPF: freeze (Table 7.1: εf, ϑf) ----
        tau_freeze_delta: float = 0.01,    # ϑf: weight change < this = stable
        tau_freeze_frac: float = 0.9,      # fraction of weights that must be stable
        freeze_check_interval: int = 300,  # steps between weight-stability checks
        freeze_patience: int = 3,          # εf: consecutive stable checks to freeze
        min_episodes_before_freeze: int = 100,
        use_eval_trigger_only: bool = False,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.lr = lr

        # Grow
        self.eps_add = eps_add
        self.M_add = M_add
        self.cooldown_episodes = cooldown_episodes
        self.min_episodes_before_grow = min_episodes_before_grow
        self.max_hidden_layers = max_hidden_layers
        self.ema_beta = ema_beta
        self.use_eval_trigger_only = use_eval_trigger_only

        # Belief
        self._tau_belief_harden = tau_belief_harden
        self._tau_belief_prune = tau_belief_prune
        self._belief_phi = 1.0 / max(n_total_steps, 1)

        # Prune
        self.tau_prune = tau_prune
        self.prune_accum_steps = prune_accum_steps
        self.max_prune_events = max_prune_events

        # Freeze
        self.tau_freeze_delta = tau_freeze_delta
        self.tau_freeze_frac = tau_freeze_frac
        self.freeze_check_interval = freeze_check_interval
        self.freeze_patience = freeze_patience
        self.min_episodes_before_freeze = min_episodes_before_freeze

        # Network and optimiser
        input_dim = sum(state_shape)
        self.network = _GPFQNetwork(input_dim, hidden_dim, n_actions)
        self.optimizer = torch.optim.Adam(
            [p for p in self.network.parameters() if p.requires_grad], lr=lr
        )

        # ----------------------------------------------------------------
        # GPF state
        # ----------------------------------------------------------------
        self.frozen_layers: set = set()
        self._steps: int = 0
        self._n_episodes: int = 0
        self._episodes_since_grow: int = 0

        # Grow: episode-level TD error history and EMA (legacy TD-based trigger)
        self._episode_td_buffer: list = []
        self._td_history: deque = deque(maxlen=M_add + 1)
        self._ema_improvement: float = 0.0

        # Grow: eval-success-rate trigger (preferred for RL; set via notify_eval)
        self._eval_sr_history: deque = deque(maxlen=M_add + 1)
        self._evals_since_grow: int = 0

        # Prune: accumulated squared gradients
        self._sq_grad: dict = {}
        self._sq_grad_count: int = 0
        self._prune_pending: bool = False
        self._init_sq_grad_accum()

        # Belief: per-weight hardening values ∈ [0, 1)
        self._belief: dict = {}
        self._init_beliefs()

        # Freeze: weight-stability snapshot tracking
        self._weight_snapshot: dict = {}
        self._freeze_stable_count: list = [0]   # one slot per hidden layer, grows on add
        self._steps_since_freeze_check: int = 0

        # ----------------------------------------------------------------
        # Diagnostics (exposed for plotting in train_gpf.py)
        # ----------------------------------------------------------------
        self.layer_history: list = []
        self.prune_events: list = []
        self.freeze_events: list = []
        self.td_error_history: list = []

    # ------------------------------------------------------------------
    # Belief system (§7.1, φh = 1/kmax)
    # ------------------------------------------------------------------

    def _init_beliefs(self):
        self._belief = {
            n: torch.zeros_like(p.data)
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }

    def _add_belief_slots(self):
        """Allocate belief tensors for parameters of the just-grown layer."""
        for n, p in self.network.named_parameters():
            if n not in self._belief and p.requires_grad:
                self._belief[n] = torch.zeros_like(p.data)

    def _update_beliefs(self):
        """Increment h_ij by φh wherever |w_ij| > ϑv (post optimizer step)."""
        for n, p in self.network.named_parameters():
            if n in self._belief:
                hard = (p.data.abs() > self._tau_belief_harden).float()
                self._belief[n] = (self._belief[n] + self._belief_phi * hard).clamp(max=0.9999)

    # ------------------------------------------------------------------
    # Gradient accumulator helpers
    # ------------------------------------------------------------------

    def _init_sq_grad_accum(self):
        self._sq_grad = {
            n: torch.zeros_like(p.data)
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }
        self._sq_grad_count = 0

    def _accumulate_sq_grads(self):
        for n, p in self.network.named_parameters():
            if p.requires_grad and p.grad is not None and n in self._sq_grad:
                self._sq_grad[n] += p.grad.detach() ** 2
        self._sq_grad_count += 1

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def q_values(self, state: tuple) -> np.ndarray:
        x = _encode(state, self.state_shape)
        with torch.no_grad():
            return self.network(x).numpy()

    def select_action(self, state: tuple, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    # ------------------------------------------------------------------
    # Semi-gradient Expected SARSA update
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, done):
        # ---- TD target (no gradient) ----
        with torch.no_grad():
            if done:
                target = float(reward)
            else:
                nq = self.network(_encode(next_state, self.state_shape)).numpy()
                best = int(np.argmax(nq))
                pi = np.full(self.n_actions, self.epsilon / self.n_actions)
                pi[best] += 1.0 - self.epsilon
                target = float(reward) + self.gamma * float(np.dot(pi, nq))

        # ---- Forward + MSE loss ----
        q_all = self.network(_encode(state, self.state_shape))
        q_sa = q_all[action]
        target_t = torch.tensor(target, dtype=torch.float32)
        loss = F.mse_loss(q_sa, target_t)
        td_error = float((q_sa - target_t).detach())

        # ---- Backward ----
        self.optimizer.zero_grad()
        loss.backward()
        self._accumulate_sq_grads()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ---- Belief update (post weight-update) ----
        self._update_beliefs()

        # ---- Bookkeeping ----
        self._steps += 1
        self._episode_td_buffer.append(abs(td_error))
        self.td_error_history.append(abs(td_error))

        # ---- Deferred prune ----
        if self._prune_pending and self._sq_grad_count >= self.prune_accum_steps:
            if len(self.prune_events) < self.max_prune_events:
                self._prune_preceding_layers()
            self._prune_pending = False

        # ---- Weight-stability check for freeze ----
        self._steps_since_freeze_check += 1
        if self._steps_since_freeze_check >= self.freeze_check_interval:
            self._check_weight_stability()
            self._maybe_freeze_if_ready()
            self._steps_since_freeze_check = 0

    # ------------------------------------------------------------------
    # ε decay + episode-level grow trigger
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Called once per episode end.  Decays ε and checks the grow trigger."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._n_episodes += 1
        self._episodes_since_grow += 1

        # Flush episode TD buffer and check grow trigger
        if self._episode_td_buffer:
            ep_avg = float(np.mean(self._episode_td_buffer))
            self._episode_td_buffer = []
            self._maybe_grow(ep_avg)

    # ------------------------------------------------------------------
    # Eval-based grow trigger (RL-appropriate: use success rate, not TD error)
    # ------------------------------------------------------------------

    def notify_eval(self, success_rate: float):
        """
        Call this after every evaluation run with the greedy success rate.

        In RL, TD error is non-monotone and its improvement EMA stays near zero
        regardless of actual learning progress.  Success rate is a cleaner signal:
        grow when it has been flat (< eps_add improvement over M_add evals).

        This overrides the episode-level TD-error grow trigger when called.
        """
        if self.network.n_hidden >= self.max_hidden_layers:
            return

        self._eval_sr_history.append(success_rate)
        self._evals_since_grow += 1

        if self._n_episodes < self.min_episodes_before_grow:
            return
        if len(self._eval_sr_history) < 2:
            return

        # Minimum cooldown in *evals* (converted from episode cooldown / eval_every).
        # We store as episodes; here we use a raw eval count of at least 2.
        eval_cooldown = max(2, self.cooldown_episodes // 500)
        if self._evals_since_grow < eval_cooldown:
            return

        sr_list = list(self._eval_sr_history)
        best_recent = max(sr_list[-max(2, len(sr_list) // 2):])
        best_early  = max(sr_list[:max(1, len(sr_list) // 2)])
        improvement = best_recent - best_early

        if improvement < self.eps_add:
            n_before = self.network.n_hidden
            new_layer = self.network.add_hidden_layer()
            self._add_belief_slots()
            self._freeze_stable_count.append(0)
            print(
                f"[GPF] GROW (eval-trigger)  layer {n_before}→{self.network.n_hidden}  "
                f"(ep={self._n_episodes}, sr_improve={improvement:.4g}, "
                f"best_sr={max(sr_list):.2%})"
            )
            self._init_sq_grad_accum()
            self._prune_pending = True
            self._optimizer_add_layer(new_layer)
            self._episodes_since_grow = 0
            self._evals_since_grow = 0

    # ------------------------------------------------------------------
    # GPF — GROW (episode-level, Table 7.1: εe / εk)
    # ------------------------------------------------------------------

    def _maybe_grow(self, ep_avg_td: float):
        """Check grow trigger using episode-averaged TD error."""
        if self.use_eval_trigger_only:
            return
        if self.network.n_hidden >= self.max_hidden_layers:
            return

        self._td_history.append(ep_avg_td)

        # Warm-up guard (εe): don't grow until the agent has had time to learn
        if self._n_episodes < self.min_episodes_before_grow:
            return
        # Cooldown guard (εk)
        if self._episodes_since_grow < self.cooldown_episodes:
            return
        if len(self._td_history) < self.M_add + 1:
            return

        td_list = list(self._td_history)
        improvements = [td_list[i - 1] - td_list[i] for i in range(1, len(td_list))]
        avg_improve = sum(improvements) / len(improvements)

        # EMA update — intentionally NOT reset after a grow event.
        # Resetting to 0 was a v1 bug: it made the condition trivially true
        # right after the next cooldown expired, causing runaway depth growth.
        self._ema_improvement = (
            self.ema_beta * self._ema_improvement
            + (1.0 - self.ema_beta) * avg_improve
        )

        if self._ema_improvement < self.eps_add:
            n_before = self.network.n_hidden
            new_layer = self.network.add_hidden_layer()
            self._add_belief_slots()
            self._freeze_stable_count.append(0)
            print(
                f"[GPF] GROW  layer {n_before}→{self.network.n_hidden}  "
                f"(ep={self._n_episodes}, EMA_improve={self._ema_improvement:.4g})"
            )
            self._init_sq_grad_accum()
            self._prune_pending = True
            self._optimizer_add_layer(new_layer)
            self._episodes_since_grow = 0

    # ------------------------------------------------------------------
    # GPF — PRUNE (OBD saliency + belief protection, §7.1)
    # ------------------------------------------------------------------

    def _prune_preceding_layers(self):
        """
        Prune all hidden layers except the newest.

        Primary criterion: OBD saliency = 0.5 · w² · E[g²] < tau_prune → zero.
        Protection: belief-hardened weights (h ≥ tau_belief_prune) survive
                    regardless of their saliency, preserving sparse but important
                    connections as described in §7.1.
        """
        if self._sq_grad_count == 0:
            return

        n_to_prune = self.network.n_hidden - 1
        total_before = total_after = 0

        for layer_idx in range(n_to_prune):
            if layer_idx in self.frozen_layers:
                continue
            layer = self.network.hidden_layers[layer_idx]
            for pname, param in layer.named_parameters():
                full_name = f"hidden_layers.{layer_idx}.{pname}"
                if full_name not in self._sq_grad:
                    continue
                avg_sq_grad = self._sq_grad[full_name] / self._sq_grad_count
                saliency = 0.5 * (param.data ** 2) * avg_sq_grad
                obd_survive = (saliency > self.tau_prune).float()

                # Belief-hardened weights are immune to pruning (§7.1)
                if full_name in self._belief:
                    belief_protected = (self._belief[full_name] >= self._tau_belief_prune).float()
                    mask = (obd_survive + belief_protected).clamp(max=1.0)
                else:
                    mask = obd_survive

                total_before += param.numel()
                total_after += int(mask.sum().item())
                param.data.mul_(mask)

        keep_ratio = total_after / max(total_before, 1)
        self.prune_events.append((self._steps, keep_ratio))
        print(
            f"[GPF] PRUNE {n_to_prune} layer(s)  "
            f"kept {keep_ratio:.2%} of weights  (step={self._steps})"
        )

    # ------------------------------------------------------------------
    # GPF — FREEZE (weight-stability snapshots, §7.1 Eq. 7.4)
    # ------------------------------------------------------------------

    def _take_weight_snapshot(self):
        """Store current weights for next stability comparison."""
        self._weight_snapshot = {
            n: p.data.clone()
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }

    def _check_weight_stability(self):
        """
        Compare weights against the last snapshot and update per-layer
        stability counters.  A weight is "stable" if |Δw| < tau_freeze_delta
        (ϑf in Eq. 7.4).  A layer's counter increments when ≥ tau_freeze_frac
        of its weights are stable.
        """
        if not self._weight_snapshot:
            self._take_weight_snapshot()
            return

        for i in range(self.network.n_hidden):
            if i in self.frozen_layers or i >= len(self._freeze_stable_count):
                continue
            layer = self.network.hidden_layers[i]
            stable_w = total_w = 0
            for pname, param in layer.named_parameters():
                full_name = f"hidden_layers.{i}.{pname}"
                if full_name not in self._weight_snapshot:
                    continue
                delta = (param.data - self._weight_snapshot[full_name]).abs()
                stable_w += int((delta < self.tau_freeze_delta).sum())
                total_w += param.numel()
            stable_frac = stable_w / max(total_w, 1)
            if stable_frac >= self.tau_freeze_frac:
                self._freeze_stable_count[i] += 1
            else:
                self._freeze_stable_count[i] = 0

        self._take_weight_snapshot()

    def _maybe_freeze_if_ready(self):
        """Freeze any eligible layer that has been stable for freeze_patience checks."""
        if self._n_episodes < self.min_episodes_before_freeze:
            return
        newest_idx = self.network.n_hidden - 1
        for i in range(self.network.n_hidden):
            if i in self.frozen_layers or i == newest_idx:
                continue
            if (i < len(self._freeze_stable_count)
                    and self._freeze_stable_count[i] >= self.freeze_patience):
                self._freeze_layer(i)

    def _freeze_layer(self, layer_idx: int):
        layer = self.network.hidden_layers[layer_idx]
        for p in layer.parameters():
            p.requires_grad_(False)
        self.frozen_layers.add(layer_idx)
        self.freeze_events.append((self._steps, layer_idx))
        if layer_idx < len(self._freeze_stable_count):
            self._freeze_stable_count[layer_idx] = 0
        self._rebuild_optimizer()
        print(
            f"[GPF] FREEZE layer {layer_idx}  "
            f"(weight stable ≥{self.tau_freeze_frac:.0%} for {self.freeze_patience} checks, "
            f"step={self._steps})"
        )

    # ------------------------------------------------------------------
    # Optimiser management (grow → add_param_group; freeze → state-preserving)
    # ------------------------------------------------------------------

    def _optimizer_add_layer(self, new_layer: nn.Module):
        """Add a just-grown layer's parameters as a new param group.

        Using add_param_group preserves all accumulated Adam first/second moment
        estimates for the existing layers — no momentum is discarded on grow.
        """
        new_params = [p for p in new_layer.parameters() if p.requires_grad]
        self.optimizer.add_param_group({'params': new_params, 'lr': self.lr})

    def _rebuild_optimizer(self):
        """Rebuild optimizer after a freeze event, preserving Adam state.

        After a layer is frozen its parameters are excluded from the new
        optimizer's param list.  We copy the Adam state tensors (exp_avg,
        exp_avg_sq, step) for every parameter that survived, so the frozen
        layer's removal doesn't reset the momentum of the remaining layers.
        """
        old_state = dict(self.optimizer.state)
        trainable = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.lr)
        for p in trainable:
            if p in old_state:
                self.optimizer.state[p] = old_state[p]

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
            'n_hidden': self.network.n_hidden,
            'frozen_layers': list(self.frozen_layers),
        }, p)

    def load(self, path: str):
        p = str(path)
        if not p.endswith('.pt'):
            p += '.pt'
        ckpt = torch.load(p, map_location='cpu')
        while self.network.n_hidden < ckpt['n_hidden']:
            self.network.add_hidden_layer()
            self._add_belief_slots()
            self._freeze_stable_count.append(0)
        self.network.load_state_dict(ckpt['network_state_dict'])
        self.epsilon = float(ckpt.get('epsilon', self.epsilon))
        for idx in ckpt.get('frozen_layers', []):
            if idx not in self.frozen_layers:
                for pp in self.network.hidden_layers[idx].parameters():
                    pp.requires_grad_(False)
                self.frozen_layers.add(idx)
        self._rebuild_optimizer()
