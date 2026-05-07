"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import os
import math
import struct
from dataclasses import dataclass

import numpy as np
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import torch.optim as optim

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 32
    vocab_size: int = 50257
    n_layer: int = 1 #12
    n_head: int = 8
    n_embd: int = 64


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        # model = GPT(config)
        model = AdaptiveGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class AdaptiveGPT(GPT):
    """
    GPT variant that dynamically adds transformer layers and prunes neurons
    using Optimal Brain Damage (LeCun 1989).
    """
    def __init__(self, config,
                 ε_add=1e-3, M_add=3, τ_prune=1e-8,
                 max_layers=12, ema_beta=0.9, cooldown_steps=200):
        super().__init__(config)
        # Adaptive training hyperparameters
        self.ε_add = ε_add
        self.M_add = M_add
        self.τ_prune = τ_prune
        self.max_layers = max_layers
        self.ema_beta = ema_beta
        self.cooldown_steps = cooldown_steps
        # Tracking
        self.loss_history = []
        self.ema_improvement = 0.0
        self.steps_since_last_add = 0
        self.layer_history = []       # number of layers at each step
        self.prune_ratios = []        # % weights kept after each prune step
    
    @torch.no_grad()
    def maybe_add_layer(self):
        """Dynamically add a layer if loss stagnates."""
        if len(self.loss_history) < self.M_add + 1:
            return
        # Compute average loss improvement
        improvements = [
            self.loss_history[i - 1] - self.loss_history[i]
            for i in range(1, len(self.loss_history))
        ]
        recent_improvements = improvements[-self.M_add:]
        avg_improve = sum(recent_improvements) / len(recent_improvements)
        # EMA smoothing
        self.ema_improvement = (
            self.ema_beta * self.ema_improvement +
            (1 - self.ema_beta) * avg_improve
        )
        # Cooldown enforcement
        if self.steps_since_last_add < self.cooldown_steps:
            self.steps_since_last_add += 1
            return
        # Add layer if improvement stagnates
        if self.ema_improvement < self.ε_add and len(self.transformer.h) < self.max_layers:
            print(f"[AdaptiveGPT] Adding new layer ({len(self.transformer.h)+1}) | EMA improve={self.ema_improvement:.4g}")
            self._add_new_layer()
            prune_ratio = self.prune_layers()
            self.prune_ratios.append(prune_ratio)
            self.steps_since_last_add = 0
    
    def _add_new_layer(self):
        """Clone last transformer block and append as new layer."""
        new_block = copy.deepcopy(self.transformer.h[-1])
        for p in new_block.parameters():
            p.data += 0.01 * torch.randn_like(p)
        self.transformer.h.append(new_block.to(next(self.parameters()).device))
        self.config.n_layer += 1
   
    def prune_layers(self):
        """Prune low-saliency weights using Optimal Brain Damage."""
        print("[AdaptiveGPT] Performing saliency pruning...")
        total_before, total_after = 0, 0
        for i, block in enumerate(self.transformer.h[:-1]):  # skip newest layer
            for name, param in block.named_parameters():
                if param.requires_grad and param.grad is not None:
                    hessian_approx = param.grad**2
                    saliency = 0.5 * (param.data**2) * hessian_approx
                    mask = saliency > self.τ_prune
                    before = param.numel()
                    after = mask.sum().item()
                    total_before += before
                    total_after += after
                    param.data *= mask  # zero pruned weights
                    print(f"  Layer {i} | {name}: kept {after}/{before} ({after/before:.2%})")
        prune_ratio = total_after / max(total_before, 1)
        print(f"[AdaptiveGPT] Overall pruning ratio: {prune_ratio:.2%}")
        return prune_ratio
    
    def training_step(self, optimizer, x, y):
        """Perform one training iteration with dynamic adaptation."""
        logits, loss = self(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        # Log metrics
        self.loss_history.append(loss.item())
        self.layer_history.append(len(self.transformer.h))
        # Attempt to adapt
        self.maybe_add_layer()
        return loss.item()
    
def load_training_tokens(device):
    # Match the baseline loader exactly: prefer tiny_shakespeare, otherwise TinyStories.
    shake_tokens_bin = "data/tiny_shakespeare_val.bin"
    story_tokens_bin = "data/TinyStories_val.bin"
    assert os.path.isfile(shake_tokens_bin) or os.path.isfile(story_tokens_bin), "you must run prepro on some dataset"
    tokens_bin = shake_tokens_bin if os.path.isfile(shake_tokens_bin) else story_tokens_bin
    assert os.path.isfile(tokens_bin)
    print(f"loading cached tokens in {tokens_bin}")
    with open(tokens_bin, "rb") as f:
        tokens = np.frombuffer(f.read(), dtype=np.int32)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    return tokens


def make_data_iter(tokens, batch_size, block_size):
    assert batch_size * block_size + 1 <= len(tokens), "not enough tokens"
    i = 0
    while True:
        x = tokens[i:i + batch_size * block_size].view(batch_size, block_size)
        y = tokens[i + 1:i + batch_size * block_size + 1].view(batch_size, block_size)
        yield x, y
        i += batch_size * block_size
        if i + batch_size * block_size + 1 >= len(tokens):
            i = 0


# --- Training with baseline dataset and adaptive model ---
def train_adaptive_gpt(num_steps=10000, batch_size=8, block_size=32):
    assert 1 <= block_size <= 1024
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    tokens = load_training_tokens(device)
    data_iter = make_data_iter(tokens, batch_size, block_size)

    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=50257,
        n_layer=1,
        n_head=8,
        n_embd=64,
    )
    model = AdaptiveGPT(model_config).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(num_steps):
        x, y = next(data_iter)
        loss_val = model.training_step(optimizer, x, y)
        if step % 50 == 0:
            print(f"Step {step} | Loss={loss_val:.4f} | Layers={len(model.transformer.h)}")
    # --- Visualization ---
    steps = list(range(len(model.loss_history)))
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.plot(steps, model.loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 3, 2)
    plt.plot(steps, model.layer_history)
    plt.title("Number of Layers")
    plt.xlabel("Epoch")
    plt.ylabel("Layers")
    plt.subplot(1, 3, 3)
    plt.plot(model.prune_ratios)
    plt.title("Average Pruning Ratio per Adaptation")
    plt.xlabel("Adaptation Event")
    plt.ylabel("Ratio of Weights Retained")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=10000, help="number of iterations to run")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    args = parser.parse_args()

    train_adaptive_gpt(
        num_steps=args.num_iterations,
        batch_size=args.batch_size,
        block_size=args.sequence_length,
    )
