"""Soft Actor-Critic with twin critics and automatic entropy tuning."""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return tuple(np.stack(x) for x in zip(*batch))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Gaussian policy squashed by tanh to [-1, 1]."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, s):
        h = F.relu(self.l2(F.relu(self.l1(s))))
        return self.mean(h), torch.clamp(self.log_std(h), -20, 2)

    def sample(self, s):
        """Reparameterized action and its log-probability."""
        mean, log_std = self(s)
        normal = Normal(mean, log_std.exp())
        x = normal.rsample()
        a = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(1, keepdim=True)


class Critic(nn.Module):
    """Two independent Q networks."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()

        def q():
            return nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

        self.q1_net, self.q2_net = q(), q()

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1_net(sa), self.q2_net(sa)


class SACAgent:
    """SAC agent. Entropy coefficient starts at 1 and targets -action_dim."""

    def __init__(self, state_dim=10, action_dim=2, hidden=256, lr=3e-4, gamma=0.99,
                 tau=0.005, buffer_size=1_000_000):
        self.gamma, self.tau = gamma, tau
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        self.actor = Actor(state_dim, action_dim, hidden)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(state_dim, action_dim, hidden)
        self.critic_target = Critic(state_dim, action_dim, hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    @torch.no_grad()
    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            return torch.tanh(self.actor(s)[0]).numpy()[0]
        return self.actor.sample(s)[0].numpy()[0]

    def update(self, batch_size: int = 256) -> None:
        """One gradient step on critics, actor, and entropy coefficient."""
        if len(self.replay_buffer) < batch_size:
            return
        s, a, r, s2, d = (torch.as_tensor(x, dtype=torch.float32)
                          for x in self.replay_buffer.sample(batch_size))
        r, d = r.unsqueeze(1), d.unsqueeze(1)
        if a.dim() == 1:
            a = a.unsqueeze(1)

        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q_next = torch.min(*self.critic_target(s2, a2)) - self.alpha * logp2
            y = r + self.gamma * (1.0 - d) * q_next

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_new, logp = self.actor.sample(s)
        actor_loss = (self.alpha * logp - torch.min(*self.critic(s, a_new))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha.exp() * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

    def save(self, episode: int, out_dir: str) -> None:
        """Save the actor alone and a full checkpoint for resuming."""
        torch.save(self.actor.state_dict(), os.path.join(out_dir, f"sac_actor_ep{episode}.pth"))
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }, os.path.join(out_dir, f"sac_full_ep{episode}.pth"))

    def load(self, path: str) -> None:
        ck = torch.load(path, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_target.load_state_dict(ck["critic_target"])
        self.log_alpha.data = ck["log_alpha"]
        self.alpha = self.log_alpha.exp().item()
        self.actor_opt.load_state_dict(ck["actor_opt"])
        self.critic_opt.load_state_dict(ck["critic_opt"])
        self.alpha_opt.load_state_dict(ck["alpha_opt"])
