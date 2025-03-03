import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.functional import one_hot
from torch.optim import Adam


@dataclass
class Config:
    seed = 0
    learning_rate: float = 1e-4
    hidden_dim: int = 1024
    num_epochs: int = 1000
    batch_size: int = 1000

    state_dim: int = 27 * 10
    num_actions: int = 27


def reset(config: Config) -> Tensor:
    state = torch.zeros(config.batch_size, 10).long()
    state = one_hot(state, 27).view(config.batch_size, -1)
    return state.float()


def step(s: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
    s = s.view(s.shape[0], -1, 27).argmax(dim=2)
    idx = s.argmin(dim=1)
    t = ~(s[:, -1] != 0) & actions.bool()
    s[t, idx[t]] = actions[t]
    dones = s[:, -1] != 0
    s = one_hot(s, 27).view(s.shape[0], -1)
    return s.float(), dones | ~actions.bool()


def mask(s: Tensor) -> Tuple[Tensor, Tensor]:
    s = s.view(s.shape[0], -1, 27).argmax(dim=2)
    t = s[:, -1] != 0
    f = torch.ones(s.shape[0], 27)
    f[t, 1:] = 0
    b = torch.ones(s.shape[0], 27)
    b[:, 0] = 0
    return f.bool(), b.bool()


def proxy(s: Tensor) -> Tensor:
    s = s.view(s.shape[0], -1, 27).argmax(dim=2)
    x = s.sum(dim=1)
    reward = torch.zeros(s.shape[0])
    reward[x >= 210] = 5
    reward[x >= 220] = 6
    reward[x >= 230] = 7
    reward[x >= 240] = 8
    reward[x >= 250] = 9
    return reward


class Agent(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.log_z = nn.Parameter(torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_actions * 2)
        )

    def forward(self, s):
        logits = self.mlp(s)
        f_m, b_m = mask(s)
        f_logits = logits[:, :self.num_actions] * f_m + ~f_m * -100
        b_logits = logits[:, self.num_actions:] * b_m + ~b_m * -100
        return f_logits, b_logits


def train(config: Config) -> Agent:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    agent = Agent(config.state_dim, config.hidden_dim, config.num_actions)
    optimizer = Adam(agent.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        states = reset(config)
        dones = torch.zeros(config.batch_size).bool()
        sum_log_prob_f = torch.zeros(config.batch_size)
        sum_log_prob_b = torch.zeros(config.batch_size)

        while not dones.all():
            f_logits, _ = agent(states[~dones])
            cat = Categorical(logits=f_logits)
            action = cat.sample()

            states[~dones], next_dones = step(states[~dones], action)
            _, b_logits = agent(states[~dones])

            sum_log_prob_f[~dones] += cat.log_prob(action)
            sum_log_prob_b[~dones] += Categorical(logits=b_logits).log_prob(action)

            dones[~dones] = next_dones

        log_r = torch.log(proxy(states)).clip(-20)
        loss = (agent.log_z + sum_log_prob_f - log_r - sum_log_prob_b).pow(2)

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f'{epoch:<7}',
                f'{loss.sum().item():<13.3f}',
                f'{agent.log_z.exp().item():<13.6f}',
                *proxy(states).unique(return_counts=True)
            )

    return agent


if __name__ == '__main__':
    configuration = Config()
    _ = train(configuration)
