import random
from dataclasses import dataclass

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
    hidden_dim: int = 256
    num_epochs: int = 2000
    batch_size: int = 50

    state_dim: int = 11 * 10
    num_actions: int = 10


def reset(batch_size: int) -> Tensor:
    state = torch.zeros(batch_size, 10).long()
    state = one_hot(state, 11).view(batch_size, -1)
    return state.float()


def step(s: Tensor, actions: Tensor) -> Tensor:
    s = s.view(s.shape[0], -1, 11).argmax(dim=2)
    index = s[0].argmin()
    s[:, index] = actions + 1
    s = one_hot(s, 11).view(s.shape[0], -1)
    return s.float()


def proxy(s: Tensor) -> Tensor:
    s = s.view(s.shape[0], -1, 11).argmax(dim=2)
    x = s.sum(dim=1)
    reward = torch.zeros(s.shape[0])
    reward[(20 <= x) & (x <= 30)] = 1
    return reward


class Agent(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.log_z = nn.Parameter(torch.zeros(1))
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.forward_policy = nn.Linear(hidden_dim, num_actions)
        self.backward_policy = nn.Linear(hidden_dim, num_actions)

    def forward(self, s):
        x = self.trunk(s)
        f_logits = self.forward_policy(x)
        b_logits = self.backward_policy(x)
        return f_logits, b_logits

    def sample(self, batch_size: int):
        states = reset(batch_size)
        for _ in range(10):
            logits, _ = self(states)
            actions = Categorical(logits=logits).sample()
            states = step(states, actions)
        states = states.view(states.shape[0], -1, 11).argmax(dim=2)
        return states


def train(config: Config) -> Agent:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    agent = Agent(config.state_dim, config.hidden_dim, config.num_actions)
    optimizer = Adam(agent.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        states = reset(config.batch_size)
        sum_log_prob_f = torch.zeros(config.batch_size)
        sum_log_prob_b = torch.zeros(config.batch_size)

        f_logits, b_logits = agent(states)
        for _ in range(10):
            cat = Categorical(logits=f_logits)
            actions = cat.sample()
            states = step(states, actions)

            f_logits, b_logits = agent(states)
            sum_log_prob_f += cat.log_prob(actions)
            sum_log_prob_b += Categorical(logits=b_logits).log_prob(actions)

        log_r = torch.log(proxy(states)).clip(-20)
        loss = (agent.log_z + sum_log_prob_f - log_r - sum_log_prob_b).pow(2)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(
                f'{epoch:<11}',
                f'{loss.mean().item():<11.3f}',
                f'{agent.log_z.exp().item():<11.3f}',
                *proxy(states).unique(return_counts=True)
            )

    return agent


if __name__ == '__main__':
    configuration = Config()
    samples = train(configuration).sample(10).numpy()
    print(samples)
