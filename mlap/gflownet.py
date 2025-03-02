import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adam

from envs.word import Env


@dataclass
class Config:
    seed = 42
    learning_rate: float = 1e-4
    hidden_dim: int = 512
    num_epochs: int = 500
    update_freq: int = 20


class Proxy:
    def __init__(self):
        self.__env = Env(10)
        self.state_dim = 10
        self.num_actions = 27

    def step(self, action) -> Tuple[Tensor, bool]:
        state, done = self.__env.step(action)
        return torch.tensor(state).float(), done

    def reset(self) -> Tensor:
        state = self.__env.reset()
        return torch.tensor(state).float()

    def reward(self) -> Tensor:
        reward = self.__env.reward()
        return torch.tensor(reward).float()

    def mask(self) -> Tuple[Tensor, Tensor]:
        f, b = self.__env.mask()
        return torch.tensor(f).bool(), torch.tensor(b).bool()

    def render(self):
        self.__env.render()


class Agent(nn.Module):
    def __init__(self, proxy: Proxy, config: Config) -> None:
        super().__init__()
        self.proxy = proxy
        self.config = config
        self.log_z = nn.Parameter(torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(proxy.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, proxy.num_actions * 2)
        )

    def forward(self, s):
        logits = self.mlp(s)
        f_mask, b_mask = self.proxy.mask()
        f_logits = logits[:self.proxy.num_actions] * f_mask + ~f_mask * -100
        b_logits = logits[self.proxy.num_actions:] * b_mask + ~b_mask * -100
        return f_logits, b_logits


def train(config: Config) -> Agent:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    proxy = Proxy()
    agent = Agent(proxy, config)
    optimizer = Adam(agent.parameters(), lr=config.learning_rate)

    batch_loss = torch.zeros(1)

    for episode in range(config.num_epochs * config.update_freq):
        sum_log_p_f = torch.zeros(1)
        sum_log_p_b = torch.zeros(1)

        state = proxy.reset()
        f_logits, b_logits = agent(state)
        done = False

        while not done:
            cat = Categorical(logits=f_logits)
            action = cat.sample()
            sum_log_p_f += cat.log_prob(action)

            state, done = proxy.step(action)
            f_logits, b_logits = agent(state)
            sum_log_p_b += Categorical(logits=b_logits).log_prob(action)

        log_r = torch.log(proxy.reward()).clip(-20)
        loss = (agent.log_z + sum_log_p_f - log_r - sum_log_p_b).pow(2)
        batch_loss += loss

        if episode % config.update_freq == 0:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss = torch.zeros(1)

    return agent


if __name__ == '__main__':
    configuration = Config()
    _ = train(configuration)
