import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from envs.word import Env


class Proxy:
    def __init__(self):
        self.env = Env(10)
        self.state_dim = self.env.cap
        self.num_actions = 27

    def step(self, action):
        s, done = self.env.step(action)
        return torch.tensor(s).float(), torch.tensor(done).bool()

    def reset(self):
        return torch.tensor(self.env.reset()).float()

    def reward(self):
        return torch.tensor(self.env.reward()).float()

    def render(self):
        self.env.render()


class Agent(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.log_z = nn.Parameter(torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * 2)
        )

    def forward(self, s):
        logits = self.mlp(s)
        f_logits = logits[:self.num_actions]
        b_logits = logits[self.num_actions:]
        return f_logits, b_logits


def train():
    proxy = Proxy()
    agent = Agent(proxy.state_dim, 512, proxy.num_actions)
    optimizer = Adam(agent.parameters(), lr=3e-4)

    for episode in range(10000):
        sum_log_p_f = 0
        sum_log_p_b = 0

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

        reward = torch.log(proxy.reward()).clip(-20)
        loss = (agent.log_z + sum_log_p_f - reward - sum_log_p_b).pow(2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    train()
