import numpy as np


class Env:
    def __init__(self, cap):
        self.cap = cap
        self.word = np.zeros(self.cap)
        self.cursor = 0

    def step(self, action):
        if self.cursor >= self.cap or not action:
            return self.word, True

        self.word[self.cursor] = action
        self.cursor += 1
        return self.word, self.cursor == self.cap

    def reset(self):
        self.word = np.zeros(self.cap)
        self.cursor = 0
        return self.word

    def reward(self):
        counter = np.zeros(26)
        for c in self.word[:self.cursor]:
            counter[int(c - 1)] += 1
        if (r := counter.max()) <= self.cap / 2:
            return r
        else:
            return self.cap - r

    def mask(self):
        backward = [0] + [1] * 26
        if self.cursor == self.cap:
            return [1] + [0] * 26, backward
        else:
            return [1] * 27, backward

    def render(self):
        chars = self.word[:self.cursor]
        word = ''.join(chr(int(i) + 64) for i in chars)
        print(word)
