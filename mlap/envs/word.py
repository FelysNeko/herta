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
        if self.cursor == 0:
            return 0
        counter = [0] * 26
        for c in self.word[:self.cursor]:
            counter[int(c - 1)] += 1
        return max(counter)

    def render(self):
        ascii = self.word[:self.cursor]
        word = ''.join(chr(int(i) + 64) for i in ascii)
        print(word)