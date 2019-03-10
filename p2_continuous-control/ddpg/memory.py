
import random
from collections import deque

class Memory:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, event):
        self.buffer.append(event)

    def sample(self, N):
        #only if N<len(self.buffer)
        if N < len(self.buffer):
            return random.sample(self.buffer, N)
