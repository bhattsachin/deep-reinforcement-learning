
import random
from collections import deque
import torch
import numpy as np

class SARSD:
    def __init__(self, states, actions, rewards, states_next, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.states_next = states_next
        self.dones = dones
    

class Memory:
    def __init__(self, buffer_size, device, seed):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.seed = random.seed(seed)

    def add(self, event):
        self.buffer.append(event)

    def sample(self, N):
        #only if N<len(self.buffer)
        if N < len(self.buffer):
            return random.sample(self.buffer, k=N)
        return None

    def deserialize(self, events):
        states = self.sq('state', events)
        actions = self.sq('action', events)
        rewards = self.sq('reward', events)
        states_next = self.sq('state_next', events)
        dones = self.sq('done', events)
        return SARSD(states, actions, rewards, states_next, dones) 
    
    def sq(self, property, obj):
        if property is not 'done':
            return torch.from_numpy(np.vstack([getattr(e, property) for e in obj if e is not None])).float().to(self.device)
        else:
            return torch.from_numpy(np.vstack([getattr(e, property) for e in obj if e is not None]).astype(np.uint8)).float().to(self.device)
