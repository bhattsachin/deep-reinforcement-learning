import numpy as np
import torch


class Past:

    def __init__(self, past):
        self.past = past
        
    def states(self):
        return torch.from_numpy(np.vstack([event.state for event in self.past if event is not None])).float().to(device)
        
    def actions(self):
        return torch.from_numpy(np.vstack([event.action for event in self.past if event is not None])).float().to(device)

    def rewards(self):
        return torch.from_numpy(np.vstack([event.reward for event in self.past if event is not None])).float().to(device)

    def next_states(self):
        return torch.from_numpy(np.vstack([event.next_state for event in self.past if event is not None])).float().to(device)

    def dones(self):
        return torch.from_numpy(np.vstack([event.done for event in self.past if event is not None])).float().to(device)
