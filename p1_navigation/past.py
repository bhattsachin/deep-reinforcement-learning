import numpy as np
import torch


class Past:

    def __init__(self, past):
        self.past = past
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def states(self):
        return torch.from_numpy(np.vstack([event.state for event in self.past if event is not None])).float().to(self.device)
        
    def actions(self):
        return torch.from_numpy(np.vstack([event.action for event in self.past if event is not None])).long().to(self.device)

    def rewards(self):
        return torch.from_numpy(np.vstack([event.reward for event in self.past if event is not None])).float().to(self.device)

    def next_states(self):
        return torch.from_numpy(np.vstack([event.next_state for event in self.past if event is not None])).float().to(self.device)

    def dones(self):
        return torch.from_numpy(np.vstack([event.done for event in self.past if event is not None]).astype(np.uint8)).float().to(self.device)
