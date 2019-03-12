import torch.nn as nn
import torch

class Actor(nn.Module):
    """
    Actor is on policy. 
    Network shape : states by action


    """
    def __init__(self, 
                seed, 
                n_state, 
                n_action, 
                h1=128,
                h2=128):
        super(Actor, self).__init__() 
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_state, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.flush_weights()

    def flush_weights(self):
        self.fc1.weight.data.uniform_(-1e4, 1e4)
        self.fc2.weight.data.uniform_(-1e4, 1e4)

    def forward(self, state):
        x = self.fc1(state)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.tanh(x)
        return x

    