import torch.nn as nn
import torch
import util

class Actor(nn.Module):
    """
    Actor is on policy. 
    Network shape : states by action


    """
    def __init__(self, 
                n_state, 
                n_action,
                seed, 
                h1=400,
                h2=300):
        super(Actor, self).__init__() 
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_state, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_action)
        self.flush_weights()

    def flush_weights(self):
        self.fc1.weight.data.uniform_(*util.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*util.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.tanh(x)
        return x

    