import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, 
                seed, 
                nState, 
                nAction, 
                hidden_layers):
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(nState, )

        pass

    def forward(self):
        pass

    