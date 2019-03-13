import torch
import torch.nn as nn

class Critic(nn.Module):
    """ Critic will take both state
    and action from actor and compare 
    with loss function. 

    How does ratio of h1 : n_action 
    effect overall.
    Q - Why critic.h1 == actor.h1
    """

    def __init__(self,
                seed,
                n_state,
                n_action,
                h1=128,
                h2=128,
                h3=64):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_state, h1)
        self.fc2 = nn.Linear(h1 + n_action, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3,1)
        self.flush_weights()

    def flush_weights(self):
        low = -1e4
        high = 1e4
        self.fc1.weight.data.uniform_(low, high)
        self.fc2.weight.data.uniform_(low, high)
        self.fc3.weight.data.uniform_(low, high)
        self.fc4.weight.data.uniform_(low, high)

    def forward(self, state, action):
        x_state = self.fc1(state)
        x_state = nn.functional.relu(x_state)
        x = torch.cat((x_state, action), dim=1) 
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        return x
