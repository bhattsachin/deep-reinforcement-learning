
from actor import Actor
from critic import Critic
from noise import OUNoise as noise
from memory import Memory as memory 
import torch.nn as nn
import torch
from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class DDPGAgent:
    def __init__(self,
                seed,
                n_state,
                n_action,
                batch_size=64,
                buffer=1e6,
                gamma=0.9,
                lr_actor=1e-3,
                lr_critic=1e-3,
                weight_decay=1e-3 
                ):

        #init actor
        self.local_actor = Actor(seed,n_state, n_action).to(device)
        self.target_actor = Actor(seed, n_state, n_action).to(device)
        self.optim_actor = torch.optim.Adam(self.local_actor.parameters(), lr=lr_actor)  
        #init critic
        self.local_critic = Critic(seed, n_state, n_action)
        self.target_critic = Critic(seed, n_state, n_action)
        self.optim_critic = torch.optim.Adam(self.local_critic.parameters(), lr=lr_critic)

        #init memory
        self.memory = memory(int(buffer))
        



    def step(self, state, action, reward, next_state, done):
        
        pass

    def act(self):
        pass

if __name__ == '__main__':
    ddpg_agent = DDPGAgent(30, n_state=30, n_action=5)
    

        
