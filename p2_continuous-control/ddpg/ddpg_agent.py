
from actor import Actor
from critic import Critic
from noise import OUNoise as noise
from memory import Memory as memory 
from event import Event
import torch.nn as nn
import torch
from collections import deque
import numpy as np

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
                weight_decay=1e-3,
                tau=1e-3
                ):
        self.batch_size = batch_size

        #init actor
        self.local_actor = Actor(seed,n_state, n_action).to(device)
        self.target_actor = Actor(seed, n_state, n_action).to(device)
        self.optim_actor = torch.optim.Adam(self.local_actor.parameters(), lr=lr_actor)  
        #init critic
        self.local_critic = Critic(seed, n_state, n_action)
        self.target_critic = Critic(seed, n_state, n_action)
        self.optim_critic = torch.optim.Adam(self.local_critic.parameters(), lr=lr_critic)

        #init memory
        self.memory = memory(int(buffer), device)
        self.tau = tau
        self.noise = noise(n_action, seed=seed)
        
    def step(self, state, action, reward, next_state, done):
        event = Event(state, action, reward, next_state, done)
        self.memory.add(event)
        self.learn()

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset() 

    def learn(self):
        """
        Update both actor and critic networks
        """
        event_batch = self.memory.sample(self.batch_size)
        
        if event_batch is None:
            return

        event_batch = self.memory.deserialize(event_batch)
        self.update_critic(event_batch)
        self.update_actor(event_batch)
        self.update_target(self.local_actor, self.target_actor, self.tau)
        self.update_target(self.local_critic, self.target_critic, self.tau)

    def update_critic(self, event_batch):
        states, actions, rewards, states_next, dones = zip(event_batch)
        print('states:{}'.format(states))
        ## TD step
        actions_pred = self.target_actor(states_next)
        critic_Q =  self.local_critic(states, actions)
        critic_target_Q = self.target_critic(states_next, actions_pred)
        #only learning from positives? negatives are good source of learning too
        critic_target_Q = rewards + (self.gamma*critic_target_q*(1-dones))
        loss = nn.functional.mse_loss(critic_Q, critic_target_Q)

        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()

    def update_actor(self, event_batch):
        states, actions, rewards, states_next, dones = zip(event_batch) 
        actions_predicted = self.local_actor(states)#fixthis
        loss = -self.local_critic(states, actions_predicted).mean() #rms

        self.optim_actor.zero_grad()
        loss.backward()
        self.optim_actor.step()

    def update_target(self, local, target, tau):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target.data.copy_(tau*local_param + (1.0-tau)*target.data)


if __name__ == '__main__':
    ddpg_agent = DDPGAgent(30, n_state=30, n_action=5)
    

        
