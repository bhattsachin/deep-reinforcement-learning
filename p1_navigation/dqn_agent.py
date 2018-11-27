from model import QNetwork
import torch.optim as optim
from brain import Brain
from event import Event
import torch.nn.functional as F
import torch
import random
import numpy as np
import logging

RECALL_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256 # minibatch
GAMMA = 0.98 # discount
TAU = 1e-3
LR = 1e-4
HOURS_DAILY = 4 # pace of the environment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('bananagent')
log_handler = logging.FileHandler('banana.log')
logger.addHandler(log_handler)

class Agent():
    """ Our actor for the given role """

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        self.network_local = QNetwork(state_size, action_size, seed).to(device)
        self.network_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)
        self.brain = Brain(action_size, RECALL_BUFFER_SIZE, BATCH_SIZE, seed)
        self.time = 0 # awake hour of the agent
        logger.info('Agent init: state:{}, action:{}'.format(state_size, action_size))

    def step(self, event):
        self.brain.experience(event)
        self.time = self.time + 1
        self.time = self.time % HOURS_DAILY # switch to new day



        if self.time == 0:
            if len(self.brain) > BATCH_SIZE:
                experiences = self.brain.intuition()
                error = self.learn(experiences, GAMMA)



    def act(self, state, eps=0.):
        """ Actions as per current state """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            actions = self.network_local(state)
        
        self.network_local.train()

        if random.random() > eps:
            return np.argmax(actions.cpu().data.numpy())
        else:
            return random.choices(np.arange(self.action_size))

    # introspection
    def learn(self, experiences, gamma):
        # QLearning here
        Q_t_plus_1 = self.network_target(experiences.next_states()).detach().max(1)[0].unsqueeze(1)
        # Exploration vs using experience
        Q_t = gamma*Q_t_plus_1*(1-experiences.dones()) + experiences.rewards()

        Q_e = self.network_local(experiences.states()).gather(1, experiences.actions())

        # checkpoint - how far is our agent from desired behaviour as 
        # error unit
        
        delta = F.mse_loss(Q_t, Q_e)
        #logger.info('mse: {}'.format(delta))
        self.optimizer.zero_grad()
        delta.backward()
        self.optimizer.step() # evolutionary step - increase survival chances
        #logger.info('avg reward: {} mse:{}'.format(delta, np.mean(experiences.rewards())))

        self.rem_sleep(self.network_local, self.network_target, TAU)
        return delta


    def rem_sleep(self, reality, dream, tau):
        """ all our experiences become part of memory when we sleep"""
        for dream_param, reality_param in zip(dream.parameters(), reality.parameters()):
            dream_param.data.copy_(tau*reality_param.data + (1.0-tau)*dream_param.data)

