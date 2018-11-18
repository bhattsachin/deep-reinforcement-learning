from model import QNetwork
import torch.optim as optim
from brain import Brain
from event import Event

RECALL_BUFFER_SIZE = int(1e5)
BATCH_SIZE = 65 # minibatch
GAMMA = 0.99 # discount
TAU = 1e-3
LR = 5e-4
HOURS_DAILY = 5 # pace of the environment

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

    def step(self, event):
        self.brain.experience(event)
        self.time = self.time + 1
        self.time = self.time % HOURS_DAILY # switch to new day


        if self.time == 0:
            if len(self.brain) > BATCH_SIZE:
                experiences = self.brain.intuition()
                self.learn(experiences, GAMMA)



    def act(self):
        pass

    def learn(self, experiences, gamma):
        # QLearning here
        Q_t_plus_1 = self.network_target(experiences.next_states).detach().max(1)[0].unsqueeze(1)
        # Exploration vs using experience
        Q_t = gamma*Q_t_plus_1*(1-experiences.dones) + experiences.rewards

        Q_e = self.network_local(experiences.states).gather(1, experiences.actions)

        # checkpoint - how far is our agent from desired behaviour as 
        # error unit

        delta = F.mse_loss(Q_t, Q_e)
        self.optimizer.zero_grad()
        delta.backward()
        self.optimizer.step() # evolutionary step - increase survival chances

        self.rem_sleep(self.network_local, self.network_target, TAU)


    def rem_sleep(self, reality, dream, tau):
        """ all our experiences become part of memory when we sleep"""
        for dream_param, reality_param in zip(dream.parameters(), reality.parameters()):
            dream.data.copy_(tau*reality_param.data + (1.0-tau)*dream_param.data)

