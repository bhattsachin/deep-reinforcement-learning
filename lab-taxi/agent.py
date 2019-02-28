import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.03
        self.gamma = 0.1
        self.decay = 0.999999
        self.epsilon = 1 
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state not in self.Q:
            return np.random.choice(self.nA)

        return np.random.choice(np.arange(self.nA), p = self.weights(state))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.epsilon = max(self.epsilon*self.decay, 0.05)
        w = self.weights(self.Q[next_state]) 
        x = self.Q[next_state]
        self.Q[state][action] += self.alpha*(reward + self.gamma*np.dot(x, w) - self.Q[state][action]) 

    def weights(self, Q_state):
        weights = np.ones(self.nA)*self.epsilon/self.nA
        g_action = np.argmax(Q_state)
        weights[g_action] += 1-self.epsilon
        return weights
