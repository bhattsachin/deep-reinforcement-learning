import numpy as np
import random

class OUNoise:
    def __init__(self, dimension, mu=0, theta=0.30, sigma=0.1, seed=30):
        self.dimension = dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.dimension) * self.mu 
        self.seed = np.random.seed(seed)

    def reset(self):
        self.state = np.ones(self.dimension) * self.mu

    def make(self):
        x = self.state
        e = np.array([random.random() for i in range(len(self.state))])
        dx = self.theta * (self.mu - x) + self.sigma * e       
        self.state = x + dx
        return self.state



if __name__ == '__main__':
    noise = OUNoise(4)
    print(noise.make())
    


    