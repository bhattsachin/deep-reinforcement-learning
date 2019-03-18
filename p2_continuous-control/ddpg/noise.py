import numpy as np
import random

class OUNoise:
    def __init__(self, dimension, mu=0, theta=0.15, sigma=0.2, seed=30):
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
        dx = self.theta * (self.mu - x) + np.random.randn(len(x))       
        self.state = x + dx
        return self.state



if __name__ == '__main__':
    noise = OUNoise(4)
    print(noise.make())
    


    