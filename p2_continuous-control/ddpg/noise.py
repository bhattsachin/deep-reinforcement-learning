import numpy as np

class OUNoise:
    def __init__(self, dimension, mu=0, theta=0.1, sigma=0.1, seed=30):
        self.dimension = dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.dimension) * self.mu 
        np.random.seed(seed)

    def reset(self):
        self.state = np.ones(self.dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + np.random.randn(len(x))       
        self.state = x + dx
        return self.state



if __name__ == '__main__':
    noise = OUNoise(4)
    print(noise.noise())
    


    