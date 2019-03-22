import gym
from ddpg_agent import DDPGAgent
from udacity_agent import Agent
import numpy as np
from collections import deque

def run():
    env = gym.make('Pendulum-v0')
    seed = 30 
    env.seed(seed)

    
    agent = DDPGAgent(seed=seed,
                    n_state = env.observation_space.shape[0],
                    n_action = env.action_space.shape[0])
    
    ''' 
    agent = Agent(state_size=env.observation_space.shape[0], 
                  action_size=env.action_space.shape[0], random_seed=seed)
    ''' 
     
    episodes_n = 2000
    steps_max = 300 
    scores = []
    print_every = 100
    
    scores_deque = deque(maxlen=print_every)

    for i_episode in range(1, episodes_n):
        state = env.reset()
        agent.reset()
        score = 0
        done_step = 0
        for step in range(steps_max):
            action = agent.act(state)
            state_next, reward, done, meta = env.step(action)
            agent.step(state, action, reward, state_next, done)
            state = state_next
            score += reward
            done_step += 1
            if done:
                break
        scores.append(score)
        scores_deque.append(score)

        print_line(i_episode, scores_deque, end="")
        if i_episode % print_every == 0:
            print_line(i_episode, scores_deque, end="\n")


       
    return scores

def print_line(i_episode, scores_deque, end=""):
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end=end)

    
if __name__ == '__main__':
    run()
        
