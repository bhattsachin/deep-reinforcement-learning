from dqn_agent import Agent
from unityagents import UnityEnvironment
from collections import deque
import torch
import numpy as np
import random
from event import Event
import logging

logger = logging.getLogger('bananagent')
log_handler = logging.FileHandler('banana.log')
logger.addHandler(log_handler)

#eps_decay 0.997 - score 16 after 1080 episodes
#eps_decay 0.996 - score 13.72 @ 690 / decay 0.060

def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.996):
    """ DQN """
    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i in range(1, n_episodes+1):
        env_obj = env.reset(train_mode=True)[brain_name]
        state = env_obj.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_obj = env.step(action)[brain_name]
            state_next = env_obj.vector_observations[0]
            reward = env_obj.rewards[0]
            done = env_obj.local_done[0]
            
            event = Event(state, action, reward, state_next, done)
            agent.step(event)

            state = state_next

            score += reward
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        
        if i%10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tdecay:{:.3f}'.format(i, np.mean(scores_window), eps))
        if np.mean(scores_window)>=18.0:
            print('\rEnvironment solved in {:d} episodes'.format(i))
            torch.save(agent.network_local.state_dict(), 'checkpoint.pth')
            break
            
    return scores


def run():
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86", no_graphics=True)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    
    scores = dqn(env, agent)
    return scores


if __name__=='__main__':
    run()
    
            
    