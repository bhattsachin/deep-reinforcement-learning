import gym
from ddpg_agent import DDPGAgent
import numpy as np

def run():
    env = gym.make('Pendulum-v0')
    seed = 30 
    env.seed(seed)
    agent = DDPGAgent(seed=seed,
                    n_state = env.observation_space.shape[0],
                    n_action = env.action_space.shape[0])

    episodes_n = 1000
    steps_max = 1000 
    scores = []
    for i_episodes in range(1, episodes_n):
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
        print('\rEpisode {} - \t{}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
            i_episodes, done_step, np.mean(scores), score), end="")
    #print('scores: {}'.format(scores))

    
if __name__ == '__main__':
    run()
        
