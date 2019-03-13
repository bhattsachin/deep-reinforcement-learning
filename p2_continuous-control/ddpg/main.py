import gym
from ddpg_agent import DDPGAgent

def run():
    env = gym.make('BipedalWalker-v2')
    seed = 30 
    env.seed(seed)
    agent = DDPGAgent(seed=seed,
                    n_state = env.observation_space.shape[0],
                    n_action = env.action_space.shape[0])

    episodes_n = 3 
    steps_max = 500 
    scores = []
    for i_episodes in range(1, episodes_n):
        state = env.reset()
        agent.reset()
        score = 0
        for step in range(steps_max):
            action = agent.act(state)
            state_next, reward, done, meta = env.step(action)
            agent.step(state, action, reward, state_next, done)
            state = state_next
            score += reward
            if done:
                break
        scores.append(score)
    print('scores: {}'.format(scores))

    
if __name__ == '__main__':
    run()
        
