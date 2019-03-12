class Event:
    def __init__(self, state, action, reward, state_next, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.state_next = state_next
        self.done = done
    
