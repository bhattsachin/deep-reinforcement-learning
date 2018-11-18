from event import Event
from past import Past

class Brain:
    """Class imitating memory function of brain"""


    def __init__(self, action_size, brain_area, recall_area, seed):
        """Initialize memory component of a brain.
        
        """
        self.action_size = action_size
        self.memory = deque(maxlen=brain_area)
        self.recall_area = recall_area


    def intuition(self):
        past = Past(random.sample(self.memory, k=self.recall_area))
        return past
        

    def experience(self, event):
        self.memory.append(event)
        

    def __len__(self):
        return len(self.memory)