import numpy as np

class QLearning:
    def __init__(self, n_actions, n_states, discount=0.9, alpha = 0.01, epsilon=0.9, min_epsilon = 0.1, decay = 1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.Q = np.zeros([n_states, n_actions])
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.chronology = []
        
    def update_epsilon(self):
#         if self.epsilon > self.min_epsilon: self.epsilon = 1 / (1 / self.epsilon + self.decay)
        if self.epsilon > self.min_epsilon: self.epsilon = self.epsilon * self.decay
            
    def act(self):
        if (np.random.uniform() < self.epsilon):
            return np.random.choice(self.n_actions)
        return np.argmax(self.Q[self.state, :])
    
    def update(self, action, reward, state, add_batch=True):
        if add_batch: self.chronology.append([self.state, action, reward, state])
        self.Q[self.state, action] += self.alpha * np.max(reward + self.discount * self.Q[state, :] - self.Q[self.state, action])        
#         self.alpha += 1 / ( self.alpha + self.decay)
        self.state = state

    def reset(self, state):
        self.state = state
        
    def play_batch(self, nb_batchs):
        prev_state = self.state
        arr = np.array(self.chronology)
        for s,a,r,sn in arr[np.random.choice(arr.shape[0], size=nb_batchs, replace=False), :]:
            # Reset position
            self.reset(s)
            # Update algorithm
            self.update(a, r, sn, add_batch = False)
        # Reset real state
        self.reset(prev_state)