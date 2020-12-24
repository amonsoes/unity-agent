import torch
import numpy as np

from torch.distributions import Normal
from a2c import FullyConnected

class TDA2CLearner:
    
    def __init__(self, gamma, nr_actions, alpha, beta, observation_dim, hidden_dim):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.eps = np.finfo(np.float32).eps.item() # for what?
        self.gamma = gamma
        self.nr_actions = nr_actions
        self.alpha = alpha
        self.beta = beta
        self.nr_input_features = observation_dim
        self.transitions = []
        self.actor = FullyConnected(observation_dim, hidden_dim, nr_actions, alpha)
        self.critic = FullyConnected(observation_dim, hidden_dim, 1, beta)
    
    def sample_action(self, state):
        mu, sigma = self.predict_policy([state])
        distribution = Normal(mu, torch.exp(sigma))
        distribution.sample(sample_shape= torch.size([self.nr_actions]))
        
    
    def update(self, state, action, reward, new_state, done):
        pass

if __name__ == '__main__':
    pass