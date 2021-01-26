import os
import torch as T
import torch.optim as optim

from torch import nn
from torch.distributions import Categorical, Normal


class ActorNetwork(nn.Module):
    def __init__(self, n_outs, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.fc1 = nn.Linear(input_dims, fc2_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu_out = nn.Linear(fc2_dims, n_outs)
        self.sigma_out = nn.Linear(fc2_dims, n_outs)
        # n_outs for crawler should be a 20x2 real valued matrix containing logits => (mu, sigma) for every agent param
        self.optimizer = optim.Adam(self.parameters(), lr=alpha) if alpha != 0.0 else optim.Adam(self.parameters())
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.fc1(state)
        state = self.relu(self.fc2(state))
        
        mu_vec = self.mu_out(state)
        sigma_vec = self.softplus(self.sigma_out(state))
        
        dist = Normal(mu_vec, sigma_vec)
        # sigma can't be a negative value
        return dist
    

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, beta, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta) if beta != 0.0 else optim.Adam(self.parameters())
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))