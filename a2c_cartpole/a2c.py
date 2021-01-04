import torch
import torch.nn.functional as F

from torch import nn

class FullyConnected(nn.Module):
    
    def __init__(self, observation_dim, hidden_dim, out_dim, lr):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.fc_in = nn.Linear(observation_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.to(self.device)
        
    def forward(self, obs):
        obs = F.relu(self.fc_in(obs))
        obs = F.relu(self.fc_hidden(obs))
        return self.fc_out(obs)

if __name__ == '__main__':
    pass
    
    
        