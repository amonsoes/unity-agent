import torch
import numpy as np

from torch.distributions import Normal
from torch.nn import functional
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
        distribution.sample(sample_shape=torch.size([self.nr_actions]))
        
    # ==== Advantage A2C Algorithm ====
    
    def update(self, sarsdtuple):
        self.transitions.append(sarsdtuple)
        if sarsdtuple.done:
            states, actions, rewards, _, _ = zip(*self.transitions)
            rewards = rewards[::-1]
            normalized_returns = self.normalize_returns(rewards)
            actions = torch.FloatTensor(actions).to(self.device)
            states = torch.FloatTensor(states).to(self.device)
            actor_probs, critic_vals = self.actor(states), self.critic(states)
            actor_loss, critic_loss = self.calculate_gradients(actor_probs, actions, critic_vals, rewards, normalized_returns)
            self.gradient_step(actor_loss, critic_loss)
        else:
            return None
    
    def normalize_returns(self, rewards):
        discounted_return, d_return_list = 0, []
        for reward in rewards:
            discounted_return = reward + self.gamma*discounted_return
            d_return_list.append(discounted_return)
        discounted_returns = torch.FloatTensor(d_return_list).to(self.device)
        normalized_returns = (discounted_returns - discounted_returns.mean())
        normalized_returns /= discounted_returns.std()
        return normalized_returns
        
    def calculate_gradients(self, actor_probs, actions, critic_vals, rewards, norm_returns):
        actor_losses, critic_losses = [], []
        i = 0
        for probs, action, value, reward, disc_return in zip(actor_probs, actions, critic_vals, rewards, norm_returns):
            td_advantage = reward + critic_vals[min(i+1, len(rewards)-1)].item() - value.item()
            distribution = Normal(probs)
            actor_losses.append(-distribution.log_prob(action) * td_advantage)
            critic_losses.append(functional.smooth_l1_loss(value, torch.tensor([disc_return])))
            i += 1
        actor_loss = torch.stack(actor_losses).sum()
        critic_loss = torch.stack(critic_losses).sum()
        return actor_loss, critic_loss
        
    def gradient_step(self, actor_loss, critic_loss):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
    
    # ======================
        

if __name__ == '__main__':
    pass