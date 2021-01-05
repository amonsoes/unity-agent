import torch

from torch.distributions import Normal, Categorical
from torch.nn import functional
from a2c_cartpole.a2c import FullyConnected


class TDA2CLearner:
    
    def __init__(self, gamma, nr_actions, nr_outputs, alpha, beta, observation_dim, hidden_dim):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.gamma = gamma
        self.nr_actions = nr_actions
        self.nr_outputs = nr_outputs
        self.alpha = alpha
        self.beta = beta
        self.nr_input_features = observation_dim
        self.transitions = []
        self.actor = FullyConnected(observation_dim, hidden_dim, 2, alpha)
        self.critic = FullyConnected(observation_dim, hidden_dim, 1, beta)
    
    
    def sample_action(self, state):
        """
        mu, sigma = self.actor(state)
        distribution = Normal(mu, torch.exp(sigma))
        probs = distribution.sample(sample_shape=torch.Size([self.nr_outputs]))
        action = torch.tanh(probs) # change this line for your problem
        """
        out = self.actor(torch.FloatTensor(state).to(self.device))
        dist = Categorical(functional.softmax(out))
        action = dist.sample()
        return action.item()
            
        
    def predict(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return functional.softmax(self.actor(states)), self.critic(states)
        
    # ==== Advantage A2C Algorithm ====
    
    def update(self, sarsdtuple):
        self.transitions.append(sarsdtuple)
        if sarsdtuple.done:
            states, actions, rewards, _, _ = zip(*self.transitions)
            rewards = rewards[::-1]
            normalized_returns = self.normalize_returns(rewards)
            actions = torch.LongTensor(actions).to(self.device)
            actor_probs, critic_vals = self.predict(states)
            loss = self.calculate_gradients(actor_probs, actions, critic_vals, rewards, normalized_returns)
            self.gradient_step(loss)
            self.transitions.clear()
        else:
            return None
    
    def normalize_returns(self, rewards):
        discounted_return, d_return_list = 0, []
        for reward in rewards:
            discounted_return = reward + self.gamma*discounted_return
            d_return_list.append(discounted_return)
        d_return_list.reverse()
        discounted_returns = torch.FloatTensor(d_return_list).to(self.device).detach()
        normalized_returns = (discounted_returns - discounted_returns.mean())
        normalized_returns /= discounted_returns.std()
        return normalized_returns
        
    def calculate_gradients(self, actor_probs, actions, critic_vals, rewards, norm_returns):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        actor_losses, critic_losses = [], []
        i = 0
        for probs, action, value, reward, disc_return in zip(actor_probs, actions, critic_vals, rewards, norm_returns):
            td_advantage = reward + critic_vals[min(i+1, len(rewards)-1)].item() - value.item()
            distribution = Categorical(probs)
            actor_losses.append(-distribution.log_prob(action) * td_advantage)
            #critic_losses.append(td_advantage**2)
            critic_losses.append(functional.smooth_l1_loss(value, torch.tensor([disc_return])))
            i += 1
        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
        return loss
        
    def gradient_step(self, loss):
        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
    
    # ======================
        
        
if __name__ == '__main__':
    pass