import torch as T
import torch.nn.functional as F


from torch import nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from .network_pendulum import ActorNetwork, CriticNetwork

        
class Agent:
    
    def __init__(self, n_actions, input_dims, ac_dim, gamma, alpha, beta, policy_clip, N, batch_size, max_grad_norm):
        self.memory = []
        self.counter = 0
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.actor = ActorNetwork(n_actions, input_dims, alpha, fc1_dims=ac_dim)
        self.critic = CriticNetwork(input_dims, beta, fc1_dims=ac_dim)
        self.N = N
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.eps = 1e-5
        
        # i put nr_steps, n_learning_iters also to agent, in order to clean main
        self.n_steps = 0
        self.learn_iters = 0

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def store(self, transition):
        self.memory.append(transition)
        self.counter += 1
        
    def full_memory(self):
        return self.counter % self.N == 0

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        with T.no_grad():
            
            dist = self.actor(state)
        action_vec = dist.sample()
        action_vec.clamp(-2.0, 2.0)
        log_probs = dist.log_prob(action_vec)

        return action_vec.item(), log_probs.item()
    
    def calc_advantage(self, states, rews, states_):
        with T.no_grad():
            t_vals = rews + self.gamma * self.critic(states_)
        advantage = (t_vals - self.critic(states)).detach()
        return advantage, t_vals
    
    def get_prob_ratio(self, states, actions, old_probs, index):
        dist = self.actor(states[index])
        action_log_probs = dist.log_prob(actions[index])
        prob_ratio = T.exp(action_log_probs - old_probs[index])
        return prob_ratio
        
    def learn(self, n_epochs):

        states = T.tensor([trans.state for trans in self.memory], dtype=T.float)
        states_ = T.tensor([trans.state_ for trans in self.memory], dtype=T.float)
        actions = T.tensor([trans.action for trans in self.memory], dtype=T.float).view(-1, 1)
        rewards = T.tensor([trans.reward for trans in self.memory], dtype=T.float).view(-1, 1)

        old_probs = T.tensor([trans.log_prob for trans in self.memory], dtype=T.float).view(-1, 1)

        norm_rews = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        advantage, t_vals = self.calc_advantage(states, norm_rews, states_)

        for _ in range(n_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.N)), self.batch_size, False):
                
                prob_ratio = self.get_prob_ratio(states, actions, old_probs, index)
                weighted_probs = prob_ratio * advantage[index]
                clipped_weightes_probs = T.clamp(prob_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantage[index]
                action_loss = -T.min(weighted_probs, clipped_weightes_probs).mean()

                self.actor.optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()

                value_loss = F.mse_loss(self.critic(states[index]), t_vals[index])
                self.critic.optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic.optimizer.step()
                
        self.learn_iters += 1
        del self.memory[:]