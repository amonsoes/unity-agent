import numpy as np
import torch as T

from .network import ActorNetwork, CriticNetwork

class PPOMemory:
    
    def __init__(self, batch_size):
        self.states = [] #T.tensor(dtype=T.float32)
        self.probs = None #T.tensor(dtype=T.float32).detach() # tensor.detach()
        self.vals = None #T.tensor(dtype=T.float32) # -> do not detach
        self.actions = [] #T.tensor(dtype=T.float32)
        self.rewards = [] #T.tensor(dtype=T.float32)
        self.dones = [] #T.tensor(dtype=T.float32)
        
        self.batch_size = batch_size

    def generate_batches(self):
        self.vals = self.vals.T.squeeze(0)
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start[:-1]]
        batches.append(indices[batch_start[-1]:])

        return np.array(self.states), \
               np.array(self.actions), \
               self.probs, \
               self.vals, \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        if self.probs == None:
            self.states.append(state)
            self.probs = T.tensor([probs]).detach()
            self.vals = vals
            self.actions = action
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            self.states.append(state)
            self.actions = T.cat((self.actions, action), 0)
            self.probs = T.cat((self.probs, T.tensor([probs])), 0).detach()
            self.vals = T.cat((self.vals,vals), 0)
            self.rewards.append(reward)
            self.dones.append(done)

    def clear_memory(self):
        self.states = [] #T.tensor(dtype=T.float32)
        self.probs = None #T.tensor(dtype=T.float32).detach()
        self.vals = None #T.tensor(dtype=T.float32)
        self.actions = [] #T.tensor(dtype=T.float32)
        self.rewards = [] #T.tensor(dtype=T.float32)
        self.dones = [] #T.tensor(dtype=T.float32)
        
        
class Agent:
    
    def __init__(self, n_actions, input_dims, ac_dim, gamma, alpha, beta, gae_lambda, policy_clip, batch_size, n_epochs, entropy_bonus):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha, fc1_dims=ac_dim, fc2_dims=ac_dim)
        self.critic = CriticNetwork(n_actions, input_dims, beta, fc1_dims=ac_dim, fc2_dims=ac_dim)
        self.memory = PPOMemory(batch_size)
        self.entropy_bonus = entropy_bonus
        self.eps = np.finfo(np.float32).eps.item()
        
        # i put nr_steps, n_learning_iters also to agent, in order to clean main
        self.n_steps = 0
        self.learn_iters = 0

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        with T.no_grad():
            
            dist = self.actor(state)
            action_vec = dist.sample()
            
            critic_input = T.cat((state, action_vec), 1)
            value = self.critic(critic_input)
            log_probs = dist.log_prob(action_vec).squeeze()

        return action_vec, sum([i.item() for i in log_probs]), value
    
    def calc_advantage(self, size, rewards, dones, values):
        advantage = T.zeros(size, dtype=T.float32)
        last_gae_lam = 0
        for step in reversed(range(size-1)):
            next_non_terminal = T.tensor(1.0 - dones[step + 1])
            next_values = values[step + 1].detach()
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantage[step] = last_gae_lam
        return T.tensor(advantage).to(self.actor.device)
    
    def eval_actions(self, obs_batch, action_batch):
        dist = self.actor(obs_batch)
        critic_input = T.cat((action_batch, obs_batch), 1).type(T.float32)
        critic_value = self.critic(critic_input)
        log_probs = T.sum(dist.log_prob(action_batch), 1)
        entropy = dist.entropy()
        return critic_value, log_probs, entropy
    
    def predict(self, obs):
        dist = self.actor(obs)
        return dist.mean

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, values, reward_arr, dones_arr, batches = self.memory.generate_batches()
            """
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            """
            values = T.tensor(values).to(self.actor.device)
            advantage = self.calc_advantage(len(reward_arr), reward_arr, dones_arr, values)
            #self.returns = self.advantages + self.values
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                critic_value, new_probs, entropy = self.eval_actions(states, actions)
                advantage_batch = advantage[batch]
                advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
                
                prob_ratio = T.exp(new_probs - old_probs)
                weighted_probs = advantage_batch.detach() * prob_ratio
                weighted_clipped_probs = advantage_batch.detach() * T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                critic_loss = (advantage_batch**2).mean()
                # entropy bonus improves exploration
                if self.entropy_bonus:
                    entropy_beta = 0.0001;
                else:
                    entropy_beta=0;
                    
                entropy_loss = -T.mean(entropy)
                
                total_loss = actor_loss + entropy_beta*entropy_loss + critic_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()