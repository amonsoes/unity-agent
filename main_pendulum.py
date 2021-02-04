import argparse
import gym

from collections import namedtuple
from ppo_pendulum import agent_pendulum as a

def main(n_episodes,
         dev_episodes,
         batch_size, 
         gamma, 
         n_epochs, 
         alpha, 
         beta, 
         policy_clip, 
         ac_dim,
         N,
         max_grad_norm,
         show):

    env = gym.make('Pendulum-v0').unwrapped
    env.reset()
    env.seed(0)
    print('env loaded')
    num_actions = env.action_space.shape[0]
    observ_dim = env.observation_space.shape[0]
    
    agent = a.Agent(n_actions=num_actions,
                gamma=gamma,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                input_dims=observ_dim,
                ac_dim=ac_dim,
                policy_clip=policy_clip,
                N=N,
                max_grad_norm=max_grad_norm)
    
    reward_run = -1000
    
    
    for e in range(n_episodes):
        score = 0
        state = env.reset()
        
        for _ in range(200):
            action, action_log_prob = agent.choose_action(state)
            state_, reward, _, _ = env.step([action]) # done flag not necessary for domain
            if show:
                env.render()
            agent.store(Trans(state, action, action_log_prob, (reward + 8) / 8, state_))
            if agent.full_memory():
                print('\n\n updating parameters... \n\n')
                agent.learn(n_epochs)
            score += reward
            state = state_
            
        reward_run = reward_run * 0.9 + score * 0.1 
        print(f"EPS {e}: {reward_run}")       
        if reward_run > -200 or e > 650:
            if reward_run > 200:
                print("\n\n ENV SOLVED \n\n")
            break
    
    dev_score = mean_dev_score(env, agent, dev_episodes)
    print('dev_score: ', dev_score)
    return dev_score

Trans = namedtuple('Trans', ['state', 'action', 'log_prob', 'reward', 'state_'])

def mean_dev_score(env, agent, dev_episodes):
    reward_run = -1000
    eps_scores = []
    for _ in range(dev_episodes):
        score = 0
        state = env.reset()
    
        for _ in range(200):
            action, _ = agent.choose_action(state)
            state_, reward, _, _ = env.step([action]) # done flag not necessary for domain
            score += reward
            state = state_
        
        reward_run = reward_run * 0.9 + score * 0.1 
        eps_scores.append(reward_run)
    return sum(eps_scores) / len(eps_scores)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--N', default=1024, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--n_episodes', default=1000,  type=int)
    parser.add_argument('--dev_episodes', default=50,  type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)
    parser.add_argument('--beta', default=3e-4, type=float)
    parser.add_argument('--policy_clip', default=0.2, type=float)
    parser.add_argument('--ac_dim', type=int, default=128)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--show', type=bool, default=False, help='render the environment')
    args = parser.parse_args()
    
    main(args.n_episodes,
         args.dev_episodes,
         args.batch_size,
         args.gamma,
         args.n_epochs,
         args.alpha,
         args.beta,
         args.policy_clip,
         args.ac_dim,
         args.N,
         args.max_grad_norm,
         args.show)
