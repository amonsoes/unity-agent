import numpy as np
import argparse
import os


from ppo import agent as a
from utils import plot_learning_curve
from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

def main(environment,
         N, 
         batch_size, 
         gamma, 
         n_epochs, 
         alpha, 
         beta, 
         n_episodes, 
         gae_lambda, 
         policy_clip, 
         dev_episodes, 
         random_eps, 
         entropy_bonus):
    
    # action space for crawler - real valued vector with 20 parameters 
    # observation space - real valued vetor with 172 parameters
    
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')
        os.mkdir('tmp/ppo')
    elif os.path.isdir('tmp/ppo')==False:
        os.mkdir('tmp/ppo')
    if os.path.isdir('plots')==False:
        os.mkdir('plots')
    
    env = UE(file_name=environment, seed=1, side_channels=[])
    env = UnityToGymWrapper(env)
    env.reset()
    print('env loaded')
    num_actions = env.action_size
    observ_dim = env.observation_space.shape[0]
    env.score_history = []
    figure_file = 'plots/agent_vals.png'
    
    agent = a.Agent(n_actions=num_actions,
                gamma=gamma,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                n_epochs=n_epochs,
                input_dims=observ_dim,
                gae_lambda=gae_lambda,
                policy_clip=policy_clip,
                entropy_bonus=entropy_bonus)
    
    best_score = 0
    avg_score = 0
    
    for i in range(n_episodes):
        if random_eps:
            score = random_episode(env, num_actions)
            print(f'for {i}, score:{score}')
        else:
            agent.learn_iters = 0
            score, avg_score = episode(env, agent, N)
            env.score_history.append(score)
            avg_score = np.mean(env.score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', agent.n_steps, 'learning_steps', agent.learn_iters)
    
    x = [i+1 for i in range(len(env.score_history))]
    
    dev_evaluation = dev_evaluate(env, agent, N, dev_episodes)
    plot_learning_curve(x, env.score_history, figure_file)
    return dev_evaluation

def episode(env, agent, N):
    
    observation = env.reset()
    done = np.array([False])
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        action = np.clip(np.array(action), a_min=-1.0, a_max=1.0)
        #print(action)
        observation_, reward, done_, _ = env.step(action)
        agent.n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if agent.n_steps % N == 0:
            #_, _, val = agent.choose_action(observation)
            #agent.learn(last_done=done_, last_val=val)
            print('...learning...')
            agent.learn()
            agent.learn_iters += 1
        observation = observation_
        done = done_
    env.score_history.append(score)
    avg_score = np.mean(env.score_history[-100:])
    return score, avg_score

def random_episode(env, num_actions):
    done = False
    total = 0
    _ = env.reset()
    while not done:
        action = np.random.randn(num_actions, dtype=np.float32) 
        action = np.clip(action, -0.99, 0.99)                  
        _, reward, done, _ = env.step(action)                                      
        total += reward
    return total                               

def dev_evaluate(env, agent, N, dev_episodes):
    scores = []
    for _ in range(dev_episodes):
        score, _ = episode(env, agent, N)
        scores.append(score)
    return sum(scores) / dev_episodes
        
        
if __name__ == '__main__':
    
    # by default, alpha & beta will be set to 0.0
    # which will result in an empty lr pass to the adam
    # optimizer in both actor & critic, so it adjust
    # the lr automatically (do not optimize)
    
    # However if wished, lr can be set manually
    
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--N', default=2048, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--n_episodes', default=20000,  type=int)
    parser.add_argument('--alpha', default=0.0003, type=float)
    parser.add_argument('--beta', default=0.0003, type=float)
    parser.add_argument('--policy_clip', default=0.3, type=float)
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--dev_episodes', default=50, type=int)
    parser.add_argument('--random', type=lambda x: x=='True', default=False)
    parser.add_argument('--entropy_bonus', type=lambda x: x=='True', default=False)
    args = parser.parse_args()
    
    main(args.env,
         args.N,
         args.batch_size,
         args.gamma,
         args.n_epochs,
         args.alpha,
         args.beta,
         args.n_episodes,
         args.gae_lambda,
         args.policy_clip,
         args.dev_episodes,
         args.random,
         args.entropy_bonus)