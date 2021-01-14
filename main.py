import gym
import numpy as np
import argparse
import os
from ppo_torch import Agent
from utils import plot_learning_curve

def main(env, N, batch_size, n_epochs, alpha, beta, n_episodes):
    env = gym.make(env)
    env.score_history = []
    figure_file = 'plots/cartpole.png'
    
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                alpha=alpha, beta=beta, n_epochs=n_epochs,
                input_dims=env.observation_space.shape)
    
    best_score = env.reward_range[0]
    learn_iters, avg_score, n_steps = 0, 0, 0
    
    for i in range(n_episodes):
        score, avg_score = episode(env, agent, N)
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    
    x = [i+1 for i in range(len(env.score_history))]
    plot_learning_curve(x, env.score_history, figure_file)

def episode(env, agent, N):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        agent.n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if agent.n_steps % N == 0:
            agent.learn()
            agent.learn_iters += 1
        observation = observation_
    env.score_history.append(score)
    avg_score = np.mean(env.score_history[-100:])
    return score, avg_score
    

if __name__ == '__main__':
    
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')
        os.mkdir('tmp/ppo')
    elif os.path.isdir('tmp/ppo')==False:
        os.mkdir('tmp/ppo')
    if os.path.isdir('plots')==False:
        os.mkdir('plots')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--N', default=5, type=int)
    parser.add_argument('--n_epochs', default=4, type=int)
    parser.add_argument('--n_episodes', default=300,  type=int)
    parser.add_argument('--alpha', default=0.0005, type=float)
    parser.add_argument('--beta', default=0.001, type=float)
    args = parser.parse_args()
    
    main(args.env, args.N, args.batch_size, args.n_epochs, args.alpha, args.beta, args.n_episodes)

    """
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=env.observation_space.shape)
    n_episodes = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    """