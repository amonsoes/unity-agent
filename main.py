import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper
import os.path

def episode(env, agent, nr_episode):
    n_games=nr_episode
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
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
        
    return score_history

def ppo_main(agent, room, training_iter, dev_iter):
    returns = episode(room.env, agent, training_iter)
    evaluation = sum(episode(room.env, agent, dev_iter)) / dev_iter
    return returns, evaluation


if __name__ == '__main__':
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')
        os.mkdir('tmp/ppo')
    elif os.path.isdir('tmp/ppo')==False:
        os.mkdir('tmp/ppo')
    if os.path.isdir('plots')==False:
        os.mkdir('plots')

    env = UE(file_name='UnityEnvironment', seed=1, side_channels=[])
   

    env.reset()
    
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    actions = len(spec.action_spec)

    print("Number of observations : ", len(spec.observation_shapes))
    print("Observation vector shape: ", spec.observation_shapes)

    inputdims = [x[0] for x in spec.observation_shapes]

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    beta=0.0003
    agent = Agent(n_actions= actions, batch_size=batch_size,
                    alpha=alpha,beta=beta,n_epochs=n_epochs,
                    input_dims=inputdims)
    n_games = 300

    #figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):

        decision_steps, terminal_steps = env.get_steps(behavior_name)
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