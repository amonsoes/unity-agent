import gym
import argparse
import matplotlib.pyplot as plot

from agent import TDA2CLearner

def episode(env, agent, nr_episode):
    state = env.reset()
    undiscounted_return = 0
    done = False
    time_step = 0
    while not done:
        env.render()
        action = agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    return undiscounted_return

if __name__ == 'main':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='specify the name of the env')
    parser.add_argument('--gamma', type=float, default=0.99, help='set the future rewards decay')
    parser.add_argument('--alpha', type=float, default=0.001, help='lr for actor')
    parser.add_argument('--beta', type=float, default=0.001, help='lr for critic')
    parser.add_argument('--training_iter', type=int, default=2000, help='set the number of training episodes for agent')
    parser.add_argument('--hidden_dim', type=int, default=600, help='set the dimension of the hidden layers in the a2c')
    args = parser.parse_args()

    env = gym.make(args.env)
    nr_actions = env.action_space.n
    observation_dim = env.observation_space.shape[0]
    
    agent = TDA2CLearner(args.gamma, nr_actions, args.alpha, args.beta, observation_dim, args.hidden_dim)
    returns = [episode(env, agent, i) for i in range(args.training_iter)]

    x = range(args.training_iter)
    y = returns

    plot.plot(x,y)
    plot.title("Progress")
    plot.xlabel("episode")
    plot.ylabel("undiscounted return")
    plot.show()
