import argparse
import matplotlib.pyplot as plot

from agent import TDA2CLearner
from gym_room import GymRoom
from collections import namedtuple

SARSD = namedtuple('SARSD', ['state', 'action', 'reward', 'next_state', 'done'])

def episode(env, agent, nr_episode):
    state = env.reset()
    undiscounted_return = 0
    done = False
    time_step = 0
    while not done:
        env.render()
        action = agent.sample_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(SARSD(state, action, reward, next_state, done))
        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    return undiscounted_return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1', help='specify the name of the env')
    parser.add_argument('--gamma', type=float, default=0.99, help='set the future rewards decay')
    parser.add_argument('--nr_outputs', type=int, default=1,  help='set output dim for action sampling')
    parser.add_argument('--alpha', type=float, default=0.0005, help='lr for actor')
    parser.add_argument('--beta', type=float, default=0.0001, help='lr for critic')
    parser.add_argument('--training_iter', type=int, default=500, help='set the number of training episodes for agent')
    parser.add_argument('--hidden_dim', type=int, default=64, help='set the dimension of the hidden layers in the a2c')
    args = parser.parse_args()

    room = GymRoom(args.env)
    agent = TDA2CLearner(gamma=args.gamma, 
                         nr_actions=room.env.action_space.n,
                         nr_outputs= args.nr_outputs,
                         alpha=args.alpha, 
                         beta=args.beta, 
                         observation_dim=room.env.observation_space.shape[0], 
                         hidden_dim=args.hidden_dim)
    
    returns = [episode(room.env, agent, i) for i in range(args.training_iter)]

    x = range(args.training_iter)
    y = returns

    plot.plot(x,y)
    plot.title("Progress")
    plot.xlabel("episode")
    plot.ylabel("undiscounted return")
    plot.show()
