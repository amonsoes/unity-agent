import argparse

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.bench import Monitor
from baselines import logger
import baselines.ppo2.ppo2 as ppo2

import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def make_unity_env(env_directory, num_env, visual, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank, use_visual=True): # pylint: disable=C0111
        def _thunk():
            unity_env = UnityEnvironment(env_directory)
            env = UnityToGymWrapper(unity_env, rank, uint8_visual=False)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])

def main(environment, nr_episodes):
    env = make_unity_env(environment, 1, True)
    ppo2.learn(
        network="mlp",
        env=env,
        total_timesteps=nr_episodes,
        lr=1e-3,
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=str, help='path to exec')
    parser.add_argument('--num_episodes', type=int, default=10000, help='set number of training episodes')
    args = parser.parse_args()
    
    main(args.executable, args.num_episodes)