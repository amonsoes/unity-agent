import os
import time
import argparse

from stable_baselines3 import PPO


from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment as UE

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10000, help='set number of training episodes')
    args = parser.parse_args()
    
    env_name = "./crawler_mac.app"
    env = UE(file_name=env_name, seed=1, side_channels=[])
    env = UnityToGymWrapper(env)

    # Create log dir
    time_int = int(time.time())
    log_dir = "stable_results/basic_env_{}/".format(time_int)
    os.makedirs(log_dir, exist_ok=True)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    
    obs = env.reset
    for i in range(args.num_episodes):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()