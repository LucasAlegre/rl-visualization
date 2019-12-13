import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import SAC

from rl_visualization.visualization_env import VisualizationEnv


if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')

    env = VisualizationEnv(env,
        steps_lookback=10000,
        refresh_time=30,
        features_names=['Car Position', 'Car Velocity'],
        actions_names=['Push car to the left (negative value) or to the right (positive value)']
    )

    model = SAC(MlpPolicy, env, verbose=1, action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.5 * np.ones(1)))
    model.learn(total_timesteps=60000)

    obs = env.reset()
    for i in range(100000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
    env.join()