import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from rl_visualization.visualization_env import VisualizationEnv

if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')

    env = VisualizationEnv(env,
        steps_lookback=10000,
        refresh_time=30,
        features_names=['Car Position', 'Car Velocity'],
        actions_names=['Push car to the left (negative value) or to the right (positive value)']
    )

    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=500000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
    env.envs[0].join()