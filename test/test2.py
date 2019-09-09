import gym
import gym_minigrid
from rl_visualization.visualization_env import VisualizationEnv

if __name__ == '__main__':

    env = gym.make('MiniGrid-Empty-5x5-v0')

    env = VisualizationEnv(env)

    while True:
        obs, r, d, _ = env.step(env.action_space.sample())
        env.render()