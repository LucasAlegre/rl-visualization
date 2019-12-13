import gym
from stable_baselines.deepq.policies import FeedForwardPolicy, MlpPolicy
from stable_baselines import DQN

from rl_visualization.visualization_env import VisualizationEnv

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64],
                                              layer_norm=True,
                                              feature_extraction="mlp")

if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    env = VisualizationEnv(env,
        steps_lookback=10000,
        episodic=True,
        features_names=['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity At Tip'],
        actions_names=['Push cart to the left', 'Push cart to the right']
    )

    model = DQN(CustomDQNPolicy, env, verbose=1, learning_rate=1e-3, exploration_fraction=0.1, exploration_final_eps=0.02, prioritized_replay=True)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
    env.envs[0].join()