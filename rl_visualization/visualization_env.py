import io
import os
from datetime import datetime
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from threading import Thread
from rl_visualization.app import start_app

sns.set(style='darkgrid')
sns.set_context('talk')


class VisualizationEnv(gym.Wrapper):

    def __init__(self, env, agent=None, path='./logs'):
        """Gym Env wrapper for visualization
        
        Args:
            env (gym.Env): Gym Env to be wrapped
        """
        super().__init__(env)
        self.env = env
        self.agent = agent

        if not os.path.exists(path):
            os.mkdir(path)
        
        self.filepath = os.path.join(path, 'rl_vis' + str(datetime.now()).split('.')[0] + "." + 'csv')
        self.experiences = []
        self.obs = None

        self.start_app()
        
    def set_agent(self, agent):
        self.agent = agent

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        
        if hasattr(self.env, 'discrete_observation_space'):
            exp = {'obs': self.env.encode(self.obs)}
            exp.update({'next_obs': self.env.encode(next_obs)})
            exp.update({'action': action, 'reward': reward, 'done': done})
            self.experiences.append(exp)
            self.obs = next_obs

        return next_obs, reward, done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def start_app(self):
        self.app_process = Thread(target=start_app, args=(self,))
        self.app_process.start()

    def get_qtable_png(self):
        if self.agent is not None and hasattr(self.agent, 'q_table') and len(self.experiences) > 0:
            plt.figure(figsize=(30,20))
            plt.title('Q-table')

            df = self.q_table_to_df(num_rows=20)
            sns.heatmap(df, annot=True, fmt="g", cmap="PiYG", linewidths=.5)
            plt.tight_layout()

            bytes_image = io.BytesIO()
            plt.savefig(bytes_image, format='png')
            bytes_image.seek(0)
            return bytes_image

    def join(self):
        self.app_process.join()

    def q_table_to_df(self, num_rows=20):
        df = []
        for exp in self.experiences[-num_rows:]:
            s = exp['obs']
            for i, q in enumerate(self.agent.q_table[s]):
                df.append({'state': str(self.env.radix_decode(s)), 'action': i, 'q': q})
        df = pd.DataFrame(df)
        df.drop_duplicates(subset=None, keep='first', inplace=True)
        df = df.pivot(index='state', columns='action', values='q')
        return df