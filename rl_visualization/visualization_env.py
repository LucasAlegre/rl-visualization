import io
import os
from datetime import datetime
import gym
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from threading import Thread
from rl_visualization.app import start_app

sns.set(style='whitegrid')
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

        self.experiences.append((self.obs, action, reward, done))
        self.obs = next_obs

        return next_obs, reward, done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def start_app(self):
        self.app_process = Thread(target=start_app, args=(self,))
        self.app_process.start()

    def get_featuresdistribution(self):
        f, ax = plt.subplots(figsize=(10, 10))
        plt.title('Features Distribution')

        dim = len(self.experiences[0][0])

        d = []
        for exp in self.experiences[-1000:]:
            s = exp[0]
            d.append({'f'+str(i): s[i] for i in range(dim)})
        df = pd.DataFrame(d)

        for i in range(dim):
            plt.subplot(3, 4, i+1)
            sns.distplot(df['f'+str(i)], hist=True, color="b", kde_kws={"shade": True}).set(xlim=(0, 1))
        plt.tight_layout()

        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def get_qtable_png(self):
        if self.agent is not None and hasattr(self.agent, 'q_table') and len(self.experiences) > 0:
            f, ax = plt.subplots(figsize=(14, 8))
            plt.title('Q-table')

            df = self.q_table_to_df(num_rows=20)
            sns.heatmap(df, annot=True, fmt="g", cmap="PiYG", linewidths=.5, center=0.0)
            plt.tight_layout()

            bytes_image = io.BytesIO()
            plt.savefig(bytes_image, format='png')
            bytes_image.seek(0)
            return bytes_image

    def q_table_to_df(self, num_rows=20):
        df = []
        for exp in self.experiences[-num_rows:]:
            s = self.env.encode(exp[0])
            for i, q in enumerate(self.agent.q_table[s]):
                df.append({'state': str(self.env.radix_decode(s)), 'action': i, 'q': q})
        df = pd.DataFrame(df)
        df.drop_duplicates(subset=None, keep='first', inplace=True)
        df = df.pivot(index='state', columns='action', values='q')
        return df
    
    def join(self):
        self.app_process.join()