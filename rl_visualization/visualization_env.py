import io
import os
import time
import gym
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from threading import Thread
from rl_visualization.app import start_app
from math import sqrt, ceil

sns.set(style='whitegrid')
sns.set_context('paper')


class VisualizationEnv(gym.Wrapper):

    def __init__(self, env, agent=None, steps_lookback=1000, episodic=True, features_names=None, actions_names=None, refresh_time=20, path='./logs'):
        """Gym Env wrapper for visualization
        
        Args:
            env (gym.Env): Gym Env to be wrapped
        """
        super().__init__(env)
        self.env = env
        
        self.agent = agent
        self.steps_lookback = steps_lookback
        self.episodic = episodic

        self.user_plots = {}
        self.user_plots_values = {}

        if isinstance(self.observation_space, gym.spaces.Discrete):
            self.state_dim = self.observation_space.n
        elif isinstance(self.observation_space, gym.spaces.Box):
            self.state_dim = self.observation_space.shape[0]
        else:
            exit('Observation space not supported.')
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            exit('Action space not supported')
        if features_names is not None:
            self.features_names = features_names
        else:
            self.features_names = ['feature_'+str(i) for i in range(self.state_dim)]
        if actions_names is not None:
            self.actions_names = actions_names
        else:
            self.actions_names = ['action'+str(i) for i in range(self.action_dim)]

        if not os.path.exists(path):
            os.mkdir(path)
        
        self.filepath = os.path.join(path, 'rl_vis' + str(datetime.now()).split('.')[0] + "." + 'csv')
        self.refresh_time = refresh_time
        self.delay = 0

        self.experiences = []
        self.epsilon = []
        self.sa_counter = Counter()
        self.obs = None

        self.start_app()
        
    def set_agent(self, agent):
        self.agent = agent

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)

        if self.delay > 0:
            time.sleep(self.delay)

        self.experiences.append((self.obs, action, reward, next_obs, done))

        if self.agent is not None and hasattr(self.agent, 'q_table'):
            self.sa_counter.update([(self.env.encode(self.obs), action)])

        self.obs = next_obs
        
        for plot in self.user_plots:
            self.user_plots_values[plot].append(self.user_plots[plot]())

        return next_obs, reward, done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def start_app(self):
        self.app_process = Thread(target=start_app, args=(self,))
        self.app_process.start()
    
    def get_available_plots(self):
        plots = []
        if len(self.experiences) == 0:
            return plots

        if self.agent is not None and hasattr(self.agent, 'q_table'):
            plots.append('Q-table')
            plots.append('Visit Count')
            self.q_table_to_df()

        plots.append('Rewards')
        if self.episodic:
            plots.append('Episode Rewards')

        plots.extend(['Features Distributions', 'Actions Distributions'])

        plots.extend(self.user_plots.keys())

        return plots

    def add_plot(self, title, get_func):
        self.user_plots[title] = get_func
        self.user_plots_values[title] = []

    def get_userplot(self, title):
        f, ax = plt.subplots(figsize=(14, 8))
        plt.title(title)
        plt.xlabel('step')
        plt.plot(self.user_plots_values[title])

        plt.tight_layout()
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        plt.close()
        bytes_image.seek(0)
        return bytes_image

    def get_featuresdistribution(self):
        f, ax = plt.subplots(figsize=(14, 8))
        plt.title('Features Distribution')

        d = []
        for exp in self.experiences[-self.steps_lookback:]:
            s = exp[0]
            d.append({self.features_names[i]: s[i] for i in range(self.state_dim)})
        df = pd.DataFrame(d)

        n = ceil(sqrt(self.state_dim))
        for i in range(self.state_dim):
            plt.subplot(1 if n == self.state_dim else n, n, i+1)
            sns.distplot(df[self.features_names[i]], hist=True, color="b", kde_kws={"shade": True})
        plt.tight_layout()

        return self.plot_to_bytes(plt)

    def get_actionsdistribution(self):
        f, ax = plt.subplots()
        plt.title('Actions Distribution')

        if not hasattr(self.experiences[0][1], '__len__'): # int, float or numpy.int
            d = []
            for exp in self.experiences[-self.steps_lookback:]:
                a = exp[1]
                d.append({'Action': self.actions_names[a]})
            df = pd.DataFrame(d)
            sns.catplot(x="Action", kind="count", data=df)
        else:
            d = []
            for exp in self.experiences[-self.steps_lookback:]:
                s = exp[1]
                d.append({self.actions_names[i]: s[i] for i in range(self.action_dim)})
            df = pd.DataFrame(d)

            n = ceil(sqrt(self.action_dim))
            for i in range(self.action_dim):
                plt.subplot(1 if n == self.state_dim else n, n, i+1)
                sns.distplot(df[self.actions_names[i]], hist=True, color="r", kde_kws={"shade": True})

        plt.tight_layout()

        return self.plot_to_bytes(plt)

    def get_qtable_png(self):
        f, ax = plt.subplots(figsize=(14, 8))
        plt.title('Q-table')

        df = self.df.pivot(index='State', columns='Action', values='q')
        sns.heatmap(df, annot=True, fmt="g", cmap="PiYG", linewidths=.5, center=0.0)
        plt.tight_layout()

        return self.plot_to_bytes(plt)
    
    def get_visitcount(self):
        f, ax = plt.subplots(figsize=(14, 8))
        plt.title('Visit Count')

        df = self.df.pivot(index='State', columns='Action', values='count')
        sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=.5)
        plt.tight_layout()

        return self.plot_to_bytes(plt)

    def get_rewards(self):
        f, ax = plt.subplots(figsize=(14, 8))
        plt.title('Rewards')
        plt.xlabel('step')
        plt.plot([exp[2] for exp in self.experiences], color='g')

        plt.tight_layout()
        return self.plot_to_bytes(plt)
    
    def get_episoderewards(self):
        f, ax = plt.subplots(figsize=(14, 8))
        plt.title('Episode Rewards')

        d = []
        ep_reward = 0
        for i in range(1, len(self.experiences)):
            if self.experiences[i][4]:
                d.append({'step': i, 'Episode Rewards': ep_reward})
                ep_reward = 0
            else:
                ep_reward += self.experiences[i][2]

        if len(d) > 0:
            sns.lineplot(x='step', y='Episode Rewards', data=pd.DataFrame(d), color='orange')
        else:
            return None

        plt.tight_layout()
        return self.plot_to_bytes(plt)

    def q_table_to_df(self, num_rows=20):
        df = []
        for exp in self.experiences[-num_rows:]:
            s = self.env.encode(exp[0])
            for i, q in enumerate(self.agent.q_table[s]):
                df.append({'State': str(self.env.radix_decode(s)), 'Action': self.actions_names[i], 'q': q, 'count': self.sa_counter[(s, i)]})
        df = pd.DataFrame(df)
        df.drop_duplicates(subset=None, keep='first', inplace=True)
        self.df = df

    def plot_to_bytes(self, plot):
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        plt.close()
        bytes_image.seek(0)
        return bytes_image
    
    def join(self):
        self.app_process.join()