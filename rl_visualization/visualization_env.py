import io
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process
from rl_visualization.app import start_app


class VisualizationEnv(gym.Wrapper):

    def __init__(self, env):
        """Gym Env wrapper for visualization
        
        Args:
            env (gym.Env): Gym Env to be wrapped
        """
        super().__init__(env)
        self.env = env

        self.start_app()
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        
        # TODO:

        return next_state, reward, done, info

    def start_app(self):
        self.app_process = Process(target=start_app, args=(self,))
        self.app_process.start()

    def get_qtable_png(self):
        f, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(np.random.randn(10, 12), annot=True)

        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def join(self):
        self.app_process.join()
        