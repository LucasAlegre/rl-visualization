# Reinforcement Learning Visualization

A Gym wrapper for visualizing reinforcement learning agents trainning in a flask web application.

![alt text](https://github.com/LucasAlegre/rl-visualization/blob/master/test/app.png)

## How to install
```
pip3 install -e .
```

## How to use

```
from rl_visualization.visualization_env import VisualizationEnv

env = YourEnv(...)  # Regular gym.Env
env = VisualizationEnv(env)  # Wrapp it! A flask web application will start at http://127.0.0.1:5000/

# Your trainning code

env.join()  # Add this line if you want to keep the flask app running after the trainning ends
```

An example with SumoEnvironment and Q-learning can be found [here](https://github.com/LucasAlegre/rl-visualization/blob/master/test/test.py).