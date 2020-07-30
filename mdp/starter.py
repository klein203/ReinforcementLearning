# import logging
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import mdp
from mdp.env import Maze2DEnv
from mdp.agent import InteractiveAgent


def interactive_agent_run():
    env = Maze2DEnv(config=mdp.default_maze_config)

    player = InteractiveAgent(env)
    player.play()
