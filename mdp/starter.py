# import logging
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import mdp
from mdp.env import Maze2DEnv
from mdp.agent import InteractiveAgent


# # test run
# def start(env, agent):
#     obs = env.reset()
#     i_step = 0
#     while True:
#         env.render()
#         action = agent.choose_action(str(obs))
#         obs_, reward, done = env.step(action)
        
#         # skip learning phrase
#         obs = obs_
        
#         if done:
#             if reward > 0:
#                 disp_state = 'WIN'
#             else:
#                 disp_state = 'BUSTED'
            
#             break
    
#     env.destroy()

def test_human():
    env = Maze2DEnv(config=mdp.default_maze_config)
    env.mainloop()

    player = InteractiveAgent(env)
    player.play()
