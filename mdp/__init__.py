'''
@author: xusheng
'''

__all__ = ['alg', 'agent', 'env', 'policy', 'starter']

import numpy as np
np.random.seed(seed=7)

import itertools as iter


default_maze_config = {
    # exit: (4, 3)
    'title_name': 'Maze2D',
    'refresh_interval': 0.05,
    'unit_pixel': 35,
    'shape': (5, 4),
    'origin': (0, 0),
    'maze_objects': {
        (1, 1): 'mine',
        (1, 2): 'mine',
        (2, 1): 'mine',
        (3, 3): 'mine',
        (4, 1): 'mine',
        (4, 3): 'exit',  # exit
    },
    'maze_object_rewards': {
        'default': 0,
        'mine': -1,
        'exit': 1,
    },
    'actions': [
        'left', 
        'right', 
        'up', 
        'down',
    ],
    'key_bindings': {
        'left': ['Left', 'a', 'A'],
        'right': ['Right', 'd', 'D'],
        'up': ['Up', 'w', 'W'],
        'down': ['Down', 's', 'S'],
    },
}

gridworld_config = {
    'title_name': 'Grid World',
    'refresh_interval': 0.05,
    'unit_pixel': 35,
    'shape': (4, 4),
    'origin': (1, 1),
    'maze_objects': {
        (0, 0): 'exit',
        (3, 3): 'exit',  # exit
    },
    'maze_object_rewards': {
        'default': -1,
        'exit': 0,
    },
    'actions': [
        'left', 
        'right', 
        'up', 
        'down',
    ],
    'key_bindings': {
        'left': ['Left', 'a', 'A'],
        'right': ['Right', 'd', 'D'],
        'up': ['Up', 'w', 'W'],
        'down': ['Down', 's', 'S'],
    },
}

alpha = 0.8
beta = 0.7
r_research = 1.5
r_wait = 1.0
autocleaner_mdp_config = {
    'states_space': ['high', 'low'],
    'actions_space': ['search', 'wait', 'recharge'],
    'rewards_space': [-3, 0, r_wait, r_research],
    'p_matrix': [
        ['high', 'search', 'high', r_research, alpha],
        ['high', 'search', 'low', r_research, 1-alpha],
        ['low', 'search', 'high', -3, 1-beta],
        ['low', 'search', 'low', r_research, beta],
        ['high', 'wait', 'high', r_wait, 1],
        ['low', 'wait', 'low', r_wait, 1],
        ['low', 'recharge', 'high', 0, 1],
    ]
}


student_mdp_config = {
    # exit: 'S5'
    'states_space': ['S%d' % i for i in range(1, 6)],
    'actions_space': [
        'Study', 'Pub', 'Facebook', 'Quit', 'Sleep'
    ],
    'rewards_space': [
        -2, -1, 0, 1, 10
    ],
    'p_matrix': [
        ['S1', 'Facebook', 'S4', -1, 1],
        ['S1', 'Study', 'S2', -2, 1],
        ['S2', 'Study', 'S3', -2, 1],
        ['S2', 'Sleep', 'S5', 0, 1],
        ['S3', 'Pub', 'S1', 1, 0.2],
        ['S3', 'Pub', 'S2', 1, 0.4],
        ['S3', 'Pub', 'S3', 1, 0.4],
        ['S3', 'Study', 'S5', 10, 1],
        ['S4', 'Facebook', 'S4', -1, 1],
        ['S4', 'Quit', 'S1', 0, 1],
    ]
}

gridworld_mdp_config = {
    # exit: (0, 0), (3, 3)
    'states_space': [
        (x, y) for x, y in iter.product(range(4), range(4))
    ],
    'actions_space': [
        'left', 'right', 'up', 'down'
    ],
    'rewards_space': [
        -1, 0
    ],
    'p_matrix': [
        [(0, 0), 'left', (0, 0), 0, 0],
        [(0, 1), 'left', (0, 1), -1, 1],
        [(0, 2), 'left', (0, 2), -1, 1],
        [(0, 3), 'left', (0, 3), -1, 1],
        [(1, 0), 'left', (0, 0), 0, 1],
        [(1, 1), 'left', (0, 1), -1, 1],
        [(1, 2), 'left', (0, 2), -1, 1],
        [(1, 3), 'left', (0, 3), -1, 1],
        [(2, 0), 'left', (1, 0), -1, 1],
        [(2, 1), 'left', (1, 1), -1, 1],
        [(2, 2), 'left', (1, 2), -1, 1],
        [(2, 3), 'left', (1, 3), -1, 1],
        [(3, 0), 'left', (2, 0), -1, 1],
        [(3, 1), 'left', (2, 1), -1, 1],
        [(3, 2), 'left', (2, 2), -1, 1],
        [(3, 3), 'left', (2, 3), -1, 0],
        [(0, 0), 'right', (1, 0), -1, 0],
        [(0, 1), 'right', (1, 1), -1, 1],
        [(0, 2), 'right', (1, 2), -1, 1],
        [(0, 3), 'right', (1, 3), -1, 1],
        [(1, 0), 'right', (2, 0), -1, 1],
        [(1, 1), 'right', (2, 1), -1, 1],
        [(1, 2), 'right', (2, 2), -1, 1],
        [(1, 3), 'right', (2, 3), -1, 1],
        [(2, 0), 'right', (3, 0), -1, 1],
        [(2, 1), 'right', (3, 1), -1, 1],
        [(2, 2), 'right', (3, 2), -1, 1],
        [(2, 3), 'right', (3, 3), 0, 1],
        [(3, 0), 'right', (3, 0), -1, 1],
        [(3, 1), 'right', (3, 1), -1, 1],
        [(3, 2), 'right', (3, 2), -1, 1],
        [(3, 3), 'right', (3, 3), 0, 0],
        [(0, 0), 'up', (0, 0), 0, 0],
        [(0, 1), 'up', (0, 0), 0, 1],
        [(0, 2), 'up', (0, 1), -1, 1],
        [(0, 3), 'up', (0, 2), -1, 1],
        [(1, 0), 'up', (1, 0), -1, 1],
        [(1, 1), 'up', (1, 0), -1, 1],
        [(1, 2), 'up', (1, 1), -1, 1],
        [(1, 3), 'up', (1, 2), -1, 1],
        [(2, 0), 'up', (2, 0), -1, 1],
        [(2, 1), 'up', (2, 0), -1, 1],
        [(2, 2), 'up', (2, 1), -1, 1],
        [(2, 3), 'up', (2, 2), -1, 1],
        [(3, 0), 'up', (3, 0), -1, 1],
        [(3, 1), 'up', (3, 0), -1, 1],
        [(3, 2), 'up', (3, 1), -1, 1],
        [(3, 3), 'up', (3, 2), -1, 0],
        [(0, 0), 'down', (0, 1), -1, 0],
        [(0, 1), 'down', (0, 2), -1, 1],
        [(0, 2), 'down', (0, 3), -1, 1],
        [(0, 3), 'down', (0, 3), -1, 1],
        [(1, 0), 'down', (1, 1), -1, 1],
        [(1, 1), 'down', (1, 2), -1, 1],
        [(1, 2), 'down', (1, 3), -1, 1],
        [(1, 3), 'down', (1, 3), -1, 1],
        [(2, 0), 'down', (2, 1), -1, 1],
        [(2, 1), 'down', (2, 2), -1, 1],
        [(2, 2), 'down', (2, 3), -1, 1],
        [(2, 3), 'down', (2, 3), -1, 1],
        [(3, 0), 'down', (3, 1), -1, 1],
        [(3, 1), 'down', (3, 2), -1, 1],
        [(3, 2), 'down', (3, 3), 0, 1],
        [(3, 3), 'down', (3, 3), 0, 0],
    ]
}

