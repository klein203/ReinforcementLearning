import logging
import time
import tkinter as tk
import numpy as np
import pandas as pd
import itertools as iter
import queue


class MarkovEnv(object):
    """
    Markov Decision Process <S, A, P, R, γ>
    """
    def __init__(self, states_space, actions_space, rewards_space, probs_matrix=None):
        self.states_space = states_space
        self.states_dict = {s: i for i, s in enumerate(states_space)}
        self.actions_space = actions_space
        self.actions_dict = {a: i for i, a in enumerate(actions_space)}
        self.rewards_space = rewards_space
        self.rewards_dict = {r: i for i, r in enumerate(rewards_space)}
        self.probs_matrix = np.zeros((self.n_states, self.n_actions, self.n_states, self.n_rewards))
        if probs_matrix != None:
            for item in probs_matrix:
                self.set_prob(item[0], item[1], item[2], item[3], item[4])
    
    @property
    def n_states(self):
        return len(self.states_space)

    @property
    def n_actions(self):
        return len(self.actions_space)

    @property
    def n_rewards(self):
        return len(self.rewards_space)
    
    def s(self, s):
        """
        state:object -> index:int
        """
        return self.states_dict.get(s, None)
    
    def a(self, a):
        """
        action:object -> index:int
        """
        return self.actions_dict.get(a, None)
    
    def r(self, r):
        """
        reward:object -> index:int
        """
        return self.rewards_dict.get(r, None)
    
    def set_prob(self, s, a, s_, r, p):
        self.probs_matrix[self.s(s), self.a(a), self.s(s_), self.r(r)] = p
    
    def prob(self, s, a=None, s_=None, r=None):
        """
        p(s_|s, a) = Σ[p(s_, r|s, a)]
        p(s_, r|s, a)
        (s:object, a:object, s_:object, r:object) -> p:float
        """
        if a == None:
            if s_ == None:
                if r == None:
                    return self.probs_matrix[self.s(s), :, :, :]
                else:
                    raise Exception('invalid')
            else:
                raise Exception('invalid')
        else:
            if s_ == None:
                if r == None:
                    return self.probs_matrix[self.s(s), self.a(a), :, :]
                else:
                    raise Exception('invalid')
            else:
                if r == None:
                    return self.probs_matrix[self.s(s), self.a(a), self.s(s_), :]
                else:
                    return self.probs_matrix[self.s(s), self.a(a), self.s(s_), self.r(r)]

    def reward(self, s, a, s_=None):
        """
        r(s, a) = Σ[r * Σ[p(s_, r|s, a)]]]
        r(s, a, s_) = Σ[r * p(s_, r|s, a)] / p(s_|s, a)
        (s:object, a:object, s_:object) -> r:float
        """
        r = None
        rewards = np.array(self.rewards_space)
        if s_ == None:
            probs = self.prob(s, a)
            r = (rewards * probs).sum(axis=1)
        else:
            probs = self.prob(s, a, s_)
            r = (rewards * probs).sum() / probs.sum()
        return r

    def get_actions(self, s):
        """
        get all available actions from s
        s:object -> a:list<object>
        """
        probs = self.probs_matrix(s).sum(axis=(1, 2))
        return [a for p, a in zip(probs, self.actions_space) if p > 0]
    
    def get_s_(self, s, a):
        """
        (s:object, a:object) -> s_:object
        """
        probs = self.prob(s, a).sum(axis=1)
        s_ = np.random.choice(range(self.n_states), p=probs)
        return self.states_space[s_]
    
    def get_states(self, s, a):
        """
        get all available states from s, a
        (s:object, a:object) -> s:list<object>
        """
        probs = self.prob(s, a).sum(axis=1)
        return [s_ for p, s_ in zip(probs, self.states_space) if p > 0]

    def is_terminal(self, s):
        """
        s:object -> b:bool
        """
        probs = self.probs_matrix[self.s(s), :, :, :].sum()
        return probs == 0

    def move_step(self, s, a):
        s_ = self.get_s_(s, a)
        r = self.reward(s, a)
        done = self.is_terminal(s_)
        return s_, r, done


class Maze2DEnv(tk.Tk):
    def __init__(self, config, *args, **kwargs):
        # tk init
        super(Maze2DEnv, self).__init__(*args, **kwargs)
        self.win_title_name = config.get('title_name', 'Maze2D')
        self.title(self.win_title_name)
        
        # render frequency, default 0.5s for rendering interval
        self.win_refresh_interval = config.get('refresh_interval', 0.5)
        
        # unit grid pixel
        self.u_px = config.get('unit_pixel', 35)
        
        # 2D shape, eg: cols x rows: 5 x 4
        self.maze_shape = config.get('shape', (5, 4))
        self.maze_ncols = self.maze_shape[0]
        self.maze_nrows = self.maze_shape[1]

        # entry, miners, exit in maze
        self.maze_objects = config.get('maze_objects', {(self.maze_ncols-1, self.maze_nrows-1): 'exit'})

        key_bindings = config.get('key_bindings', None)
        self.key_bindings = self._init_key_bindings(key_bindings)
        self.origin = config.get('origin', (0, 0))

        scn_w_px, scn_h_px = self.winfo_screenwidth(), self.winfo_screenheight()
        self.win_w_px, self.win_h_px = self.maze_ncols * self.u_px, self.maze_nrows * self.u_px
        self.geometry('%dx%d+%d+%d' % (self.win_w_px, self.win_h_px, scn_w_px - self.win_w_px, scn_h_px - self.win_h_px))

        self.manual_mode = False
        self.msg = None
        self.msg_canvas = None

        # observation, default (0, 0) (from the very left/top position)
        self.obs = self.origin
        self.obs_canvas = None

        # init MDP
        self.actions_space = self._init_actions_space(key_bindings)
        self.states_space = self._init_states_space(self.maze_ncols, self.maze_nrows)
        self.rewards_space = self._init_rewards_space(config.get('rewards_space'))

        self.mdp = MarkovEnv(self.states_space, self.actions_space, self.rewards_space)
        self._init_p_data(config.get('p_matrix', None))

        # draw maze
        self.canvas = tk.Canvas(self, bg='white',
                                width=self.maze_ncols * self.u_px,
                                height=self.maze_nrows * self.u_px)
        self.canvas.pack()

        self.draw_all(self.canvas)
        self.update()

    def _init_controllers(self, maze_controllers):
        list(maze_controllers.keys())

    def _init_key_bindings(self, key_bindings):
        bindings = dict()
        for action, keys in key_bindings.items():
            for key in keys:
                bindings[key] = action
        return bindings

    def _init_actions_space(self, key_bindings):
        return list(key_bindings.keys())

    def _init_states_space(self, ncols, nrows):
        return list(iter.product(range(ncols), range(nrows)))
    
    def _init_rewards_space(self, rewards_space):
        return rewards_space

    def _init_p_data(self, data):
        for d in data:
            self.mdp.set_prob(d[0], d[1], d[2], d[3], d[4])
    
    def _init_r_data(self, data):
        return data

    @property
    def n_states(self):
        return len(self.mdp.states_space)

    @property
    def n_actions(self):
        return len(self.mdp.actions_space)

    @property
    def n_rewards(self):
        return len(self.mdp.rewards_space)
    
    def s(self, s):
        return self.mdp.states_dict.get(s, None)
    
    def a(self, a):
        return self.mdp.actions_dict.get(a, None)
    
    def r(self, r):
        return self.mdp.rewards_dict.get(r, None)

    def reset(self):
        # reset observation
        self.obs = self.origin
        # reset message
        self.msg = None
        return self.obs
    
    def redraw_partial(self, canvas):
        if self.obs_canvas != None:
            canvas.delete(self.obs_canvas)
        
        self.obs_canvas = canvas.create_rectangle(
            self.obs[0] * self.u_px + 2,
            self.obs[1] * self.u_px + 2,
            (self.obs[0] + 1) * self.u_px - 2,
            (self.obs[1] + 1) * self.u_px - 2,
            fill='grey')
        
        if self.msg == None:
            if self.msg_canvas != None:
                canvas.delete(self.msg_canvas)
        else:
            self.msg_canvas = canvas.create_text(
                self.win_w_px//2, self.win_h_px//2,
                text=self.msg,
                font=('Arial', 30, 'bold'),
                fill='red')
    
    def update_message(self, obs):
        obj = self.get_object(obs)
        if obj == 'exit':
            self.msg = 'WIN'
        else:
            self.msg = 'BUSTED'
        return self.msg

    def draw_all(self, canvas):
        # draw grids
        for x in range(0, self.maze_ncols * self.u_px, self.u_px):
            x0, y0, x1, y1 = x, 0, x, self.maze_nrows * self.u_px
            canvas.create_line(x0, y0, x1, y1)
        for y in range(0, self.maze_nrows * self.u_px, self.u_px):
            x0, y0, x1, y1 = 0, y, self.maze_ncols * self.u_px, y
            canvas.create_line(x0, y0, x1, y1)
            
        # draw mines, black square
        # draw exit, yellow square
        for pos, obj in self.maze_objects.items():
            color = 'black'
            if obj == 'exit':
                color = 'yellow'
            elif obj == 'mine':
                color = 'black'
            else:
                raise Exception('invalid maze object')
                
            canvas.create_rectangle(
                pos[0] * self.u_px + 2, 
                pos[1] * self.u_px + 2,
                (pos[0] + 1) * self.u_px - 2,
                (pos[1] + 1) * self.u_px - 2,
                fill=color)

        # draw obs, grey square; msg
        self.redraw_partial(canvas)

    def render(self):
        time.sleep(self.win_refresh_interval)
        self.redraw_partial(self.canvas)
        self.update()
    
    def get_object(self, obs):
        return self.maze_objects.get(obs)

    def move_step(self, obs, a):
        self.obs, r, done = self.mdp.move_step(obs, a)
        return self.obs, r, done
    
    def is_terminal(self, obs):
        return self.mdp.is_terminal(obs)

    def get_actions(self, obs):
        return self.mdp.get_actions(obs)
    
    def disable_manual_mode(self):
        self.manual_mode = False
        self.action_buffer = None
        self.unbind('<KeyPress>', self.keypress_handler)

    def enable_manual_mode(self):
        self.manual_mode = True
        self.action_buffer = queue.Queue(1)
        self.bind('<KeyPress>', self.keypress_handler)

    def keypress_handler(self, event):
        action = self.key_bindings.get(event.keysym)
        if action != None:
            if not self.action_buffer.full():
                self.action_buffer.put_nowait(action)
        else:
            # ignore invalid key pressed
            pass
    
    def fetch_action(self):
        if self.manual_mode:
            if self.action_buffer.empty():
                return None
            else:
                return self.action_buffer.get_nowait()
        else:
            return None
