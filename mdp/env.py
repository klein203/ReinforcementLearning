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
        probs = self.prob(s).sum(axis=(1, 2))
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
        (s:object, a:object) -> s_:list<object>
        """
        probs = self.prob(s, a).sum(axis=1)
        return [s_ for p, s_ in zip(probs, self.states_space) if p > 0]

    def is_terminal(self, s):
        """
        s:object -> b:bool
        """
        probs = self.prob(s).sum()
        return probs == 0

    def move_step(self, s, a):
        s_ = self.get_s_(s, a)
        r = self.reward(s, a, s_)
        done = self.is_terminal(s_)
        return s_, r, done


class Maze2DEnv(tk.Tk, MarkovEnv):
    def __init__(self, config, *args, **kwargs):
        # tk init, graphic settings
        super(Maze2DEnv, self).__init__(*args, **kwargs)

        # unit grid pixel
        self.u_px = config.get('unit_pixel', 35)

        # render frequency, default 0.5s for rendering interval
        self.win_refresh_interval = config.get('refresh_interval', 0.5)
        
        # Maze2D init, shape, eg: cols x rows: 5 x 4
        self.maze_shape = config.get('shape', (5, 4))
        self.maze_ncols = self.maze_shape[0]
        self.maze_nrows = self.maze_shape[1]

        # windows
        self.win_title_name = config.get('title_name', 'Maze2D')
        self.title(self.win_title_name)

        scn_w_px, scn_h_px = self.winfo_screenwidth(), self.winfo_screenheight()
        self.win_w_px, self.win_h_px = self.maze_ncols * self.u_px, self.maze_nrows * self.u_px
        self.geometry('%dx%d+%d+%d' % (self.win_w_px, self.win_h_px, scn_w_px - self.win_w_px, scn_h_px - self.win_h_px))

        # entry, miners, exit in maze
        self.maze_objects = config.get('maze_objects')
        self.maze_object_rewards = config.get('maze_object_rewards')

        # controller init, actions & key bindings
        self.actions = config.get('actions')
        self._init_key_bindings(config)

        # obs, default (0, 0) (from the very left/top position)
        self.origin = config.get('origin', (0, 0))
        self.obs = self.origin
        self.obs_canvas = None

        # canvas msg
        self.msg = None
        self.msg_canvas = None

        # agent play mode
        self.manual_mode = False

        # init markov env
        self._init_markov_env(config)

        # init canvas, draw maze
        self.canvas = tk.Canvas(self, bg='white',
                                width=self.maze_ncols * self.u_px,
                                height=self.maze_nrows * self.u_px)
        self.canvas.pack()

        self.draw_all(self.canvas)
        self.update()
    
    def _init_markov_env(self, config):
        states_space = list(iter.product(range(self.maze_ncols), range(self.maze_nrows)))
        actions_space = self.actions

        rewards_space = list(self.maze_object_rewards.values())
        
        MarkovEnv.__init__(self, states_space, actions_space, rewards_space)
        self._init_probs_matrix()
        
    def _init_probs_matrix(self):
        for s in self.states_space:
            if s in self.maze_objects.keys():
                continue
            
            for a in self.actions_space:
                if a == 'left':
                    s_ = (max(s[0]-1, 0), s[1])
                elif a == 'right':
                    s_ = (min(s[0]+1, self.maze_ncols-1), s[1])
                elif a == 'up':
                    s_ = (s[0], max(s[1]-1, 0))
                elif a == 'down':
                    s_ = (s[0], min(s[1]+1, self.maze_nrows-1))
                
                r = self._get_maze_object_reward(s_)
                self.set_prob(s, a, s_, r, 1)

    
    def _get_maze_object_reward(self, obs):
        reward = self.maze_object_rewards.get('default')
        if obs in self.maze_objects.keys():
            obj = self._get_maze_object(obs)
            reward = self.maze_object_rewards.get(obj)
        return reward

    def _init_key_bindings(self, config):
        self.key_bindings = dict()
        for action, keys in config.get('key_bindings').items():
            for key in keys:
                self.key_bindings[key] = action

    def reset(self):
        # reset obs
        self.obs = self.origin

        # reset message
        self.msg = None
        return self.obs
        
    def move_step(self, s, a):
        s_ = self.get_s_(s, a)
        r = self.reward(s, a, s_)
        done = self.is_terminal(s_)
        self.obs = s_
        return s_, r, done

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
        obj = self._get_maze_object(obs)
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
    
    def _get_maze_object(self, obs):
        return self.maze_objects.get(obs)
    
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
