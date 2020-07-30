import logging
import time
import tkinter as tk
import numpy as np
import pandas as pd
import itertools as iter
import queue


class MarkovDecisionProcess(object):
    """
    Markov Decision Process <S, A, P, R, γ>
    """
    def __init__(self, states_space, actions_space, p_df, r_df, discount_factor=0.9, init_state=None, terminal_states=None):
        """
        p_df is a pd.DataFrame like structure with columns ['s', 'a', 's_', 'p']
        r_df is a pd.DataFrame like structure with columns ['s', 'a', 's_', 'r']
        """
        self.states_space = states_space
        self.actions_space = actions_space
        self.p_df = pd.DataFrame(data=p_df, columns=['s', 'a', 's_', 'p']).astype({'s': int, 'a': int, 's_': int, 'p': float})
        self.r_df = pd.DataFrame(data=r_df, columns=['s', 'a', 's_', 'r']).astype({'s': int, 'a': int, 's_': int, 'r': float})
        self.gamma = discount_factor
        self.init_state = init_state
        self.terminal_states = terminal_states
    
    @property
    def states(self):
        return self.states_space
    
    @property
    def actions(self):
        return self.actions_space

    @property
    def discount_factor(self):
        return self.gamma

    @property
    def n_states(self):
        return len(self.states_space)

    @property
    def n_actions(self):
        return len(self.actions_space)
    
    def p(self, s, a, s_):
        df = self.p_df
        filter_df = df[(df['s']==s) & (df['a']==a) & (df['s_']==s_)]
        if filter_df.empty:
            if s < 0 or s >= self.n_states:
                raise Exception('invalid s=%s' % s)
            if a < 0 or a >= self.n_actions:
                raise Exception('invalid a=%s' % a)
            if s_ < 0 or s_ >= self.n_states:
                raise Exception('invalid s_=%s' % s_)

            raise Exception('invalid p(%s|%s, %s)=p(%s|%s, %s)' % (self.states_space[s_], self.states_space[s], self.actions_space[a], s_, s, a))
        return filter_df['p'].values[0]
    
    def r(self, s, a, s_):
        df = self.r_df
        filter_df = df[(df['s']==s) & (df['a']==a) & (df['s_']==s_)]
        if filter_df.empty:
            if s < 0 or s >= self.n_states:
                raise Exception('invalid s=%s' % s)
            if a < 0 or a >= self.n_actions:
                raise Exception('invalid a=%s' % a)
            if s_ < 0 or s_ >= self.n_states:
                raise Exception('invalid s_=%s' % s_)

            raise Exception('invalid r(%s|%s, %s)=r(%s|%s, %s)' % (self.states_space[s_], self.states_space[s], self.actions_space[a], s_, s, a))
        return filter_df['r'].values[0]

    def get_actions_probs(self, s):
        df = self.p_df
        filter_df = df[(df['s']==s)]
        return filter_df[['a', 'p']] 
    
    def get_next_state(self, s, a):
        df = self.p_df
        filter_df = df[(df['s']==s) & (df['a']==a)][['s_', 'p']]
        return np.random.choice(filter_df['s_'], p=filter_df['p'])
    
    def is_terminal(self, s):
        if s < 0 and s >= self.n_states:
            raise Exception('invalid s=%s' % s)
        return (s in self.terminal_states)

    # def get_available_actions(self, s):
    #     df = self.p_df
    #     filter_df = df[(df['s']==s)]
    #     return filter_df[['a', 's_', 'p']]
    
    def move_step(self, s, a):
        s_ = self.get_next_state(s, a)
        r = self.r(s, a, s_)
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
        self.origin = config.get('origin', (0, 0))

        scn_w_px, scn_h_px = self.winfo_screenwidth(), self.winfo_screenheight()
        win_w_px, win_h_px = self.maze_ncols * self.u_px, self.maze_nrows * self.u_px
        self.geometry('%dx%d+%d+%d' % (win_w_px, win_h_px, scn_w_px - win_w_px, scn_h_px - win_h_px))

        # observation, default (0, 0) (from the very left/top position)
        self.obs = self.origin

        self.manual_mode = False

        # init MDP
        actions_space = ['l', 'r', 'u', 'd']
        states_space = list(iter.product(range(self.maze_ncols), range(self.maze_nrows)))
        init_state = self._coord_to_s(self.origin)
        terminal_states = [self._coord_to_s(coord) for coord in self.maze_objects.keys()]
        p_data = config.get('transit_matrix', None)
        r_data = config.get('reward_matrix', None)

        self.mdp = MarkovDecisionProcess(states_space, actions_space, p_data, r_data, init_state=init_state, terminal_states=terminal_states)

        # draw maze
        self.canvas = tk.Canvas(self, bg='white',
                                width=self.maze_ncols * self.u_px,
                                height=self.maze_nrows * self.u_px)
        self.canvas.pack()
        self.redraw_all(self.canvas)
        self.update()
    
    def reset(self):
        # reset observation
        self.obs = self.origin
        self.render()
        return self.obs
    
    def redraw_obs(self, canvas):
        if self.obs_canvas != None:
            canvas.delete(self.obs_canvas)
        
        self.obs_canvas = canvas.create_rectangle(
            self.obs[0] * self.u_px + 2,
            self.obs[1] * self.u_px + 2,
            (self.obs[0] + 1) * self.u_px - 2,
            (self.obs[1] + 1) * self.u_px - 2,
            fill='grey')


    def redraw_all(self, canvas):
        # clear all objects on the canvas
        # canvas.delete("all")
        
        # draw grids
        for x in range(0, self.maze_ncols * self.u_px, self.u_px):
            x0, y0, x1, y1 = x, 0, x, self.maze_nrows * self.u_px
            canvas.create_line(x0, y0, x1, y1)
        for y in range(0, self.maze_nrows * self.u_px, self.u_px):
            x0, y0, x1, y1 = 0, y, self.maze_ncols * self.u_px, y
            canvas.create_line(x0, y0, x1, y1)
            
        # draw mines, black square
        # draw treasure, yellow square
        for pos, obj in self.maze_objects.items():
            color = 'black'
            if obj == 'treasure':
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

        # draw obs, grey square
        self.obs_canvas = canvas.create_rectangle(
            self.obs[0] * self.u_px + 2,
            self.obs[1] * self.u_px + 2,
            (self.obs[0] + 1) * self.u_px - 2,
            (self.obs[1] + 1) * self.u_px - 2,
            fill='grey')

    def render(self):
        time.sleep(self.win_refresh_interval)
        self.redraw_obs(self.canvas)
        self.update()
    
    def get_object(self, obs):
        return self.maze_objects.get(obs)

    def _coord_to_s(self, obs):
        return self.maze_ncols * obs[1] + obs[0]

    def _s_to_coord(self, s):
        obs = [s%self.maze_ncols, s//self.maze_ncols]
        return tuple(obs)
    
    def move_step(self, a):
        s_, _, done = self.mdp.move_step(self._coord_to_s(self.obs), a)
        self.obs = self._s_to_coord(s_)
        return self.obs, done

    def bind_handler(event, handler, *args, **kwargs):
        self.bind(event, handler, *args, **kwargs)

    def disable_manual_mode(self):
        self.manual_mode = False
        self.action_buffer = None
        self.unbind('<KeyPress>', self.keypress_handler)

    def enable_manual_mode(self):
        self.manual_mode = True
        self.action_buffer = queue.Queue(1)
        self.bind('<KeyPress>', self.keypress_handler)

    def keypress_handler(self, event):
        a = None
        if event.keysym in ['Left', 'a', 'A']:
            a = 0
        elif event.keysym in ['Right', 'd', 'D']:
            a = 1
        elif event.keysym in ['Up', 'w', 'W']:
            a = 2
        elif event.keysym in ['Down', 's', 'S']:
            a = 3

        if a != None:
            if not self.action_buffer.full():
                self.action_buffer.put_nowait(a)
            else:
                # ignore key pressed
                pass
    
    def get_queue_action(self):
        if self.manual_mode:
            if self.action_buffer.empty():
                return None
            else:
                return self.action_buffer.get_nowait()
        
        return None