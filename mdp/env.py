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
    def __init__(self, states_space, actions_space, p_data, discount_factor=0.9):
        """
        p_df is a pd.DataFrame like structure with columns ['s', 'a', 's_', 'r', 'p']
        """
        self.states_space = states_space
        self.actions_space = actions_space
        self.p_df = pd.DataFrame(data=p_data, columns=['s', 'a', 's_', 'r', 'p']).astype({'s': object, 'a': object, 's_': object, 'r': float, 'p': float})
        self.gamma = discount_factor
    
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
    
    def p(self, s, a, s_, r=None):
        """
        p(s_, r | s, a); p(s_ | s, a)
        (s:object, a:object, s_:object, r:float) -> p:float
        """
        df = self.p_df
        filter_df = df[(df['s']==s) & (df['a']==a) & (df['s_']==s_)]
        if filter_df.empty:
            if s not in self.states_space:
                raise Exception('invalid s=%s' % s)
            if a not in self.actions_space:
                raise Exception('invalid a=%s' % a)
            if s_ not in self.states_space:
                raise Exception('invalid s_=%s' % s_)
            raise Exception('invalid p(%s %s|%s, %s)' % (s_, r, s, a))
        if r != None:
            filter_df = filter_df(filter_df['r']==r)
        
        return filter_df['p'].sum()
    
    def r(self, s, a, s_=None):
        """
        r(s, a, s_); r(s, a)
        (s:object, a:object, s_:object) -> r:float
        """
        df = self.p_df
        filter_df = df[(df['s']==s) & (df['a']==a)]
        # if filter_df.empty:
        #     if s not in self.states_space:
        #         raise Exception('invalid s=%s' % s)
        #     if a not in self.actions_space:
        #         raise Exception('invalid a=%s' % a)
        #     if s_ != None:
        #         if s_ not in self.states_space:
        #             raise Exception('invalid s_=%s' % s_)

        #     raise Exception('invalid r(%s, %s, %s)' % (s, a, s_))
        # if s_ != None:
        #     filter_df = filter_df(filter_df['s_']==s_)
        if s_ == None:
            return (filter_df['r'] * filter_df['p']).sum()
        filter_df[['s_', 'r']]
        self.p(s, a, s_, r)
        return filter_df['r'].values[0]

    def get_actions_prob(self, s):
        """
        s:object -> df:pd.DataFrame(['a', 'p'])
        """
        df = self.p_df
        filter_df = df[(df['s']==s)]
        return filter_df[['a', 'p']] 
    
    def get_next_state(self, s, a):
        """
        (s:object, a:object) -> s_:int
        """
        df = self.p_df
        filter_df = df[(df['s']==s) & (df['a']==a)]#[['s_', 'p']]
        return np.random.choice(filter_df['s_'], p=filter_df['p'])
    
    def is_terminal(self, s):
        """
        s:object -> b:bool
        """
        if s not in self.states_space:
            raise Exception('invalid s=%s' % s)

        df = self.p_df
        filter_df = df[(df['s']==s)]
        if filter_df.empty:
            return True
        else:
            return False
    
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
        key_bindings = config.get('key_bindings', None)
        self.key_bindings = self._init_key_bindings(key_bindings)
        self.origin = config.get('origin', (0, 0))

        scn_w_px, scn_h_px = self.winfo_screenwidth(), self.winfo_screenheight()
        self.win_w_px, self.win_h_px = self.maze_ncols * self.u_px, self.maze_nrows * self.u_px
        self.geometry('%dx%d+%d+%d' % (self.win_w_px, self.win_h_px, scn_w_px - self.win_w_px, scn_h_px - self.win_h_px))

        self.manual_mode = False
        self.msg = None

        # observation, default (0, 0) (from the very left/top position)
        self.obs = self.origin
        self.obs_canvas = None

        # init MDP
        actions_space = self._init_actions_space(key_bindings)
        states_space = self._init_states_space(self.maze_ncols, self.maze_nrows)
        p_data = self._init_p_data(config.get('p_matrix', None))

        self.mdp = MarkovDecisionProcess(states_space, actions_space, p_data)

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
    
    def _init_p_data(self, data):
        return data
    
    def _init_r_data(self, data):
        return data
    
    def reset(self):
        # reset observation
        self.obs = self.origin
        self.render()
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
        
        if self.msg != None:
            self.draw_message(canvas)
    
    def set_message(self, msg):
        self.msg = msg

    def draw_message(self, canvas):
        canvas.create_text(self.win_w_px//2, self.win_h_px//2, text=self.msg, font=('Arial', 30, 'bold'), fill='red')

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

    def move_step(self, a):
        self.obs, _, done = self.mdp.move_step(self.obs, a)
        return self.obs, done

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
