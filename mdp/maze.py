import time
import tkinter as tk

"""
2D Maze Environment
"""

class Maze2DEnv(tk.Tk, object):
    def __init__(self, config, *args, **kwargs):
        super(Maze2DEnv, self).__init__(*args, **kwargs)
        
        self._name = config.get('name', 'Maze2D')
        self.title(self._name)
        
        # render frequency, default 0.5s for rendering interval
        self._refresh_interval = config.get('refresh_interval', 0.5)
        
        # unit width and height
        self._unit_width = config.get('unit_pixel', 35)
        self._unit_height = config.get('unit_pixel', 35)
        
        # 2D shape, eg: cols x rows: 5 x 4
        self._shape = config.get('shape', (5, 4))
        
        # space for actions
        self._action_space = ['left', 'right', 'up', 'down']
        self._n_actions = len(self._action_space)
        
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        window_width, window_height = self._shape[0] * self._unit_width, self._shape[1] * self._unit_height
        self.geometry('%dx%d+%d+%d' % (window_width, window_height, screen_width - window_width, screen_height- window_height))
        
        # init objects in maze
        self._maze_objects = config.get('maze_objects')

        # observation, default (0, 0) (from the very left/top position)
        self._origin = config.get('origin', (0, 0))
        self.obs = self._origin
        
        # draw maze
        self._draw_maze();
        
#         self.bind("<Left>", lambda e: self._move_obs(0))
#         self.bind("<Right>", lambda e: self._move_obs(1))
#         self.bind("<Up>", lambda e: self._move_obs(2))
#         self.bind("<Down>", lambda e: self._move_obs(3))
    
    def _draw_maze(self):
        # canvas
        self.canvas = tk.Canvas(self, bg='white',
                                width=self._shape[0] * self._unit_width,
                                height=self._shape[1] * self._unit_height)
        self.canvas.pack()
        self.redraw_all(self.canvas)
        
    def redraw_all(self, canvas):
        # clear all objects on the canvas
        canvas.delete("all")
        
        # draw grids
        for x in range(0, self._shape[0] * self._unit_width, self._unit_width):
            x0, y0, x1, y1 = x, 0, x, self._shape[1] * self._unit_height
            canvas.create_line(x0, y0, x1, y1)
        for y in range(0, self._shape[1] * self._unit_height, self._unit_height):
            x0, y0, x1, y1 = 0, y, self._shape[0] * self._unit_width, y
            canvas.create_line(x0, y0, x1, y1)
            
        # draw mines, black square
        # draw treasure, yellow square
        for key, value in self._maze_objects.items():
            color = 'black'
            if value == 'treasure':
                color = 'yellow'
            elif value == 'mine':
                color = 'black'
                
            canvas.create_rectangle(
                key[0] * self._unit_width + 2, key[1] * self._unit_height + 2,
                (key[0] + 1) * self._unit_width - 2, (key[1] + 1) * self._unit_height - 2,
                fill=color)

        # draw obs, grey square
        canvas.create_rectangle(
            self.obs[0] * self._unit_width + 2, self.obs[1] * self._unit_height + 2,
            (self.obs[0] + 1) * self._unit_width - 2, (self.obs[1] + 1) * self._unit_height - 2,
            fill='grey')
        
    def _conflict_check(self, obs):
        conflict = obs in self._maze_objects.keys()
        obj = self._maze_objects.get(obs, None)
        return conflict, obj
    
    def reset(self):
        # reset observation
        self.obs = self._origin
        self.render()
        return self.obs

    def step(self, action):
        # init
        obs_ = list(self.obs)
        reward = 0
        done = False
        
        # action move, 
        if action == 0:  # 'left'
            obs_[0] -= 1
            if obs_[0] < 0:
                obs_[0] = 0
        elif action == 1:  # 'right'
            obs_[0] += 1
            if obs_[0] > self._shape[0] - 1:
                obs_[0] = self._shape[0] - 1
        elif action == 2:  # 'up'
            obs_[1] -= 1
            if obs_[1] < 0:
                obs_[1] = 0
        elif action == 3:  # 'down'
            obs_[1] += 1
            if obs_[1] > self._shape[1] - 1:
                obs_[1] = self._shape[1] - 1
        else:
            raise Exception('invalid action code', action)

        # conflict check
        obs_ = tuple(obs_)
        conflict, obj = self._conflict_check(obs_)
        if conflict:
            if obj == 'treasure':
                obs_ = obj
                reward = 1
                done = True
            elif obj == 'mine':
                obs_ = obj
                reward = -1
                done = True
            else:
                raise Exception('invalid object', obj)
        
        self.obs = obs_
        
        return obs_, reward, done
        
    def render(self):
        time.sleep(self._refresh_interval)
        self.redraw_all(self.canvas)
        self.update()
    
    @property
    def n_actions(self):
        return self._n_actions
    
    @property
    def name(self):
        return self._name
