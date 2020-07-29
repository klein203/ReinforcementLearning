import logging
import queue


class AbstractAgent(object):
    def __init__(self, env, *args, **kwargs):
        self.env = env

    def choose_action(self, *args, **kwargs):
        raise NotImplementedError()


class InteractiveAgent(AbstractAgent):
    def __init__(self, env, *args, **kwargs):
        super(InteractiveAgent, self).__init__(env, *args, **kwargs)
        self.input_buffer = queue.Queue(1)
        self.env.bind_func('<KeyPress>', self.bind_func)

    def bind_func(self, event):
        a = self.choose_action(event)
        if action != None:
            self.input_buffer.put(a)

    def choose_action(self, event):
        action = None
        if event.keysym in ['Left', 'a', 'A']:
            action = 0
        elif event.keysym in ['Right', 'd', 'D']:
            action = 1
        elif event.keysym in ['Up', 'w', 'W']:
            action = 2
        elif event.keysym in ['Down', 's', 'S']:
            action = 3
        return action
    
    def play(self):
        s = self.env.reset()
        while True:
            self.env.render()

            # wait valid keypress
            if self.input_buffer.empty():
                continue
            else:
                a = self.input_buffer.get()
            
            s_, r, done = self.env.move_step(a)
            
            if done:
                obj = self.env.get_object(s_)
                if obj == 'treasure':
                    disp_state = 'WIN'
                else:
                    disp_state = 'BUSTED'
                logging.info('------------------%s----------------' % disp_state)
                
                self.env.render()
                break

            s = s_
        
        env.destroy()
