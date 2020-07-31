import logging


class AbstractAgent(object):
    def __init__(self, env, *args, **kwargs):
        self.env = env

    def choose_action(self, *args, **kwargs):
        raise NotImplementedError()


class InteractiveAgent(AbstractAgent):
    def __init__(self, env, *args, **kwargs):
        super(InteractiveAgent, self).__init__(env, *args, **kwargs)
    
    def _play(self):
        obs = self.env.reset()
        while True:
            self.env.render()

            # fetch valid keypress from queue in env
            action = self.env.get_queue_action()
            if action == None:
                continue
            
            obs_, done = self.env.move_step(action)
            
            if done:
                obj = self.env.get_object(obs_)
                if obj == 'exit':
                    disp_state = 'WIN'
                else:
                    disp_state = 'BUSTED'
                logging.info('------------------%s----------------' % disp_state)
                
                self.env.render()
                break

            obs = obs_

        # self.env.destroy()

    def play(self):
        self.env.enable_manual_mode()
        self.env.after(100, self._play)
        self.env.mainloop()
