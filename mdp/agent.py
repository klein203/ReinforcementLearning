import logging
import itertools as iter
import numpy as np
import pandas as pd
import pickle


class AbstractAgent(object):
    def __init__(self, env, *args, **kwargs):
        self.env = env

    def choose_action(self, *args, **kwargs):
        raise NotImplementedError()


class InteractiveAgent(AbstractAgent):
    def __init__(self, env, *args, **kwargs):
        super(InteractiveAgent, self).__init__(env, *args, **kwargs)
    
    def _play(self):
        _ = self.env.reset()
        i_step = 0

        while True:
            self.env.render()

            
            action = self.env.fetch_action()
            if action == None:
                continue
            else:
                i_step += 1
            
            obs_, _, done = self.env.move_step(action)
            
            if done:
                msg = self.env.update_message(obs_)
                logging.info('Run for [%d] step(s), finally [%s] at [%s]' % (i_step, msg, obs_))
                self.env.render()
                break

        # self.env.destroy()

    def play(self):
        self.env.enable_manual_mode()
        self.env.after(100, self._play)
        self.env.mainloop()


class QLearningAgent(AbstractAgent):
    def __init__(self, env, *args, **kwargs):
        super(QLearningAgent, self).__init__(env, *args, **kwargs)
        self.silent_mode = False
        self.q_df = pd.DataFrame(
            data=iter.product(self.env.states_space, self.env.actions_space, [0.0]), 
            columns=['s', 'a', 'q']).astype({'s': object, 'a': object, 'q': float})
        self.discount_factor = 0.9 # gamma -> 1, farseer

        self.epsilon = 0.1 # e-greedy
        self.learning_rate = 0.1

    def choose_action(self, s):
        df = self.q_df
        if np.random.rand() >= self.epsilon:
            actions = self.env.get_actions(s)
            if df[(df['s']==s) & (df['a'].isin(actions))]['q'].max() == 0:
                idxs = df[(df['s']==s) & (df['q']==0)].index
                action = df.loc[np.random.choice(idxs), 'a']
            else:
                idxmax = df[(df['s']==s) & (df['a'].isin(actions))]['q'].argmax()
                action = df.loc[idxmax, 'a']
        else:
            action = np.random.choice(self.env.get_actions(s))
        
        return action

    def learn(self, s, a, r, s_):
        # self._check_state_available(obs_)
        
        df = self.q_df
        if self.env.is_terminal(s_):
            q_target = r
        else:
            q_target = r + self.discount_factor * df[(df['s']==s)]['q'].max()
        
        idx = df[(df['s']==s) & (df['a']==a)].index
        q_predict = df.loc[idx, 'q']
        df.loc[idx, 'q'] = q_predict + self.learning_rate * (q_target - q_predict)

    def _play(self, load_fname):
        self.silent_mode = False
        self.epsilon = 0.0 # maxQ, no random
        self.load_pickle('.'.join([load_fname, 'pickle']))
        
        obs = self.env.reset()
        i_step = 0
        while True:
            self.env.render()
            
            i_step += 1
            a = self.choose_action(obs)
            obs_, r, done = self.env.move_step(a)
            
            if done:
                msg = self.env.update_message(obs_)
                logging.info('Run for [%d] step(s), finally [%s] at [%s]' % (i_step, msg, obs_))
                self.env.render()
                break

            obs = obs_

    def play(self, load_fname):
        self.env.after(100, self._play, load_fname)
        self.env.mainloop()
    
    def _train(self, n_episode, n_step=1000, save_fname=None, load_fname=None):
        if load_fname != None:
            self.load_pickle('.'.join([load_fname, 'pickle']))

        for i_episode in range(n_episode):
            obs = self.env.reset()
            i_step = 0
            while True:
                # logging.info(self.q_df)
                if self.silent_mode == False:
                    self.env.render()
                
                i_step += 1
                a = self.choose_action(obs)
                obs_, r, done = self.env.move_step(a)
                self.learn(obs, a, r, obs_)
                
                if done:
                    msg = self.env.update_message(obs_)
                    logging.info('Episode [%d], run for [%d] step(s), finally [%s] at [%s]' % (i_episode+1, i_step, msg, obs_))
                    
                    if self.silent_mode == False:
                        self.env.render()
                    break

                obs = obs_
        
        if save_fname != None:
            self.save_pickle('.'.join([save_fname, 'pickle']))
            self.save_csv('.'.join([save_fname, 'csv']))

    def train(self, n_episode=100, n_step=1000, save_fname=None, load_fname=None):
        """
        train agent
        """
        self.env.after(100, self._train, n_episode, n_step, save_fname, load_fname)
        self.env.mainloop()

    def save_csv(self, filename):
        """
        only for easy checking (column original format lost), save well-trained params to file in csv format
        """
        self.q_df.to_csv(filename)
        logging.debug('save params to file %s:\n%s' % (filename, self.q_df))

    def save_pickle(self, filename):
        """
        save well-trained params to file in pickle format
        """
        self.q_df.to_pickle(filename)
        logging.debug('save params to file %s:\n%s' % (filename, self.q_df))

    def load_pickle(self, filename):
        """
        load params from file in pickle format
        """
        self.q_df = pd.read_pickle(filename)
        logging.debug('load params from file %s:\n%s' % (filename, self.q_df))
