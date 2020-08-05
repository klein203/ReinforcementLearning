import logging
import itertools as iter
import numpy as np
import pandas as pd
import pickle


class AbstractAgent(object):
    def __init__(self, env, *args, **kwargs):
        self.env = env

    def play(self, *args, **kwargs):
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


class AIAgent(AbstractAgent):
    def __init__(self, env, *args, **kwargs):
        super(AIAgent, self).__init__(env, *args, **kwargs)

    def choose_action(self, *args, **kwargs):
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def save_csv(self, filename):
        raise NotImplementedError()

    def save_pickle(self, filename):
        raise NotImplementedError()

    def load_pickle(self, filename):
        raise NotImplementedError()


class QTableBasedAgent(AIAgent):
    def __init__(self, env, *args, **kwargs):
        super(QTableBasedAgent, self).__init__(env, *args, **kwargs)
        self.silent_mode = False
        self.q_def_val = 0
        self.q_df = pd.DataFrame(
            data=iter.product(self.env.states_space, self.env.actions_space, [self.q_def_val]), 
            columns=['s', 'a', 'q']).astype({'s': object, 'a': object, 'q': float})
        self.discount_factor = 0.9 # gamma -> 1, farseer

        self.epsilon = 0.1 # e-greedy
        self.learning_rate = 0.1

    def choose_action(self, s):
        df = self.q_df
        if np.random.rand() >= self.epsilon:
            actions = self.env.get_actions(s)
            if df[(df['s']==s) & (df['a'].isin(actions))]['q'].max() == self.q_def_val:
                idx = df[(df['s']==s) & (df['q']==self.q_def_val)].index
                action = df.loc[np.random.choice(idx), 'a']
            else:
                q_df = df[(df['s']==s) & (df['a'].isin(actions))]['q']
                if q_df.empty:
                    action = None
                else:
                    idxmax = q_df.argmax()
                    action = df.loc[idxmax, 'a']
        else:
            q_list = self.env.get_actions(s)
            if len(q_list) == 0:
                action = None 
            else:
                action = np.random.choice(q_list)
        
        return action

    def learn(self, *args, **kwargs):
        raise NotImplementedError()

    def _play(self, load_fname):
        self.epsilon = 0.0 # maxQ, no random
        self.load_pickle('.'.join([load_fname, 'pickle']))
        
        i_step = 0
        obs = self.env.reset()
        while True:
            i_step += 1
            self.env.render()
            
            a = self.choose_action(obs)
            obs_, r, done = self.env.move_step(a)
            
            if done:
                msg = self.env.update_message(obs_)
                logging.info('Run for [%d] step(s), finally [%s] at [%s]' % (i_step, msg, obs_))
                self.env.render()
                break

            obs = obs_

    def play(self, load_fname):
        self.silent_mode = False
        self.env.after(100, self._play, load_fname)
        self.env.mainloop()
    
    def _train(self, n_episode, n_step=1000, save_fname=None, load_fname=None):
        raise NotImplementedError()

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


class QLearningAgent(QTableBasedAgent):
    def __init__(self, env, *args, **kwargs):
        super(QLearningAgent, self).__init__(env, *args, **kwargs)
    
    def learn(self, s, a, r, s_):
        df = self.q_df
        if self.env.is_terminal(s_):
            q_target = r
        else:
            # Q = R(s, a) + γ * maxQ(s', a')
            q_target = r + self.discount_factor * df[(df['s']==s_)]['q'].max()
        
        idx = df[(df['s']==s) & (df['a']==a)].index
        q_predict = df.loc[idx, 'q']
        df.loc[idx, 'q'] = q_predict + self.learning_rate * (q_target - q_predict)
    
    def _train(self, n_episode, n_step=1000, save_fname=None, load_fname=None):
        if load_fname != None:
            self.load_pickle('.'.join([load_fname, 'pickle']))

        for i_episode in range(n_episode):
            i_step = 0
            obs = self.env.reset()
            while True:
                i_step += 1
                if self.silent_mode == False:
                    self.env.render()
                
                a = self.choose_action(obs)
                obs_, r, done = self.env.move_step(a)

                self.learn(obs, a, r, obs_)

                obs = obs_
                
                if done:
                    msg = self.env.update_message(obs_)
                    logging.info('Episode [%d], run for [%d] step(s), finally [%s] at [%s]' % (i_episode+1, i_step, msg, obs_))
                    
                    if self.silent_mode == False:
                        self.env.render()
                    break

        self.env.destroy()
        
        if save_fname != None:
            self.save_pickle('.'.join([save_fname, 'pickle']))
            self.save_csv('.'.join([save_fname, 'csv']))


class SarsaAgent(QTableBasedAgent):
    def __init__(self, env, *args, **kwargs):
        super(SarsaAgent, self).__init__(env, *args, **kwargs)

    def learn(self, s, a, r, s_, a_):
        df = self.q_df
        if self.env.is_terminal(s_):
            q_target = r
        else:
            # Q = R(s, a) + γ * Q(s', a')
            q_target = r + self.discount_factor * df[(df['s']==s_) & (df['a']==a_)]['q'].sum()
        
        idx = df[(df['s']==s) & (df['a']==a)].index
        q_predict = df.loc[idx, 'q']
        df.loc[idx, 'q'] = q_predict + self.learning_rate * (q_target - q_predict)

    def _train(self, n_episode, n_step=1000, save_fname=None, load_fname=None):
        if load_fname != None:
            self.load_pickle('.'.join([load_fname, 'pickle']))

        for i_episode in range(n_episode):
            i_step = 0
            obs = self.env.reset()
            a = self.choose_action(obs)
            while True:
                i_step += 1
                if self.silent_mode == False:
                    self.env.render()
                
                obs_, r, done = self.env.move_step(a)
                a_ = self.choose_action(obs_)

                self.learn(obs, a, r, obs_, a_)
                
                obs = obs_
                a = a_

                if done:
                    msg = self.env.update_message(obs_)
                    logging.info('Episode [%d], run for [%d] step(s), finally [%s] at [%s]' % (i_episode+1, i_step, msg, obs_))
                    
                    if self.silent_mode == False:
                        self.env.render()
                    break
        
        self.env.destroy()
        
        if save_fname != None:
            self.save_pickle('.'.join([save_fname, 'pickle']))
            self.save_csv('.'.join([save_fname, 'csv']))
