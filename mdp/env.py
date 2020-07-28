import logging
import numpy as np
import pandas as pd

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
        self.p_df = pd.DataFrame(data=p_df, columns=['s', 'a', 's_', 'p'])
        self.r_df = pd.DataFrame(data=r_df, columns=['s', 'a', 's_', 'r'])
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
        filter_df = df[(df['s']==s)&(df['a']==a)&(df['s_']==s_)]
        if filter_df.empty:
            if s < 0 or s >= self.n_states:
                raise Exception('invalid s=%s' % s)
            if a < 0 or a >= self.n_actions:
                raise Exception('invalid a=%s' % a)
            if s_ < 0 or s_ >= self.n_states:
                raise Exception('invalid s_=%s' % s_)

            raise Exception('invalid p(%s|%s, %s)=p(%s|%s, %s)' % (self.states_space[s_], self.states_space[s], self.actions_space[a], s_, s, a))
        return filter_df['p'][0]
    
    def r(self, s, a, s_):
        df = self.r_df
        filter_df = df[(df['s']==s)&(df['a']==a)&(df['s_']==s_)]
        if filter_df.empty:
            if s < 0 or s >= self.n_states:
                raise Exception('invalid s=%s' % s)
            if a < 0 or a >= self.n_actions:
                raise Exception('invalid a=%s' % a)
            if s_ < 0 or s_ >= self.n_states:
                raise Exception('invalid s_=%s' % s_)

            raise Exception('invalid r(%s|%s, %s)=r(%s|%s, %s)' % (self.states_space[s_], self.states_space[s], self.actions_space[a], s_, s, a))
        return filter_df['r'][0]

    def get_actions_probs(self, s):
        df = self.p_df
        filter_df = df[(df['s']==s)]
        return filter_df[['a', 'p']] 
    
    def is_terminal(self, s):
        if s not in self.states_space:
            raise Exception('invalid s=%s' % s)
        return s in self.terminal_states

    # def get_available_actions(self, s):
    #     df = self.p_df
    #     filter_df = df[(df['s']==s)]
    #     return filter_df[['a', 's_', 'p']]

    def move(self):
        limited_move = 1000
        # while True:

