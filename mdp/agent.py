import logging
import itertools as iter
import numpy as np
import pandas as pd
import pickle
import os


class AbstractAgent(object):
    def __init__(self, env, *args, **kwargs):
        self.env = env

    def play(self, *args, **kwargs):
        raise NotImplementedError()


class InteractiveAgent(AbstractAgent):
    def __init__(self, env, *args, **kwargs):
        super(InteractiveAgent, self).__init__(env, *args, **kwargs)
    
    def _play(self):
        obs = self.env.reset()
        i_step = 0

        while True:
            self.env.render()
            
            action = self.env.fetch_action()
            if action == None:
                continue
            else:
                i_step += 1
            
            obs_, _, done = self.env.move_step(obs, action)

            obs = obs_
            
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

    def save_csv(self, path, filename):
        raise NotImplementedError()

    def save_pickle(self, path, filename):
        raise NotImplementedError()

    def load_pickle(self, path, filename):
        raise NotImplementedError()


class QTableBasedAgent(AIAgent):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.1, *args, **kwargs):
        super(QTableBasedAgent, self).__init__(env, *args, **kwargs)
        self.silent_mode = False
        self.q_def_val = 0
        self.q = np.zeros((self.env.n_states, self.env.n_actions))
        self.gamma = gamma # gamma -> 1, farseer; discount factor

        self.epsilon_ori = epsilon
        self.epsilon = self.epsilon_ori # e-greedy
        self.alpha = lr # alpha, learning rate

    def choose_action(self, s):
        actions = self.env.get_actions(s)
        if len(actions) == 0:
            action = None
        else:
            if np.random.rand() >= self.epsilon:
                qvals = self.q[self.env.s(s), :]

                if qvals.max() == self.q_def_val:
                    idxs = np.argwhere(qvals==self.q_def_val).reshape(-1)
                    action = self.env.actions_space[np.random.choice(idxs)]
                else:
                    action = self.env.actions_space[qvals.argmax()]
            else:
                action = np.random.choice(actions)
        
        return action

    def learn(self, *args, **kwargs):
        raise NotImplementedError()

    def _play(self, path, load_fname):
        self.epsilon = 0.0 # maxQ, no random
        self.load_pickle(path, '.'.join([load_fname, 'pickle']))
        
        i_step = 0
        obs = self.env.reset()
        while True:
            i_step += 1
            self.env.render()
            
            a = self.choose_action(obs)
            obs_, r, done = self.env.move_step(obs, a)

            obs = obs_
            
            if done:
                msg = self.env.update_message(obs_)
                logging.info('Run for [%d] step(s), finally [%s] at [%s]' % (i_step, msg, obs_))
                self.env.render()
                break

        self.epsilon = self.epsilon_ori

    def play(self, path, load_fname):
        self.silent_mode = False
        self.env.after(100, self._play, path, load_fname)
        self.env.mainloop()
    
    def _train(self, n_episode, n_step=1000, path=None, save_fname=None, load_fname=None):
        raise NotImplementedError()

    def train(self, n_episode=100, n_step=1000, path=None, save_fname=None, load_fname=None):
        """
        train agent
        """
        self.env.after(100, self._train, n_episode, n_step, path, save_fname, load_fname)
        self.env.mainloop()

    def save_csv(self, path, filename):
        """
        only for easy checking (column original format lost), save well-trained params to file in csv format
        """
        np.savetxt(os.path.join(path, filename), self.q, delimiter=',')

    def save_pickle(self, path, filename):
        """
        save well-trained params to file in pickle format
        """
        with open(os.path.join(path, filename), 'wb') as fs:
            pickle.dump(self.q, fs)

    def load_pickle(self, path, filename):
        """
        load params from file in pickle format
        """
        with open(os.path.join(path, filename), 'rb') as fs:
            self.q = pickle.load(fs, encoding='bytes')


class QLearningAgent(QTableBasedAgent):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.1, *args, **kwargs):
        super(QLearningAgent, self).__init__(env, gamma, lr, epsilon, *args, **kwargs)
    
    def learn(self, s, a, r, s_):
        if self.env.is_terminal(s_):
            q_target = r
        else:
            # Q = R(s, a) + γ * maxQ(s', a')
            q_target = r + self.gamma * self.q[self.env.s(s_), :].max()
        
        q_predict = self.q[self.env.s(s), self.env.a(a)]
        self.q[self.env.s(s), self.env.a(a)] = q_predict + self.alpha * (q_target - q_predict)
    
    def _train(self, n_episode, n_step=1000, path='.', save_fname=None, load_fname=None):
        if load_fname != None:
            self.load_pickle(path, '.'.join([load_fname, 'pickle']))

        for i_episode in range(n_episode):
            i_step = 0
            obs = self.env.reset()
            while True:
                i_step += 1
                if self.silent_mode == False:
                    self.env.render()
                
                a = self.choose_action(obs)
                obs_, r, done = self.env.move_step(obs, a)

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
            self.save_pickle(path, '.'.join([save_fname, 'pickle']))
            self.save_csv(path, '.'.join([save_fname, 'csv']))


class SarsaAgent(QTableBasedAgent):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.1, *args, **kwargs):
        super(SarsaAgent, self).__init__(env, gamma, lr, epsilon, *args, **kwargs)

    def learn(self, s, a, r, s_, a_):
        if self.env.is_terminal(s_):
            q_target = r
        else:
            # Q = R(s, a) + γ * Q(s', a')
            q_target = r + self.gamma * self.q[self.env.s(s_), self.env.a(a_)]
        
        q_predict = self.q[self.env.s(s), self.env.a(a)]

        self.q[self.env.s(s), self.env.a(a)] = q_predict + self.alpha * (q_target - q_predict)

    def _reset_episode_params(self):
        pass

    def _train(self, n_episode, n_step=1000, path='.', save_fname=None, load_fname=None):
        if load_fname != None:
            self.load_pickle(path, '.'.join([load_fname, 'pickle']))

        for i_episode in range(n_episode):
            self._reset_episode_params()
            i_step = 0

            obs = self.env.reset()
            a = self.choose_action(obs)
            while True:
                i_step += 1
                if self.silent_mode == False:
                    self.env.render()
                
                obs_, r, done = self.env.move_step(obs, a)
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
            self.save_pickle(path, '.'.join([save_fname, 'pickle']))
            self.save_csv(path, '.'.join([save_fname, 'csv']))


class SarsaLambdaAgent(SarsaAgent):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.1, lamda=0.9, *args, **kwargs):
        super(SarsaLambdaAgent, self).__init__(env, gamma, lr, epsilon, *args, **kwargs)
        # add eligibility columns, default 0
        self.elig_def_val = 0
        self.elig_table = np.zeros((self.env.n_states, self.env.n_actions))

        # if lambda = 0, Sarsa(λ) -> Sarsa
        self.lamda = lamda

    def learn(self, s, a, r, s_, a_):
        # δ = R(s, a) + γ * Q(s', a') - Q(s, a)
        # q_target - q_predict
        delta = r + self.gamma * self.q[self.env.s(s_), self.env.a(a_)]\
            - self.q[self.env.s(s), self.env.a(a)]
        
        self._update_elig(s, a)
        
        # for all s, a: Q(s, a) = Q(s, a) + α * δ * E(s, a)
        self.q[:, :] = self.q[:, :] + self.alpha * delta * self.elig_table[:, :]
        # E(s, a) = γ * λ * E(s, a)
        self.elig_table[:, :] = self.gamma * self.lamda * self.elig_table[:, :]

    def _update_elig(self, s, a):
        # accumulating trace
        # E(s, a) = E(s, a) + 1
        # self.elig_table[self.env.s(s), self.env.a(a)] += 1

        # replacing trace
        # E(s, :) = 0, E(s, a) = 1
        self.elig_table[self.env.s(s), :] = 0
        self.elig_table[self.env.s(s), self.env.a(a)] = 1

    def _reset_lamda(self):
        self.elig_table[:, :] = self.elig_def_val

    def _reset_episode_params(self):
        self._reset_lamda()


class PolicyIteration(AIAgent):
    def __init__(self, env, gamma=0.9, *args, **kwargs):
        super(PolicyIteration, self).__init__(env, *args, **kwargs)
        self.gamma = gamma

        self.v = np.zeros((self.env.n_states))
        self.action_policy = np.random.randint(low=0, high=self.env.n_actions, size=(self.env.n_states))

    def v_func(self, s):
        return self.v[self.env.s(s)]

    def choose_action(self, s):
        return self.env.actions_space[self.action_policy[self.env.s(s)]]
    
    def argmax_action(self, s):
        argmax = None
        max = -np.inf
        actions = self.env.get_actions(s)

        for a in actions:
            states_ = self.env.get_states(s, a)
            tmp = 0
            for s_ in states_:
                tmp += self.env.prob(s, a, s_).sum() * (self.env.reward(s, a, s_) + self.gamma * self.v_func(s_))
            if tmp > max:
                max = tmp
                argmax = a
        return self.env.a(argmax)

    def update_action_policy(self, s):
        self.action_policy[self.env.s(s)] = self.argmax_action(s)
        
    def policy_eval(self):
        theta = 1e-4
        i_step = 0
        while True:
            i_step += 1
            error = 0
            v_tmp = np.zeros((self.env.n_states))
            for s in self.env.states_space:
                # skip terminal states
                if self.env.is_terminal(s):
                    continue
                
                # v_k = V(s)
                v = self.v_func(s)
                
                # v_k_1 = V(s) = Σ p(s', r|s, π(s))*[r + γ * V(s')]
                a = self.choose_action(s)
                states_ = self.env.get_states(s, a)
                for s_ in states_:
                    v_tmp[self.env.s(s)] = v_tmp[self.env.s(s)] + self.env.prob(s, a, s_).sum() * (self.env.reward(s, a, s_) + self.gamma * self.v_func(s_))
                
                # error = max(delta, |v - V(s)|)
                error = max(error, abs(v - v_tmp[self.env.s(s)]))

            # v_k+1
            self.v = v_tmp.copy()

            if error < theta:
                logging.info('After %d times policy evaluation:' % i_step)
                logging.info('error = %.6f' % error)
                logging.info('v = %s' % self.v)
                break

    def policy_impr(self):
        is_stable = True
        for s in self.env.states_space:
            # skip terminal states
            if self.env.is_terminal(s):
                continue

            # choose old-action
            a = self.choose_action(s)
            a_tmp = a
            
            # compute new-action
            self.update_action_policy(s)

            # change it to unstable state when old-action != new-action
            a = self.choose_action(s)
            if a_tmp != a:
                is_stable = False
                
        return is_stable
    
    def policy_iter(self):
        is_stable = False

        while is_stable == False:
            # run policy evaluation when policy is not stable, else we find V ~= v*, π ~= π*
            self.policy_eval()
            is_stable = self.policy_impr()

