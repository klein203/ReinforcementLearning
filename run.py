import logging
import time

import os

def ch2_case():
    import bandit.starter as bandit
    # bandit.ch2_3(10, 1000)
    # bandit.ch2_5(100, 10000)
    # bandit.move_step_value_trends()
    # bandit.ch2_7(10, 1000)
    # bandit.ch2_8(1000, 1000)

def ch3_case():
    import mdp.starter as mdp
    # mdp.interactive_agent_run()
    # mdp.ch3_1_autocleaner(100)

def ch4_case():
    import mdp.starter as mdp
    mdp.ch4_3_gridworld_policy_iteration()
    mdp.ch4_4_gridworld_value_iteration()

def ch6_case():
    today = time.strftime("%Y%m%d", time.localtime())

    weights_path = os.path.join(os.path.realpath('.'), 'mdp', 'temp')
    
    filename = 'q_learning_%s_0' % today
    # mdp.ch3_qlearning(mode='train', n_episode=200, path=weights_path, filename=filename, load_file=False, silent_mode=True)
    # mdp.ch3_qlearning(mode='play', path=weights_path, filename=filename)
    
    filename = 'sarsa_%s_0' % today
    # mdp.ch3_sarsa(mode='train', n_episode=200, path=weights_path, filename=filename, load_file=False, silent_mode=True)
    # mdp.ch3_sarsa(mode='play', path=weights_path, filename=filename)

    filename = 'sarsa_lambda_%s_0' % today
    # mdp.ch3_sarsa_lambda(mode='train', n_episode=200, path=weights_path, filename=filename, load_file=False, silent_mode=True)
    # mdp.ch3_sarsa_lambda(mode='play', path=weights_path, filename=filename)



if __name__ == "__main__":
    # log to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log to file
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='20200804.log', filemode='w')

    # ch2_case()
    # ch3_case()
    ch4_case()

    # ch6_case()
