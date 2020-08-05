import logging

def ch2_case():
    import bandit.starter as ch2
    # ch2.ch2_3(10, 1000)
    # ch2.ch2_5(100, 10000)
    # ch2.move_step_value_trends()
    # ch2.ch2_7(10, 1000)
    # ch2.ch2_8(1000, 1000)

def ch3_case():
    import mdp.starter as ch3
    # ch3.interactive_agent_run()
    # ch3.ch3_6(100)

    filename = 'q_learning_20200805_0'
    ch3.ch3_qlearning(mode='train', n_episode=100, filename=filename, load_file=True, silent_mode=True)
    # ch3.ch3_qlearning(mode='play', filename=filename)


if __name__ == "__main__":
    # log to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log to file
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='20200804.log', filemode='w')

    # ch2_case()
    ch3_case()
