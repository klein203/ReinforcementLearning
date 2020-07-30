import logging


if __name__ == "__main__":
    # log to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log to file
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='20200728.log', filemode='w')

    # import bandit.starter as ch2
    # ch2.ch2_3(10, 1000)
    # ch2.ch2_5(100, 10000)
    # ch2.move_step_value_trends()
    # ch2.ch2_7(10, 1000)
    # ch2.ch2_8(1000, 1000)
    
    import mdp.starter as ch3
    ch3.interactive_agent_run()
