

def env_selector_real_robot(config_general, config_feedback, config_task):
    environment_name = config_general['environment'] # pendulum, metaworld, robosuite, obsAvoid, cartpole, mountaincar
    task = config_general['task']

    use_space_mouse = config_feedback['use_space_mouse']


    from tools.feedback_keyboard_3d import Feedback_keyboard_3d
    from tools.feedback_keyboard_3d_qbhand import Feedback_keyboard_3d_qbhand
    from tools.feedback_spacenav import Feedback_spaceNav
    from tools.feedback_spacenav_position import Feedback_spaceNav_position

    if environment_name in ['Kuka']:
        from env.kuka.kuka_env_ball_fixedX import KUKAenv_ball_fixedX 
        from env.kuka.kuka_env_6dEE_SoftH import KUKAenv_6dEE_SoftH
        from env.kuka.kuka_env_6dEE import KUKAenv_6dEE
        from env.kuka.kuka_env_PushT import KUKAenv_pushT 
        print("type of task: ", type(task), " ", config_general['task'])
        if config_general['task'] == "kuka-ball":
            env = KUKAenv_ball_fixedX()
            print("use kuka-ball")
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav()
        elif config_general['task'] == 'kuka-6dEE':
            env = KUKAenv_6dEE()
            print("KUKAenv_6dEE_SoftH")
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav()
        elif config_general['task'] == 'kuka-pushT':
            env = KUKAenv_pushT()
            print("KUKAenv_pushT")
            if use_space_mouse:
                feedback_receiver = Feedback_spaceNav_position()
        else:
            raise NotImplementedError
    
    feedback_receiver_keyboard = Feedback_keyboard_3d(
                        key_type=config_feedback['key_type'],
                        h_up=config_feedback['h_up'],
                        h_down=config_feedback['h_down'],
                        h_right=config_feedback['h_right'],
                        h_left=config_feedback['h_left'],
                        h_null=config_feedback['h_null'])
    
    feedback_receiver_keyboard_qbhand = Feedback_keyboard_3d_qbhand(
                        key_type=config_feedback['key_type'],
                        h_up=config_feedback['h_up'],
                        h_down=config_feedback['h_down'],
                        h_right=config_feedback['h_right'],
                        h_left=config_feedback['h_left'],
                        h_null=config_feedback['h_null'])

            
    #env.init_varaibles()
    print("task: ", task)
    policy_oracle = None
    return env, policy_oracle, feedback_receiver, feedback_receiver_keyboard