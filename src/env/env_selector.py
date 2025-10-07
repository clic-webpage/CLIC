def env_selector(config_general, config_feedback, config_task):
    """
    Selects and initializes the environment, feedback receiver, and oracle policy
    based on the provided configuration dictionaries.
    """
    environment_name = config_general['environment'] # pendulum, metaworld, robosuite, obsAvoid, cartpole, mountaincar
    task = config_general['task']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']
    use_space_mouse = config_feedback['use_space_mouse']
    use_abs_action = config_general['use_abs_action']

    if use_space_mouse:
        from tools.feedback_spacenav import Feedback_spaceNav

    if human_teacher is True: 
        from tools.feedback_keyboard_3d import Feedback_keyboard_3d
        from tools.feedback_keyboard_2d import Feedback_keyboard_2d

    # ---------------- PushT environment ---------------- #
    if environment_name in ['PushT']:
        from env.pusht.pusht_env import PushTEnv
        env = PushTEnv(use_abs_action=use_abs_action, config=config_task)
            
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback_keyboard_2d(
                                    key_type=config_feedback['key_type'],
                                    h_up=config_feedback['h_up'],
                                    h_down=config_feedback['h_down'],
                                    h_right=config_feedback['h_right'],
                                    h_left=config_feedback['h_left'],
                                    h_null=config_feedback['h_null'])
        policy_oracle = None   # policy is defined in env.control_policy()

    if environment_name in ['cartpole']:
        from tools.feedback import Feedback
        env = gym.make(environment_name)  # create environment
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback(env=env,
                                key_type=config_feedback['key_type'],
                                h_up=config_feedback['h_up'],
                                h_down=config_feedback['h_down'],
                                h_right=config_feedback['h_right'],
                                h_left=config_feedback['h_left'],
                                h_null=config_feedback['h_null'])
    
    # ---------------- Robosuite robotic environments ---------------- #
    if environment_name in ['robosuite']:
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback_keyboard_3d(
                                key_type=config_feedback['key_type'],
                                h_up=config_feedback['h_up'],
                                h_down=config_feedback['h_down'],
                                h_right=config_feedback['h_right'],
                                h_left=config_feedback['h_left'],
                                h_null=config_feedback['h_null'])
        
        from env.robotsuite.env_robosuite import EnvRobosuite
        env = EnvRobosuite(env_name=task, config=config_task)
        if task == 'NutAssemblySquare':
            from env.robotsuite.nutassemblysquare_policy import NutAssemblyPolicy
            policy_oracle = NutAssemblyPolicy(use_abs_action= use_abs_action)
        elif task == 'PickPlaceCan':
            from env.robotsuite.pickCan_policy import PickCanPolicy
            policy_oracle = PickCanPolicy(use_abs_action= use_abs_action)
        elif task == 'TwoArmLift':
            from env.robotsuite.twoarmList_policy import TwoArmLiftPolicy
            policy_oracle = TwoArmLiftPolicy(use_abs_action=use_abs_action)

    if environment_name in ['mountaincar_continuous']:
        import gymnasium as gym
        env = gym.make('MountainCarContinuous-v0', render_mode="human")  
       
        from tools.feedback_keyboard_1d import Feedback_keyboard_1d
        feedback_receiver = Feedback_keyboard_1d(
                                key_type=config_feedback['key_type'],
                                h_up=config_feedback['h_up'],
                                h_down=config_feedback['h_down'],
                                h_right=config_feedback['h_right'],
                                h_left=config_feedback['h_left'],
                                h_null=config_feedback['h_null'])

    if environment_name in ['line_following']:
        # line following task
        from env.followLine.followLine_env import LineFollowerEnv
        # env = PushTEnv()
        if task == "line_following":
            env = LineFollowerEnv() 
            policy_oracle = env.control_policy
            if human_teacher is False: 
                feedback_receiver = None
            else:
                feedback_receiver = Feedback_keyboard_3d(
                                key_type=config_feedback['key_type'],
                                h_up=config_feedback['h_up'],
                                h_down=config_feedback['h_down'],
                                h_right=config_feedback['h_right'],
                                h_left=config_feedback['h_left'],
                                h_null=config_feedback['h_null'])
            
        elif task == "multiLine-Following":
            from env.followLine.multifollowLine_env import MultiLineFollowerEnv
            from tools.feedback_keyboard_multi_2d import Feedback_keyboard_multi_2d
            env = MultiLineFollowerEnv()
            policy_oracle = None
            if human_teacher is False: 
                feedback_receiver = None
            else:
                feedback_receiver = Feedback_keyboard_multi_2d(
                                    key_type=config_feedback['key_type'],
                                    h_up=config_feedback['h_up'],
                                    h_down=config_feedback['h_down'],
                                    h_right=config_feedback['h_right'],
                                    h_left=config_feedback['h_left'],
                                    h_null=config_feedback['h_null'],
                                    robot_num=env.n_balls)
        else: # raise error
            print("task not found")
            raise ValueError

    # ---------------- MetaWorld robotic environments ---------------- #    
    elif environment_name in ['metaworld']:
        from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
        # from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy
        from env.metaworld_env.sawyer_hammer_v2_policy import SawyerHammerV2Policy
        from env.metaworld_env.sawyer_assembly_v2_policy import  SawyerAssemblyV2Policy
        from env.metaworld_env.metaworld import MetaWorldSawyerEnv
        env = MetaWorldSawyerEnv(task)

        if use_space_mouse:
            feedback_receiver = Feedback_spaceNav()
        if human_teacher is False: 
            feedback_receiver = None
        else:
            feedback_receiver = Feedback_keyboard_3d(
                                key_type=config_feedback['key_type'],
                                h_up=config_feedback['h_up'],
                                h_down=config_feedback['h_down'],
                                h_right=config_feedback['h_right'],
                                h_left=config_feedback['h_left'],
                                h_null=config_feedback['h_null'])
            
        # Create Oracle policy
        if task == "drawer-open-v2-goal-observable" or task == "drawer-open-v2":
            policy_oracle = SawyerDrawerOpenV2Policy()
           
        elif task == "hammer-v2-goal-observable" or task == "hammer-v2":
            policy_oracle = SawyerHammerV2Policy()
            
        elif task == "assembly-v2-goal-observable" or task == "assembly-v2":
            policy_oracle = SawyerAssemblyV2Policy()
        else:
            raise NameError(f'Unknown task type: {task}')

    if environment_name in ['obsAvoid']:
        from env.DesiredActionSpace_ToyEnv.DesiredA_toyenv import DesiredA_ToyEnv
        from env.DesiredActionSpace_ToyEnv.DesiredA_toyenv_square import DesiredA_ToyEnv_square
        from env.DesiredActionSpace_ToyEnv.DesiredA_toyenv_TwoCircles import DesiredA_ToyEnv_TwoCircles
        if task == 'Desired_A':
            env = DesiredA_ToyEnv()
        elif task == 'Desired_A_Square':
            env = DesiredA_ToyEnv_square()

        feedback_receiver = None
        policy_oracle = None
        
    return env, policy_oracle, feedback_receiver