import os
import time
import random
import datetime
from datetime import date

import numpy as np
import pandas as pd
import cv2
import pygame
import tensorflow as tf
import mujoco
import matplotlib.pyplot as plt

import pdb

from tools.buffer_trajectory import TrajectoryBuffer
import hydra
from omegaconf import DictConfig

# (Optional) Prevent TensorFlow from allocating the entire GPU memory
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from env.env_selector_real_robot import env_selector_real_robot
from agents.agent_selector import agent_selector
from tools.oracle_feedback import ( 
    oracle_gimme_feedback,  
    oracle_feedback_HGDagger, 
    oracle_feedback_intervention_diff
)

from tools.Q_value_visualize_2d import (
    visualize_normalized_q_values_tf_2d, 
    combine_images
)

from tools.Q_value_visualize_1d import (
    visualize_normalized_q_values_tf_1d
)

from env.robotsuite.env_robosuite import combine_obs_dicts, extract_latest_obs_dict


# ==========================
#      HELPER FUNCTIONS
# ==========================

def is_env_done(info):
    """
    For environments such as MetaWorld, 'success' in info indicates done.
    Returns (env_done, success_indicator).
    """
    if info.get('success', 0) == 1:
        return True, 1
    return False, 0


def evaluation_saving_results_process(eval_agent, eval_env, feedback_receiver,  i_episode, i_repetition, max_steps, render_savefig_flag, history, data, eval_time_acc, SEED_id, config_general, config_agent):
    eval_start = time.time()
    success_rate, mean_error = evaluate_agent(
        eval_agent, eval_env, feedback_receiver,  i_episode, max_steps=max_steps, config_general=config_general, verbose=False, render_savefig_flag= render_savefig_flag,
    )
    eval_end = time.time()
    eval_time_acc += (eval_end - eval_start)

    # Save results

    data[-2:] = success_rate, mean_error
    for his, d in zip(history, data):
        his.append(d)

    if config_general['save_results']:
        save_training_progress(*history, eval_agent.e, config_general['task'], SEED_id, i_repetition, config_agent=config_agent)
    return eval_time_acc, history

def evaluate_agent(agent, env, feedback_receiver, i_episode, max_steps, config_general, render_savefig_flag=False, verbose=False, onlinetraining = True):
    """
    Evaluate agent by running 'evaluations_per_training' episodes.
    Returns: (success_rate, mean_error_to_teacher)
    """
    successes, error_list = 0, []
    # Optionally skip or reduce frequency in certain conditions
    evaluations_per_training_ = config_general['evaluations_per_training']
    environment_name = config_general['environment']
    task = config_general['task']
    

    if i_episode < 5 and onlinetraining:
        evaluations_per_training_ = 1

    agent.evaluation = True
    print("start evalution")
    for i_episode_ in range(evaluations_per_training_):

                    
        SEED = 100001 + i_episode_ * evaluations_per_training_
        random.seed(SEED)
        np.random.seed(SEED)

        ep_success = 0
        ep_error = 0
        obs, info = env.reset()

        last_action = np.zeros(agent.dim_a)
        steps_stuck = 0

        for t_ev in range(max_steps):
            # obs_proc = process_observation(obs)
            obs_proc = obs
            action = agent.action(obs_proc) 

            obs, reward, done, _, info = env.step(action)
            env_done_fake, success = is_env_done(info)

            done_restart = feedback_receiver.ask_for_done()
            done = done or env_done_fake or done_restart

            if success == 1:
                successes += 1
                ep_success = 1
                print('-------------success: ', successes, " num: ", i_episode_ + 1)

            last_action = action

            if done:
                break

        error_list.append(ep_error)
        if verbose:
            print(f"Evaluation -> success={bool(ep_success)}")

    success_rate = successes / evaluations_per_training_
    mean_error = np.mean(error_list) if error_list else 0.0

    print("end evalution")
    return success_rate, mean_error

def save_training_progress(ep_list, ts_list, time_list, fb_list, success_list,
                           error_list, e_mat, task_short, seed, rep_idx, config_agent):
    """Saves training progress to a CSV file in ./results/ directory."""
    if not os.path.exists('./results'):
        os.makedirs('./results')

    df = pd.DataFrame({
        'Episode': ep_list,
        'Timesteps': ts_list,
        'time': time_list,
        'Amount of feedback': fb_list,
        'Success rate': success_list,
        'error to simulated teacher': error_list
    })

    # 'e_mat' might be a single scalar or array; adapt as needed
    exp_id = config_agent['experiment_id']
    agent_type = config_agent['agent']
    agent_algorithm = config_agent['algorithm'] 
    e_val = e_mat[0] if isinstance(e_mat, (list, np.ndarray)) else e_mat
    fname = (f'./results/{exp_id}'
             f'_{agent_type}'
             f'_Alg-{agent_algorithm}'
             f'_{rep_idx}.csv')
    df.to_csv(fname, index=False)
    print(f"Saved training progress to {fname}")

def generate_random_numbers(n):
    """Generate random integers (0 to 1000) as seeds."""
    return [random.randint(0, 1000) for _ in range(n)]


# TODO in real-world, just teleoperate 
def test_teacher_policy(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent):

    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition +1
    random.seed(SEED)
    np.random.seed(SEED)

    evaluations_per_training_ = config_general['evaluations_per_training']
    environment_name = config_general['environment']
    task = config_general['task']
    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']

    episode_counter = 0
    control_frequency = 100
    rate = rospy.Rate(control_frequency)

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):


        observation, info = env.reset()

        # wait for human to start the robot
        print("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()

        episode_counter += 1

        action_agent = np.zeros(config_agent.dim_a)
        # for t in range(1, max_time_steps_episode + 1):
        
        while True:
            env.render_mode = 'human'
            env.render()
            # Time used for logging

            h = feedback_receiver.get_h()
            if h is not None and np.any(h):
                teacher_action = h
            else:
                # for velocity output
                teacher_action = np.zeros(config_agent.dim_a)
            action_agent = teacher_action
            # import pdb
            # pdb.set_trace()
            # Step environment
            observation, reward, done, _, info = env.step(teacher_action)

            if feedback_receiver.ask_for_done():
                done = True

            # rate.sleep()
            # time.sleep(0.03)
            if environment_name in ['metaworld']:
                state = env.get_state()
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake 

            if done:
                break
 

# ==========================
#      TRAINING LOGIC
# ==========================

def train_single_repetition(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent, render_savefig_flag = True):  # used in delftblue
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    import rospy
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']
    oracle_teaching_how_offen = config_general['oracle_teaching_how_offen']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obsAvoid, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']

    # Pendulum environment seeding example
    if environment_name in ['PushT']:
        env.set_seed(SEED)
        env_seeds = generate_random_numbers(max_num_of_episodes)
        env_seeds_eval = generate_random_numbers(max_num_of_episodes * 10)

    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)

    load_policy = config_agent.load_policy
    if load_policy:
        print('LOAD MODELS')
        from hydra.utils import get_original_cwd
        project_root = get_original_cwd()
        # import pdb; pdb.set_trace()
        full_path = os.path.join(project_root, current_agent.load_dir)
        print("load path: ", full_path)
        current_agent.load_dir = full_path
        current_agent.load_model()
        filename = current_agent.load_dir + 'buffer_data.pkl'
        current_agent.buffer.load_from_file(filename)
        # # for tactile sensor task, set sensor_readings to zero to see if sensor info helps 
        # for i in range(len(current_agent.buffer.buffer)):
        #     current_agent.buffer.buffer[i][0][:,-5:] = np.zeros(5)

        print("length of buffer: ", current_agent.buffer.length())

    # print("self.load_policy_flag: ", self.load_policy_flag)
    # # also load the buffer
    # print("LOAD buffer from load_dir")
    # filename = self.load_dir + 'buffer_data.pkl'
    # self.buffer.load_from_file(filename)
    # print("length of buffer: ", self.buffer.length())

    # print('LOAD MODELS')
    # from hydra.utils import get_original_cwd
    # project_root = get_original_cwd()
    # # import pdb; pdb.set_trace()
    # full_path = os.path.join(project_root, self.load_dir)
    # self.load_dir = full_path
    # # load model now
    # # Check if the model file exists
    # self.load_model()
                    

    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    start_time = time.time()

    control_frequency = 100
    rate = rospy.Rate(control_frequency)

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        print("i_episode: ", i_episode)
        ## i_episode % 2 == 1 is feedback mode
        # no_feedback_mode = i_episode % 2 == 0
        # record_preference_data = True
        no_feedback_mode = False


        observation, info = env.reset()

        # wait for human to start the robot
        print("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()


        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
        done_restart = False  # set by human feedback

        current_agent.evaluation = False
        t = 0
        # for t in range(1, max_time_steps_episode + 1):
        while True:  # no max episode length, only reset from human feedback
            # print("agent.evaluation: ", current_agent.evaluation)
            t = t + 1
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            # Time used for logging
            elapsed = (time.time() - start_time - eval_time_acc)
            time_str = str(datetime.timedelta(seconds=elapsed))

            # obs_proc = process_observation(observation) # TODO remove this line, check whether CLIC breaks
            obs_proc = observation
            
            # Agent's action or teacher's correction
            if not executed_human_correction:
                action_agent = current_agent.action(obs_proc)
                action = action_agent
            else:
                # If "execute_human_correction" is True
                # h = feedback_receiver.get_h() if human_teacher else h
                if h is not None and np.any(h):
                    # action_agent = current_agent.action(obs_proc)
                    action_agent = None
                    if agent_type in ['HG_DAgger','Implicit_BC','PVP','Diffusion']:
                        action = h
                    else:
                        action = last_action + np.matmul(current_agent.e, h)
                    action = np.clip(action, -1, 1)
                    # TODO for now we use teacher action at current state, instead of the positive action at last state
                    h_counter += 1
                    print("(1) action: ", action, " last_action: ", last_action)
                else:
                    # action_agent = current_agent.action(obs_proc)
                    start = time.time()
                    action_agent = current_agent.action(obs_proc)
                    end   = time.time()

                    print(f"Elapsed time: {end - start:.6f} seconds")
                    action = action_agent
                    print("action_agent: ", action_agent)
            
            # Step environment
            observation, reward, done, _, info = env.step(action)
            last_action = action 

            if human_teacher:
                # Real-time user feedback (example)
                h = feedback_receiver.get_h()
                if feedback_receiver.ask_for_done():
                    done = True

                h_raw = feedback_receiver.get_h()
                '''test adding noise to the robot'''
                print("(2) h_raw: ", h_raw)
                done_restart = feedback_receiver.ask_for_done()
                if np.any(h_raw):
                    if agent_algorithm in [ "CLIC_Explicit", "CLIC_EBM"]: # for policy contrastive intervention, we assume human teacher always demonstration
                        if action_agent is None:
                            action_agent = current_agent.action(obs_proc)  # key: the h is the difference between the action from the agent and the action from the human teacher!
                                                                    #         not its past action, which could be the action from the human teacher!!!
                        if config_agent.use_abs_action:
                            h_raw = h_raw * env.scale + env.ee_pose[:2]
                            h_raw = env.normalize_abs_action(h_raw, max_list=env.action_max, min_list=env.action_min)
                        h_raw, h_raw_nothreshold = oracle_feedback_intervention_diff(h_raw.copy(), action_agent, h, config=config_general)
                        # current_agent.e = np.identity(current_agent.dim_a)

                    elif agent_type in ['HG_DAgger','Implicit_BC','PVP','Diffusion']:
                        if action_agent is None:
                            action_agent = current_agent.action(obs_proc) 
                        if config_agent.use_abs_action:
                            h_raw = h_raw * env.scale + env.ee_pose[:2]
                            h_raw = env.normalize_abs_action(h_raw, max_list=env.action_max, min_list=env.action_min)
                        print("(2.0.0) h_raw: ", h_raw)
                        h_raw, h_raw_nothreshold = oracle_feedback_HGDagger(h_raw.copy(), action_agent, None,config=config_general)
                        
                        if np.any(h_raw):  # TODO ,test next time
                            last_action = action_agent
                        
                    
                h = h_raw
                print("(2.1) h: ", h)

            print("action: ", action, " action_agent: ", action_agent)
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or done_restart

            # Train agent
            # to do, to optimize, everytime we define a new algorithm, we need to do the same modifications for the three fuctions:train_single_repetition, train_single_repetition_offline
            if not no_feedback_mode:
                metrics = current_agent.collect_data_and_train(
                    last_action=last_action, 
                    h=h, obs_proc=obs_proc,
                    next_obs=observation, t=t_total, done=done,
                    agent_algorithm=agent_algorithm, agent_type=agent_type, i_episode=i_episode
                )
                
            t_total += 1
            rate.sleep()

            traj_buffer.add_transition(
                obs={'low_dim':obs_proc},      # or processed observation if you prefer
                teacher_action=h,
                done=done,
                timestep=t,
                no_robot_action = np.any(h),
                no_teacher_action = not np.any(h),
                robot_action=action_agent if action_agent is not None else last_action,                   # the feedback signal (if any)
                episode_id= i_episode
            )

            if done:
                base_dir = config_agent['base_dir']
                current_agent.saved_dir = os.path.join(base_dir, config_agent['experiment_id'], f"repetition_{i_repetition:03d}/") 
                current_agent.load_dir = current_agent.saved_dir
                if not os.path.exists(current_agent.saved_dir):
                    os.makedirs(current_agent.saved_dir)  
                
                traj_buffer.finish_trajectory()
                traj_buffer.save_to_file("trajectory_buffer_"+str(i_repetition))
                cumm_feedback += h_counter
                
                # Optionally save policy or buffer
                
                if save_policy:
                    current_agent.save_model()   # TODO change the save function of all other agents such as CLIC, HG_DAgger to save_model

                pfile = os.path.join(current_agent.saved_dir, 'buffer_data.pkl')
                current_agent.buffer.save_to_file(pfile)
                with open(os.path.join(current_agent.saved_dir, 'data.txt'), 'w') as f:
                    f.write(f"Steps: {t_total}\nEpisode: {i_episode}\nTime: {time_str}\nFeedbacks: {cumm_feedback}\n")
                break
        
        # traj_buffer.save_to_file("trajectory_buffer_"+str(i_repetition)+".pkl")

        # use_multi_process_evalution = True
        use_multi_process_evalution = False
        # Evaluate agent periodically
        evalutation_frequency = 10
        # TODO change pendulum env name to PushT
        if environment_name in ['robosuite', 'PushT']:  # for push-T
            if i_episode > 150:
                evalutation_frequency = 1
            else:
                # evalutation_frequency = 5
                evalutation_frequency = 10

        


        if i_episode % evalutation_frequency == 0 or repetition_done:
            skip_evaluation = None
            print("press 's' key to skip the whole evaluation, 'q' for not skip")
            while skip_evaluation is None:
                skip_evaluation = feedback_receiver_keyboard.ask_whether_skip_evaluation()
            print("skip_evaluation: ", skip_evaluation)
            print("press space key to start")
            while not feedback_receiver_keyboard.ask_for_done():
                env.hold_on_mode()

            '''Start evaluation here'''
            if not skip_evaluation:
                history = [ep_history, ts_history, time_history, fb_history, success_history, error_history]
                success_rate, mean_error = None, None
                data =    [episode_counter, t_total, time_str, cumm_feedback, success_rate, mean_error]

                eval_time_acc, history_new = evaluation_saving_results_process(eval_agent=current_agent, eval_env= env, feedback_receiver=feedback_receiver, i_episode=i_episode, i_repetition=i_repetition,
                                                    max_steps=max_time_steps_episode, render_savefig_flag = render_savefig_flag, 
                                                    history=history, data=data, eval_time_acc = eval_time_acc, SEED_id=SEED, config_general=config_general, config_agent=config_agent)
                ep_history, ts_history, time_history, fb_history, success_history, error_history = history_new

        if repetition_done:
            break

    print(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")



def collect_dataset_repetition(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent, render_savefig_flag = True):  # used in delftblue
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    import rospy
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']
    oracle_teaching_how_offen = config_general['oracle_teaching_how_offen']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obsAvoid, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']


    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)    
    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False
    

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    start_time = time.time()

    control_frequency = 10
    rate = rospy.Rate(control_frequency)

    # TODO define saving freq for the traj data

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        print("i_episode: ", i_episode)
        ## i_episode % 2 == 1 is feedback mode
        # no_feedback_mode = i_episode % 2 == 0
        # record_preference_data = True
        no_feedback_mode = False


        observation, info = env.reset()

        # wait for human to start the robot
        print("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()


        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
        done = False
        done_restart = False

        current_agent.evaluation = False
        t = 0
        # for t in range(1, max_time_steps_episode + 1):
        while True:  # no max episode length, only reset from human feedback
            # print("agent.evaluation: ", current_agent.evaluation)
            t = t + 1
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            # Time used for logging
            elapsed = (time.time() - start_time - eval_time_acc)
            time_str = str(datetime.timedelta(seconds=elapsed))

            # obs_proc = process_observation(observation) # TODO remove this line, check whether CLIC breaks
            obs_proc = observation
            
            # Agent's action or teacher's correction
            h = feedback_receiver.get_h()
            # if feedback_receiver.ask_for_done():   # TODO this two lines makes done always false, very strange. should find out why
            #     done = True
            done_restart = feedback_receiver.ask_for_done()
            if np.any(h):
                if config_agent.use_abs_action:
                    action = h * env.scale + env.ee_pose[:2]
                    print("action before normalization: ", action)
                    action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
                else:
                    action = h
            else:
                if config_agent.use_abs_action:
                    action = env.ee_pose[:2]
                    print("action before normalization: ", action)
                    action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
                else:
                    action = np.zeros(current_agent.dim_a)
            print("h: ", h, " done: ", done, " action: ", action)
            # Step environment
            observation, reward, done, _, info = env.step(action)

            
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or done_restart 

            print("h: ", h, " done: ", done)
            last_action = action 
            
            # if receive_feedback_phrase:
            #     teacher_action_to_buffer = teacher_action_i
            #     robot_action_to_buffer = action_agent_i
            #     no_robot_action = False
            #     no_teacher_action = False
            # else:
            #     robot_action_to_buffer = action_i
            #     teacher_action_to_buffer = np.zeros_like(robot_action_to_buffer)
            #     no_robot_action = False
            #     no_teacher_action = True
            
            # save to trajectory buffer
            
            if np.any(h):  # donot record the data if the human doesn't take any actions
                teacher_action_to_buffer = action
                traj_buffer.add_transition(
                    obs=extract_latest_obs_dict(obs_proc),      # or processed observation if you prefer
                    teacher_action=teacher_action_to_buffer,
                    done=done,
                    timestep=t,
                    no_robot_action = True,
                    no_teacher_action = False,
                    robot_action=np.zeros_like(teacher_action_to_buffer),                   # the feedback signal (if any)
                    episode_id= i_episode
                )

            if done:
                traj_buffer.finish_trajectory()
                traj_buffer.save_to_file("trajectory_buffer_"+str(i_repetition))

            if done:
                print("leaave the while")
                base_dir = config_agent['base_dir']
                current_agent.saved_dir = os.path.join(base_dir, config_agent['experiment_id'], f"repetition_{i_repetition:03d}/") 
                current_agent.load_dir = current_agent.saved_dir
                if not os.path.exists(current_agent.saved_dir):
                    os.makedirs(current_agent.saved_dir)  
                cumm_feedback += h_counter
                # Optionally save policy or buffer
            
            if done:
                break
            
            rate.sleep()


        if repetition_done:
            break

    print(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")

def Replay_collected_dataset(env, policy_oracle, feedback_receiver, feedback_receiver_keyboard, i_repetition, config_general, config_agent, render_savefig_flag = True):  # used in delftblue
    traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning
    import rospy
    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']
    oracle_teaching_how_offen = config_general['oracle_teaching_how_offen']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obsAvoid, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']


    print('LOAD Actions')
    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    # import pdb; pdb.set_trace()
    action_path = 'outputs_docker/teacher_action.pkl'
    full_path = os.path.join(project_root, action_path)
    print("load path: ", full_path)
    import pickle
    with open(full_path, 'rb') as f:
        loaded_teacher_action = pickle.load(f)
    print('loaded_teacher_action: ', loaded_teacher_action)
    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)
    
    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False
    

    control_frequency = 20
    rate = rospy.Rate(control_frequency)

    # TODO define saving freq for the traj data

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        print("i_episode: ", i_episode)
        ## i_episode % 2 == 1 is feedback mode
        # no_feedback_mode = i_episode % 2 == 0
        # record_preference_data = True
        no_feedback_mode = False


        observation, info = env.reset()

        # wait for human to start the robot
        print("press space key to start")
        while not feedback_receiver_keyboard.ask_for_done():
            env.hold_on_mode()


        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
        done = False
        done_restart = False

        current_agent.evaluation = False
        t = 0
        # for t in range(1, max_time_steps_episode + 1):
        while True:  # no max episode length, only reset from human feedback
            # print("agent.evaluation: ", current_agent.evaluation)
            t = t + 1


            # obs_proc = process_observation(observation) # TODO remove this line, check whether CLIC breaks
            obs_proc = observation
            
            # Agent's action or teacher's correction


            # if feedback_receiver.ask_for_done():   # TODO this two lines makes done always false, very strange. should find out why
            #     done = True
            done_restart = feedback_receiver.ask_for_done()

            action = loaded_teacher_action[t] if t<len(loaded_teacher_action) else loaded_teacher_action[-1]
            # action = env.normalize_abs_action(action, max_list=env.action_max, min_list=env.action_min)
            # Step environment
            observation, reward, done, _, info = env.step(action)

            
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or done_restart 

            last_action = action 
            
            
            if done:
                break
            
            rate.sleep()


        if repetition_done:
            break

    print(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")

def train_offline_repetition( i_repetition, config_general, config_agent, render_savefig_flag = False):  # used in delftblue
    # traj_buffer = TrajectoryBuffer()  # used to save trajectory-level data for preference learning

    """Run one repetition of training: init seeds, run episodes, eval & save."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']
    oracle_teaching_how_offen = config_general['oracle_teaching_how_offen']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obsAvoid, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']


    # Initialize agent
    current_agent = agent_selector(agent_type, config_agent)

    print("LOAD buffer")
    '''Method 1 of Loading buffer: from traj_dataset'''
    # traj_buffer_path ='/media/110394e6-69f5-4c10-a88b-f27235baa55d/CLIC_DP_real_exp_data/trajectory_buffer_0716_insertT_abs_86eps.hdf5'
    # current_agent.buffer.ingest_trajectory_hdf5(traj_filename =traj_buffer_path )
    
    '''Method 2 of Loading buffer: from buffer.h5 (generated files by Method1)'''
    # buffer_path = '/media/110394e6-69f5-4c10-a88b-f27235baa55d/CLIC_DP_real_exp_data/trajectory_buffer_0716_insertT_abs_86eps.h5'
    # current_agent.buffer.load_from_file(buffer_path)

    '''Method 3 of loading buffer: load pkl (if have)'''
    print('LOAD MODELS')
    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    # import pdb; pdb.set_trace()
    full_path = os.path.join(project_root, current_agent.load_dir)
    print("load path: ", full_path)
    filename = current_agent.load_dir + 'buffer_data.pkl'
    current_agent.buffer.load_from_file(filename)
    # #for tactile sensor task, set sensor_readings to zero to see if sensor info helps 
    for i in range(len(current_agent.buffer.buffer)):
        current_agent.buffer.buffer[i][0][:,-5:] = np.zeros(5)

    print("length of buffer: ", current_agent.buffer.length())


    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False
    obs_proc = None
    time_str = 0
    observation =None

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    start_time = time.time()

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        print("i_episode: ", i_episode)
        ## i_episode % 2 == 1 is feedback mode
        # no_feedback_mode = i_episode % 2 == 0
        # record_preference_data = True
        record_preference_data = False
        no_feedback_mode = False

        episode_counter += 1
        h_counter = 0  # how many times feedback was given
        last_action = None
        h = None
        done = True

        current_agent.evaluation = False
        
        # Train agent
        # to do, to optimize, everytime we define a new algorithm, we need to do the same modifications for the three fuctions:train_single_repetition, train_single_repetition_offline
        if not no_feedback_mode:
            metrics = current_agent.collect_data_and_train(
                last_action=last_action, 
                h=h, obs_proc=obs_proc,
                next_obs=observation, t=t_total, done=done,
                agent_algorithm=agent_algorithm, agent_type=agent_type, i_episode=i_episode
            )
        
        base_dir = config_agent['base_dir']
        current_agent.saved_dir = os.path.join(base_dir, config_agent['experiment_id'], f"repetition_{i_repetition:03d}/") 
        current_agent.load_dir = current_agent.saved_dir
        if not os.path.exists(current_agent.saved_dir):
            os.makedirs(current_agent.saved_dir)  

        
        if save_policy:
            current_agent.save_model()   # TODO change the save function of all other agents such as CLIC, HG_DAgger to save_model    
    
        if repetition_done:
            break

    print(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


def evalaution_without_training(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent):
    """Run one repetition of training using offline data."""

    SEED = 100001 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']
    oracle_teaching_how_offen = config_general['oracle_teaching_how_offen']

    environment_name = config_general['environment'] # pendulum(PushT), metaworld, robosuite, obsAvoid, cartpole, mountaincar
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']

    # Initialize agent
    eval_agent = agent_selector(agent_type, config_agent)

    # print("LOAD buffer")
    # filename = current_agent.load_pretrained_dir + 'buffer_data.pkl'
    # current_agent.buffer.load_from_file(filename)
    # print("length of buffer: ", current_agent.buffer.length())
        

    base_dir = config_agent['base_dir']
    eval_agent.saved_dir = os.path.join(base_dir, config_agent['experiment_id'], f"repetition_{i_repetition:03d}/") 
    eval_agent.load_dir = eval_agent.saved_dir
    if not os.path.exists(eval_agent.saved_dir):
        os.makedirs(eval_agent.saved_dir)  

    print("eval_agent.saved_dir: ", eval_agent.saved_dir)
    eval_agent.load_model()
    
    # Tracking
    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    time_str = 0
    repetition_done = False

    # For saving
    ep_history, ts_history, time_history = [], [], []
    fb_history, success_history, error_history = [], [], []

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        SEED = 100006 + i_episode

        # 100001 fails
        random.seed(SEED)
        np.random.seed(SEED)
        print("i_episode: ", i_episode)
        episode_counter += 1
      
    
        history = [ep_history, ts_history, time_history, fb_history, success_history, error_history]
        success_rate, mean_error = None, None
        data =    [episode_counter, t_total, time_str, cumm_feedback, success_rate, mean_error]

        eval_time_acc, history_new = evaluation_saving_results_process(eval_agent=eval_agent, eval_env= env, i_episode=i_episode, i_repetition=i_repetition,
                                            max_steps=max_time_steps_episode, render_savefig_flag = False, 
                                            history=history, data=data, eval_time_acc = eval_time_acc, SEED_id=SEED)

        ep_history, ts_history, time_history, fb_history, success_history, error_history = history_new

        if repetition_done:
            break

    print(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")


# run by using: python main-v3-cleaned_new_config.py --config-name =train_CLIC_Diffusion_image_Ta8
@hydra.main(config_path="config_real", config_name="train_CLIC_Diffusion_image_Ta1")
def main(cfg: DictConfig):
    config_general  = cfg.GENERAL
    config_agent    = cfg.AGENT
    config_feedback = cfg.FEEDBACK
    config_task = cfg.task
    # set up env here

    """Main entry point: run multiple training repetitions."""
    print("Starting training...\n")
    for rep_idx in range(config_general['number_of_repetitions']):
        # train_single_repetition_trajectory_segments(rep_idx, config_agent)  # online training, generate segement dataset

        # import pdb; pdb.set_trace()

        # env, policy_oracle, feedback_receiver, feedback_receiver_keyboard = env_selector_real_robot(config_general, config_feedback, config_task)
        # collect_dataset_repetition(env, policy_oracle, feedback_receiver,feedback_receiver_keyboard, rep_idx, config_general, config_agent)
        # # Replay_collected_dataset(env, policy_oracle, feedback_receiver,feedback_receiver_keyboard, rep_idx, config_general, config_agent)

        
        if config_agent['evaluate']:
            import rospy
            env, policy_oracle, feedback_receiver, feedback_receiver_keyboard = env_selector_real_robot(config_general, config_feedback, config_task)
            evalaution_without_training(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent)
        elif config_agent['offline_training']:
            train_offline_repetition( rep_idx, config_general, config_agent)   # offline training with dataset
        else:
            import rospy
            env, policy_oracle, feedback_receiver, feedback_receiver_keyboard = env_selector_real_robot(config_general, config_feedback, config_task)
            train_single_repetition(env, policy_oracle, feedback_receiver,feedback_receiver_keyboard, rep_idx, config_general, config_agent)  # online training


  
if __name__ == "__main__":
    main()
