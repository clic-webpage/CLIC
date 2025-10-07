import os
import time
import random
import datetime

import numpy as np
import pandas as pd

from tools.buffer_trajectory import TrajectoryBuffer
import hydra
from omegaconf import DictConfig

# (Optional) Prevent TensorFlow from allocating the entire GPU memory
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from env.env_selector import env_selector
from agents.agent_selector import agent_selector
from tools.oracle_feedback import (
    oracle_gimme_feedback,
    oracle_feedback_HGDagger,
    oracle_feedback_intervention_diff
)

# ==========================
#     HELPER FUNCTIONS
# ==========================
def is_env_done(info):
    """
    For environments such as MetaWorld, 'success' in info indicates done.
    Returns (env_done, success_indicator).
    """
    if info.get('success', 0) == 1:
        return True, 1
    return False, 0

def get_teacher_action(environment_name, observation, action_agent = None, env=None, policy_oracle=None):
    """
    If there's an oracle teacher, get the teacher action. 
    Clips to [-1, 1] if out of range.
    """
    if environment_name in ['metaworld', 'robosuite']:
        # Get action from oracle
        if environment_name in ['robosuite']:
            action_teacher = policy_oracle.get_action(env.obs_extracted, env)
        else:
            action_teacher = policy_oracle.get_action(observation)

    if environment_name in ['mountaincar']:
        action_teacher = env.control_policy(observation)

    if environment_name in ['PushT']:
        action_teacher = env.control_policy(action_agent)

    if environment_name in ['obsAvoid']:
        action_teacher = env.control_policy(observation, action_agent)

    if environment_name in ['line_following']:
        action_teacher = env.control_policy(observation)

    return action_teacher

def save_training_progress(history_dict, task_short, seed, rep_idx, config_agent):
    """Saves training progress from a dictionary to a CSV file in ./results/ directory."""
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Directly create a DataFrame from the history dictionary
    df = pd.DataFrame(history_dict)
    
    exp_id = config_agent['experiment_id']
    fname = f'./results/{exp_id}_{rep_idx}.csv'
    df.to_csv(fname, index=False)
    print(f"Saved training progress to {fname}")


def evaluation_saving_results_process(eval_agent, eval_env, policy_oracle, i_episode, i_repetition,
                                      max_steps, render_savefig_flag,
                                      history, current_data, eval_time_acc, SEED_id,
                                      config_general, config_agent):
    """
    Evaluates the agent, updates the history and current data dictionaries, and saves the results.
    This version is flexible and handles any metrics passed in `current_data`.
    """
    eval_start = time.time()
    success_rate, mean_error = evaluate_agent(
        eval_agent, eval_env, policy_oracle, i_episode, max_steps=max_steps,
        config_general=config_general, verbose=False, render_savefig_flag=render_savefig_flag,
    )
    eval_end = time.time()
    eval_time_acc += (eval_end - eval_start)

    # Add evaluation results to the current data point
    current_data['Success rate'] = success_rate
    current_data['error to simulated teacher'] = mean_error

    # Append all data from the current step to the main history dictionary
    # .setdefault() creates a new list if the key doesn't exist yet, preventing errors
    for key, value in current_data.items():
        history.setdefault(key, []).append(value)

    if config_general['save_results']:
        save_training_progress(history, config_general['task'], SEED_id, i_repetition, config_agent=config_agent)

    return eval_time_acc, history

# ==========================
#     Evalaution 
# ==========================
def evaluate_agent(agent, env, policy_oracle, i_episode, max_steps, config_general, render_savefig_flag=False, verbose=False, onlinetraining = True):
    successes, error_list = 0, []
    # Optionally skip or reduce frequency in certain conditions
    evaluations_per_training_ = config_general['evaluations_per_training']
    environment_name = config_general['environment']
    task = config_general['task']
    

    if i_episode < 5 and onlinetraining:
        evaluations_per_training_ = 1

    if environment_name in ['robosuite', 'PushT']:
        if i_episode < 30 and onlinetraining:
            evaluations_per_training_ = 1
        env._evaluation = True
    
    if environment_name in ['line_following']:
        evaluations_per_training_ = 1

    agent.evaluation = True
    print("start evalution")
    for i_episode_ in range(evaluations_per_training_):
                  
        SEED = 100001 + i_episode * evaluations_per_training_ + i_episode_
        random.seed(SEED)
        np.random.seed(SEED)

        ep_success = 0
        ep_error = 0
        obs, info = env.reset()

        if environment_name in ['PushT']:
            print("set seed: ", SEED)
            env.set_seed(SEED)

        last_action = np.zeros(agent.dim_a)
        steps_stuck = 0

        for t_ev in range(max_steps):
            # Render if needed
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            obs_proc = obs
            action = agent.action(obs_proc)

            if environment_name in ['PushT']:
                teacher = np.array([0, 0])  # save time by avoiding this
                # teacher = get_teacher_action(environment_name, obs,action, env, policy_oracle)
            else:
                teacher = get_teacher_action(environment_name, obs,action, env, policy_oracle)
            if teacher is not None:
                ep_error += np.linalg.norm(teacher - action, ord=2)

            obs, reward, done, _, info = env.step(action)
            # obs, reward, done, _, info = env.step(teacher)
            env_done_fake, success = is_env_done(info)
            done = done or env_done_fake or t_ev == max_steps-1

            if success == 1:
                successes += 1
                ep_success = 1
                print('-------------success: ', successes, " num: ", i_episode_ + 1)

            # Optional: check "stuck" if using MetaWorld
            if environment_name in ['metaworld']:
                if np.linalg.norm(last_action - action) < 0.01:
                    steps_stuck += 1
                else:
                    steps_stuck = 0
                if steps_stuck > 200:
                    done = True
            last_action = action

            if done:
                if environment_name in ['robosuite']:
                    policy_oracle.reset() # reset the state machine
                break

        error_list.append(ep_error)
        if verbose:
            print(f"Evaluation -> success={bool(ep_success)}")

    success_rate = successes / evaluations_per_training_
    mean_error = np.mean(error_list) if error_list else 0.0

    if environment_name in ['robosuite']:
        policy_oracle.reset() # reset the state machine
    print("end evalution")
    return success_rate, mean_error

def generate_random_numbers(n):
    """Generate random integers (0 to 1000) as seeds."""
    return [random.randint(0, 1000) for _ in range(n)]

def test_teacher_policy(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent):
    """test the teacher policy"""
    SEED = 48 + i_repetition +1
    random.seed(SEED)
    np.random.seed(SEED)

    environment_name = config_general['environment']
    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    
    # Pendulum environment seeding example
    if environment_name in ['PushT']:
        env.set_seed(SEED)
        env_seeds = generate_random_numbers(max_num_of_episodes)
        env_seeds_eval = generate_random_numbers(max_num_of_episodes * 10)

    episode_counter = 0
    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):

        if environment_name in ['PushT']:
            env.set_seed(env_seeds[i_episode])

        observation, info = env.reset()

        episode_counter += 1

        action_agent = np.zeros(config_agent.dim_a)
        for t in range(1, max_time_steps_episode + 1):
            env.render_mode = 'human'
            env.render()
            # Time used for logging

            teacher_action = get_teacher_action(environment_name, observation, action_agent=action_agent,  env=env, policy_oracle=policy_oracle)
            action_agent = teacher_action + np.random.normal(loc=0.0, scale=0.25, size=teacher_action.shape)

            teacher_action_, h_no_threshold = oracle_feedback_HGDagger(teacher_action, action_agent, None, config=config_general)
            if not np.any(teacher_action_):
                teacher_action_ = teacher_action

            observation, reward, done, _, info = env.step(np.array(teacher_action_))

            if environment_name in ['metaworld']:
                state = env.get_state()
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or t == max_time_steps_episode

            if done and environment_name in ['robosuite']:
                policy_oracle.reset() # reset the state machine

            if done:
                break

# =========================================================
#     TRAINING LOGIC: Interactive Imitation Learning Loop
# =========================================================
def train_single_repetition(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent, render_savefig_flag = False):
    """Run one repetition of training using the flexible dictionary-based logging."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    executed_human_correction = config_general['executed_human_correction']
    oracle_teacher = config_general['oracle_teacher']
    human_teacher = config_general['human_teacher']
    oracle_teaching_how_offen = config_general['oracle_teaching_how_offen']

    environment_name = config_general['environment']
    task = config_general['task']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']

    if environment_name in ['PushT']:
        env.set_seed(SEED)
        env_seeds = generate_random_numbers(max_num_of_episodes)
        env_seeds_eval = generate_random_numbers(max_num_of_episodes * 10)

    current_agent = agent_selector(agent_type, config_agent)
    #TODO, add load model and load buffer function

    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False
    oracle_teaching_not_used = 0
    # For saving: Initialize history as a dictionary of empty lists
    history = {}
    
    start_time = time.time()

    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):   
        print("i_episode: ", i_episode)
        if environment_name in ['PushT']:
            env.set_seed(env_seeds[i_episode])
        if environment_name in ['robosuite']:
            env._evaluation = False
        
        observation, info = env.reset()

        episode_counter += 1
        h_counter = 0
        last_action = None
        h = None
        current_agent.evaluation = False
        
        training_metrics = {} # To store metrics like loss for this episode

        # ===== LOOP of each episode =====
        for t in range(1, max_time_steps_episode + 1):
            env.render_mode = 'human'
            if render_savefig_flag: env.render()
            elapsed = (time.time() - start_time - eval_time_acc)
            time_str = str(datetime.timedelta(seconds=elapsed))

            obs_proc = observation

            # === Pick the action to take: either from the robot action or the human action
            if not executed_human_correction:
                action_agent = current_agent.action(obs_proc)
                action = action_agent
            else:
                if h is not None and np.any(h):
                    action_agent = None
                    if agent_type in ['HG_DAgger','Implicit_BC','PVP','Diffusion']:
                        action = h
                    else:
                        action = last_action + np.matmul(current_agent.e, h)
                    action = np.clip(action, -1, 1)
                    h_counter += 1
                else:
                    action_agent = current_agent.action(obs_proc)
                    action = action_agent
            
            teacher_action = get_teacher_action(environment_name, observation, action_agent=action, env=env, policy_oracle=policy_oracle)
            observation, reward, done, _, info = env.step(action)
            env_done_fake, success_ep = is_env_done(info)
            done = done or env_done_fake or t == max_time_steps_episode
            last_action = action
            h = None
            
            # === Simulated teacher gives corrections 
            if oracle_teacher and ( (t % oracle_teaching_how_offen == 0) or oracle_teaching_not_used > 0):
                if action_agent is None:
                    action_agent = current_agent.action(obs_proc)
                if agent_type in ['HG_DAgger','Implicit_BC','PVP','Diffusion']:
                    if agent_algorithm in ['Diffusion_policy_relative','ibc_relative','pvp_relative']:
                        tmp_h, h_no_threshold = oracle_gimme_feedback(teacher_action, action_agent, None, config=config_general)
                        h = current_agent.e * np.array(tmp_h) + action_agent
                    else:
                        h, h_no_threshold = oracle_feedback_HGDagger(teacher_action, action_agent, None, config=config_general)
                else:
                    if agent_algorithm in ["CLIC_EBM", 'CLIC_Explicit']:
                        h, h_no_threshold = oracle_feedback_intervention_diff(teacher_action, action_agent, None, config=config_general)
                        current_agent.e = np.identity(current_agent.dim_a)
                    else:
                        h, h_no_threshold = oracle_gimme_feedback(teacher_action, action_agent, None, config=config_general)
                last_action = action_agent

                if not np.any(h) and t % oracle_teaching_how_offen == 0:  # if current feedback is not used
                    oracle_teaching_not_used = oracle_teaching_not_used + 1
                
                if np.any(h) and t % oracle_teaching_how_offen != 0: # if use a feedback not at the predefined time
                    oracle_teaching_not_used = oracle_teaching_not_used - 1

            if human_teacher and not oracle_teacher:
                h = feedback_receiver.get_h()
                if feedback_receiver.ask_for_done():
                    done = True
            
            # === Data collection & Policy update
            metrics = current_agent.collect_data_and_train(
                last_action=last_action if agent_type != 'Diffusion' else teacher_action , 
                h=h, obs_proc=obs_proc,
                next_obs=observation, t=t_total, done=done,
                agent_algorithm=agent_algorithm, agent_type=agent_type, i_episode=i_episode
            )
            
            # Aggregate metrics (e.g., average loss over the episode)
            if metrics:
                for key, value in metrics.items():
                    if key not in training_metrics:
                        training_metrics[key] = []
                    training_metrics[key].append(value)

            t_total += 1

            if done and environment_name in ['robosuite']:
                policy_oracle.reset()

            if done:
                # Save the policy & data buffer
                base_dir = config_agent['base_dir']
                current_agent.saved_dir = os.path.join(base_dir, f"repetition_{i_repetition:03d}/")
                current_agent.load_dir = current_agent.saved_dir
                if not os.path.exists(current_agent.saved_dir):
                    os.makedirs(current_agent.saved_dir)
                
                cumm_feedback += h_counter
                if save_policy:
                    current_agent.save_model()
                if save_policy and agent_type in ['HG_DAgger']:
                    current_agent.save_policy()
                if config_agent['save_buffer']:
                    pfile = os.path.join(current_agent.saved_dir, 'buffer_data.pkl')
                    current_agent.buffer.save_to_file(pfile)
                    with open(os.path.join(current_agent.saved_dir, 'data.txt'), 'w') as f:
                        f.write(f"Steps: {t_total}\nEpisode: {i_episode}\nTime: {time_str}\nFeedbacks: {cumm_feedback}\n")
                break

        # Evaluate agent periodically
        evalutation_frequency = 10 if i_episode <= 150 else 1
        
        if i_episode % evalutation_frequency == 0 or repetition_done:
            # Prepare the data dictionary for the current evaluation point
            current_data_point = {
                'Episode': episode_counter,
                'Timesteps': t_total,
                'time': str(datetime.timedelta(seconds=(time.time() - start_time - eval_time_acc))),
                'Amount of feedback': cumm_feedback
            }
            
            # Add aggregated training metrics to the data point
            for key, values in training_metrics.items():
                # Store the mean value for this episode
                current_data_point[f'mean_{key}'] = np.mean(values) if values else 0

            # Call the refactored evaluation and saving function
            eval_time_acc, history = evaluation_saving_results_process(
                eval_agent=current_agent, eval_env=env, policy_oracle=policy_oracle,
                i_episode=i_episode, i_repetition=i_repetition,
                max_steps=max_time_steps_episode, render_savefig_flag=render_savefig_flag,
                history=history, current_data=current_data_point, eval_time_acc=eval_time_acc,
                SEED_id=SEED, config_general=config_general, config_agent=config_agent
            )

        if repetition_done:
            break
            
    print(f"== Finished Repetition {i_repetition} with {episode_counter} episodes ==")

# =========================================================
#     TRAINING from Offline Dataset
# =========================================================
def train_offline_repetition(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent, render_savefig_flag = False):
    """Run one repetition of offline training."""
    SEED = 48 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    environment_name = config_general['environment']

    save_policy = config_agent['save_policy']
    agent_algorithm = config_agent['algorithm']
    agent_type = config_agent['agent']

    if environment_name in ['PushT']:
        env.set_seed(SEED)
        env_seeds = generate_random_numbers(max_num_of_episodes)

    current_agent = agent_selector(agent_type, config_agent)
    print("LOAD buffer")
    filename = current_agent.load_pretrained_dir + 'buffer_data.pkl'
    current_agent.buffer.load_from_file(filename)
    print("length of buffer: ", current_agent.buffer.length())

    t_total = 1
    episode_counter = 0
    cumm_feedback = 0
    eval_time_acc = 0
    repetition_done = False
    
    # For saving
    history = {}
    start_time = time.time()
    
    # ===== EPISODES LOOP (for training iterations) =====
    for i_episode in range(max_num_of_episodes):
        print("i_episode: ", i_episode)

        current_agent.evaluation = False
        # Train agent
        metrics = current_agent.collect_data_and_train(
            last_action=None, h=None, obs_proc=None, next_obs=None,
            t=t_total, done=True, agent_algorithm=agent_algorithm,
            agent_type=agent_type, i_episode=i_episode
        )
        
        base_dir = config_agent['base_dir']
        current_agent.saved_dir = os.path.join(base_dir, f"repetition_{i_repetition:03d}/")
        current_agent.load_dir = current_agent.saved_dir
        if not os.path.exists(current_agent.saved_dir):
            os.makedirs(current_agent.saved_dir)
        
        if save_policy:
            current_agent.save_model()
        if save_policy and agent_type in ['CLIC','HG_DAgger','Implicit_BC']:
            current_agent.save_policy() if agent_type != 'Implicit_BC' else current_agent.save_Q_value_model()
        
        # Evaluate agent periodically
        evalutation_frequency = 10 if i_episode <= 150 else 5
        
        if i_episode % evalutation_frequency == 0 or repetition_done:
            episode_counter = i_episode + 1
            current_data_point = {
                'Episode': episode_counter,
                'Timesteps': (i_episode + 1) * config_agent.get('batch_size', 256), # Approximate timesteps
                'time': str(datetime.timedelta(seconds=(time.time() - start_time - eval_time_acc))),
                'Amount of feedback': 0 # Offline
            }
            if metrics:
                for key, value in metrics.items():
                    current_data_point[key] = value # Store latest metric

            eval_time_acc, history = evaluation_saving_results_process(
                eval_agent=current_agent, eval_env=env, policy_oracle=policy_oracle,
                i_episode=i_episode, i_repetition=i_repetition,
                max_steps=max_time_steps_episode, render_savefig_flag=render_savefig_flag,
                history=history, current_data=current_data_point, eval_time_acc=eval_time_acc,
                SEED_id=SEED, config_general=config_general, config_agent=config_agent
            )

        if repetition_done:
            break
            
    print(f"== Finished Repetition {i_repetition} with {i_episode + 1} training iterations ==")

# =========================================================
#     Evaluating checkpoints
# =========================================================
def evalaution_without_training(env, policy_oracle, feedback_receiver, i_repetition, config_general, config_agent, render_savefig_flag=True):
    """Run one repetition of evaluating a pre-trained agent without training."""
    SEED = 100001 + i_repetition
    random.seed(SEED)
    np.random.seed(SEED)

    max_num_of_episodes = config_general['max_num_of_episodes']
    max_time_steps_episode = config_general['max_time_steps_episode']
    agent_type = config_agent['agent']
    
    eval_agent = agent_selector(agent_type, config_agent)
    
    base_dir = config_agent['base_dir']
    eval_agent.saved_dir = os.path.join(base_dir, config_agent['experiment_id'], f"repetition_{i_repetition:03d}/")
    if not os.path.exists(eval_agent.saved_dir):
        os.makedirs(eval_agent.saved_dir)

    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    full_path = os.path.join(project_root, eval_agent.load_dir)
    eval_agent.load_dir = full_path
    print("eval_agent.load_dir: ", eval_agent.load_dir)
    eval_agent.load_model()
    
    eval_time_acc = 0
    history = {}
    
    # ===== EPISODES LOOP =====
    for i_episode in range(max_num_of_episodes):
        print("i_episode: ", i_episode)
        
        current_data_point = {
            'Evaluation Episode': i_episode + 1,
        }

        # The SEED for the saving filename is based on the repetition, not the episode
        saving_seed = 48 + i_repetition
        
        eval_time_acc, history = evaluation_saving_results_process(
            eval_agent=eval_agent, eval_env=env, policy_oracle=policy_oracle,
            i_episode=i_episode, i_repetition=i_repetition,
            max_steps=max_time_steps_episode, render_savefig_flag=render_savefig_flag,
            history=history, current_data=current_data_point, eval_time_acc=eval_time_acc,
            SEED_id=saving_seed, config_general=config_general, config_agent=config_agent
        )

    print(f"== Finished Evaluation for Repetition {i_repetition} ==")


# run by using: python main_IIL.py --config-name train_CLIC_Explicit
@hydra.main(config_path="config", config_name="train_CLIC_Explicit")
def main(cfg: DictConfig):
    config_general  = cfg.GENERAL
    config_agent    = cfg.AGENT
    config_feedback = cfg.FEEDBACK
    config_task = cfg.task
    # set up env here
    env, policy_oracle, feedback_receiver = env_selector(config_general, config_feedback, config_task)

    """Main entry point: run multiple training repetitions."""
    print("Starting training...\n")
    for rep_idx in range(config_general['number_of_repetitions']):

        if config_agent['evaluate']:
            evalaution_without_training(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent, render_savefig_flag = config_general.get('render_savefig_flag', False))
        elif config_agent['offline_training']:
            train_offline_repetition(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent)
        else:
            train_single_repetition(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent, render_savefig_flag = config_general.get('render_savefig_flag', False))

        # test_teacher_policy(env, policy_oracle, feedback_receiver, rep_idx, config_general, config_agent)

if __name__ == "__main__":
    main()