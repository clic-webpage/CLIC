import numpy as np

from tools.buffer import Buffer

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from agents.CLIC_tf import sample_action_from_Q_function, grad_penalty, make_counter_example_actions
"""
Implementation of 'Learning from Active Human Involvement through Proxy Value Propagation'
"""
class pvp:
    def __init__(self, dim_a, dim_o, action_upper_limits, action_lower_limits, e_matrix, loss_weight_inverse_e, 
            buffer_min_size, buffer_max_size,
            buffer_sampling_rate, buffer_sampling_size, train_end_episode, policy_model_learning_rate, 
            saved_dir, load_dir, load_policy, number_training_iterations):
        # Initialize variables
        self.h = None
        self.dim_a = dim_a
        self.dim_o = dim_o
        self.action_upper_limits = action_upper_limits
        self.action_lower_limits = action_lower_limits
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        self.number_training_iterations = number_training_iterations
        self.train_end_episode = train_end_episode
        self.e = 1 # not used in HG-Dagger
        self.policy_model_learning_rate = policy_model_learning_rate

        self.use_CLIC_algorithm = True # not used in HG-Dagger
        # Initialize HG_DAgger buffer 
        self.buffer = Buffer(min_size=self.buffer_min_size, max_size=self.buffer_max_size)
        self.buffer_no_intervention = Buffer(min_size=self.buffer_min_size, max_size=self.buffer_max_size)
        self.buffer_temporary = Buffer(min_size=1, max_size=self.buffer_max_size)

        self.saved_dir = saved_dir  # used to save for the buffer & network models
        self.load_dir = load_dir
        self.latested_data_pair = None # used to save the latest data pair for online training
        self.load_policy_flag = load_policy
        self.evaluation = False

        from agents.DP_model.neural_network import NeuralNetwork
        neural_network = NeuralNetwork(dim_a=self.dim_a,
                                dim_a_used=self.dim_a,
                                dim_o=self.dim_o)
        self.action_value_model = neural_network.action_value_function_model()
        self.action_value_model_target = neural_network.action_value_function_model()
        self.optimizer_action_value_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)


    def action_value_single_update_proxy_value_loss(self, observation, action, h_human, next_observation):
        # observatoin: [batch, dim_s]
        Q_s_a_target_negative = tf.constant(-1.0)
        Q_s_a_target_positive = tf.constant(1.0)

        with tf.GradientTape() as tape_policy:
            
            Q_s_a = self.action_value_model([observation, action]) # size: [sample_h_size * batch, 1]
            Q_s_a_plus_eh = self.action_value_model([observation, h_human])
           
            Q_s_sampleA_Postive = (Q_s_a_target_positive - Q_s_a_plus_eh)**2  # Q**2 used as regularization is better without regularization
            Q_s_sampleA_Negative =  (Q_s_a_target_negative - Q_s_a)**2 
           
            policy_loss =  Q_s_sampleA_Postive + Q_s_sampleA_Negative
            policy_loss = tf.reduce_mean(policy_loss)


            grads = tape_policy.gradient(policy_loss, self.action_value_model.trainable_variables)
        # clip the gradients
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]

        self.optimizer_action_value_model.apply_gradients(zip(grads, self.action_value_model.trainable_variables))


    def action_value_single_update_temporal_difference_loss(self, observation, action, next_observation):
        # observatoin: [batch, dim_s]
        sample_h_size = 128 # 512 better than 128
        next_observation_tiled = tf.tile(next_observation, [sample_h_size, 1])

        sampled_action, _ = make_counter_example_actions(self.action_value_model, next_observation_tiled, action.shape[0], action.shape[-1], sample_closer_to_optimal_a=True, sample_actions_size=sample_h_size)
        
        # print("sampled_action: ", sampled_action.shape)
        sampled_action = tf.reshape(sampled_action, [sample_h_size*action.shape[0], -1])

        # Q_s_a_target_tiled = self.action_value_model_target([next_observation_tiled, sampled_action])
        Q_s_a_target_tiled = self.action_value_model([next_observation_tiled, sampled_action])
        Q_s_a_target_tiled = tf.stop_gradient(Q_s_a_target_tiled)
        Q_s_a_target_tiled = tf.transpose(tf.reshape(Q_s_a_target_tiled, [sample_h_size, -1])) 

        # Q_s_a_target = tf.reduce_mean(Q_s_a_target_tiled, axis=-1)   # mean is worse than max
        Q_s_a_target = 0.99 * tf.reduce_max(Q_s_a_target_tiled, axis=-1)

        with tf.GradientTape() as tape_policy:
            Q_s_a_target_tiled = self.action_value_model([next_observation_tiled, sampled_action])

            Q_s_a = self.action_value_model([observation, action])

            policy_loss = tf.reduce_mean( (Q_s_a_target - Q_s_a)**2)      
            q_value_penalty_strength = 0.01  # You can adjust this value
            q_value_penalty = q_value_penalty_strength * tf.nn.l2_loss(tf.square(Q_s_a_target_tiled))

            policy_loss = policy_loss + q_value_penalty
            grads = tape_policy.gradient(policy_loss, self.action_value_model.trainable_variables)
        # clip the gradients
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]

        self.optimizer_action_value_model.apply_gradients(zip(grads, self.action_value_model.trainable_variables))


    def action_value_batch_update_pvp(self, batch):
        state_batch = [np.array(pair[0]) for pair in batch]  
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]  
        next_state_batch = [np.array(pair[3]) for pair in batch]

        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        next_observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(next_state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)

        self.action_value_single_update_proxy_value_loss(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor, next_observation_reshaped_tensor)


    def action_value_batch_update_TD(self, batch): 
        state_batch = [np.array(pair[0]) for pair in batch]  
        action_batch = [np.array(pair[1]) for pair in batch]
        next_state_batch = [np.array(pair[2]) for pair in batch]
  
        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        next_observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(next_state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)

        self.action_value_single_update_temporal_difference_loss(observation_reshaped_tensor, action_reshaped_tensor, next_observation_reshaped_tensor)


    def action(self, state_representation):
        self.count += 1

        action = sample_action_from_Q_function(self.action_value_model, state_representation, 1, 1024, self.evaluation, self.dim_a)

        action = action.numpy()
        out_action = []
        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) 
            out_action.append(action[0, i])

        return np.array(out_action)


    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py """
        return self.TRAIN_Q_value(last_action, h, obs_proc, next_obs, t, done)   


    def TRAIN_Q_value(self, action, h, observation, next_observation,  t, done):
        if np.any(h):  # if any element is not 0
            self.buffer.add([observation, action, h, next_observation])
            if self.buffer.initialized():
                for i in range(10):
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    # include the new data in this batch
                    batch[-1] = [observation, action, h, next_observation]
                    self.action_value_batch_update_pvp(batch)
                if self.buffer_no_intervention.initialized():
                    batch = self.buffer_no_intervention.sample(batch_size=self.buffer_sampling_size)
                    self.action_value_batch_update_TD(batch)
        else:
            # save the non-intervention data. This data is used for Temporal difference loss. 
            self.buffer_no_intervention.add([observation, action, next_observation])

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            print("train q value")
            for i in range(10): 
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.action_value_batch_update_pvp(batch)
            if self.buffer_no_intervention.initialized():
                batch = self.buffer_no_intervention.sample(batch_size=self.buffer_sampling_size)
                self.action_value_batch_update_TD(batch)

        if done:
            self.last_action = None

        if self.buffer.initialized() and (self.train_end_episode and done):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length(), " buffer_no_intervention size: ", self.buffer_no_intervention.length())
                    
                for i in range(10): 
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.action_value_batch_update_pvp(batch)
                if self.buffer_no_intervention.initialized():
                    batch = self.buffer_no_intervention.sample(batch_size=self.buffer_sampling_size)
                    self.action_value_batch_update_TD(batch)


    def save_policy(self):
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        self.policy_model.save(network_saved_dir + 'policy.h5')

    def load_policy_model(self):
        network_load_dir = self.load_dir + 'network_params/'
        if not os.path.exists(network_load_dir):
            print("The load path does not exist: ", network_load_dir)
        else:
            self.policy_model = load_model(network_load_dir + 'policy.h5')
            print("load policy model successfully! ")

    def save_Q_value_model(self):
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        self.action_value_model.save(network_saved_dir + 'Q_value_model.h5')

    def load_Q_value_model(self):
        network_load_dir = self.load_dir + 'network_params/'
        if not os.path.exists(network_load_dir):
            print("The load path does not exist: ", network_load_dir)
        else:
            self.action_value_model = load_model(network_load_dir + 'Q_value_model.h5')
            print("load policy model successfully! ")
