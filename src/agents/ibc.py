import numpy as np

from tools.buffer import Buffer

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random
from agents.CLIC_tf import sample_action_from_Q_function, grad_penalty, make_counter_example_actions

"""
Implementation of Implicit BC, mainly copied from https://github.com/google-research/ibc
"""
class ibc:
    def __init__(self, dim_a, dim_o, action_upper_limits, action_lower_limits, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, number_training_iterations, train_end_episode, policy_model_learning_rate, 
                 saved_dir, load_dir, load_policy):
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
        self.e = 0.2 # not used in HG-Dagger
        self.policy_model_learning_rate = policy_model_learning_rate

        self.use_CLIC_algorithm = False # not used in HG-Dagger
        # Initialize HG_DAgger buffer 
        self.buffer = Buffer(min_size=self.buffer_min_size, max_size=self.buffer_max_size)

        self.saved_dir = saved_dir  # used to save for the buffer & network models
        self.load_dir = load_dir

        self.latested_data_pair = None # used to save the latest data pair for online training

        self.load_policy_flag = load_policy
        self.evaluation = False
        self.sampled_action_last = None

        from agents.DP_model.neural_network import NeuralNetwork
        neural_network = NeuralNetwork(dim_a=self.dim_a,
                                dim_a_used=self.dim_a,
                                dim_o=self.dim_o)
        self.action_value_model = neural_network.action_value_function_model()
        # '''test new action_value_model'''
        # self.action_value_model = neural_network.action_value_model_resnet()
        self.optimizer_action_value_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)

    
    def action_value_single_update_ibc_with_info_nce(self, observation, h_human, softmax_temperature=0.1, action_negative = None):
        sample_h_size = 512  # number of sampled actions

        # sample_h_size = 128  # For real kuka exps
        batch_size = observation.shape[0]
        observation_tiled = tf.tile(observation, [sample_h_size, 1])

        sampled_action, _ = make_counter_example_actions(self.action_value_model, observation_tiled, h_human.shape[0], h_human.shape[-1], sample_actions_size=sample_h_size)

        sampled_action = tf.concat([tf.expand_dims(h_human, 0), 
                            sampled_action[1:, :, :]], axis=0)   # [sample_h_size, batch_size, dim_a]

        sampled_action = tf.reshape(sampled_action, [sample_h_size*h_human.shape[0], -1])

        self.sampled_action_last = sampled_action
        Q_s_a_target = self.action_value_model([observation_tiled, sampled_action])
        Q_s_a_target = tf.stop_gradient(Q_s_a_target) 
        
        h_human_tiled = tf.tile(h_human, [sample_h_size, 1])

        # Create a condition vector with all zeros
        condition_vector = tf.zeros([sample_h_size, batch_size], dtype=tf.float32)

        # Set the first row (corresponding to tf.expand_dims(h_human, 0)) to 1
        condition_vector = tf.concat([tf.expand_dims(tf.ones([batch_size], dtype=tf.float32), 0),
                                      condition_vector[1:, :]], axis=0)  # [sample_h_size, batch_size]
        # Backpropagate the loss
        with tf.GradientTape() as tape:
            # Compute similarity condition
            log_prob_pi_a_plus_eh = -tf.reduce_sum((sampled_action - h_human_tiled) ** 2, axis=1, keepdims=True)

            condition = -log_prob_pi_a_plus_eh < tf.constant(0.0001)  # True for "positive" actions
            condition = tf.reshape(condition, [sample_h_size, batch_size])
            condition = tf.transpose(condition)
            condition = tf.cast(condition, dtype=tf.float32)

            # Compute Q-values for all actions
            Q_s_a = self.action_value_model([observation_tiled, sampled_action])
            Q_s_a_reshaped = tf.reshape(Q_s_a, [sample_h_size, -1])
            Q_s_a_reshaped = tf.transpose(Q_s_a_reshaped)
            predictions = Q_s_a_reshaped

            # Adjust labels based on condition, marking both true and sufficiently close actions as "positive"
            labels = condition
            labels /= tf.reduce_sum(labels, axis=1, keepdims=True)  # Normalize to ensure sum to 1

            # Calculate dynamic InfoNCE Loss
            softmaxed_predictions = tf.nn.softmax(predictions / softmax_temperature, axis=1)
            kl_divergence = tf.keras.losses.KLDivergence()
            per_example_loss = kl_divergence(labels, softmaxed_predictions)

            #  # Add penalty for large Q-values
            # q_value_penalty = q_value_penalty_strength * tf.nn.l2_loss(tf.square(predictions))
            grad_penalty_loss = grad_penalty(self.action_value_model, batch_size, observation_tiled, sampled_action)
            grad_penalty_loss = tf.reduce_mean(grad_penalty_loss)
            per_example_loss = grad_penalty_loss + per_example_loss
            
            grads = tape.gradient(per_example_loss, self.action_value_model.trainable_variables)

        # clip the gradients
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]

        self.optimizer_action_value_model.apply_gradients(zip(grads, self.action_value_model.trainable_variables))


    def action_value_batch_update_ibc(self, batch):
        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        h_human_batch = [np.array(pair[1]) for pair in batch]  # last


        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)

        self.action_value_single_update_ibc_with_info_nce(observation_reshaped_tensor, h_human_reshaped_tensor)


    def action(self, state_representation):
        self.count += 1
        # sample from the EBM to obtain action that minimize this EBM (through MCMC sampling).
        # Note: avoid using 'iterative_dfo' as this function only works in low-dim action space.
        action = sample_action_from_Q_function(self.action_value_model, state_representation, 1, 512, self.evaluation, self.dim_a)
        action = action.numpy()

        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) 
            out_action.append(action[0, i])

        return np.array(out_action)

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Policy(last_action, t, done, i_episode, h, obs_proc)
    
    def TRAIN_Policy(self, action, t, done, i_episode, h, observation):
        if np.any(h):  # if human teleoperates, also update the policy model
            # save the data pair to the buffer

            # in HG-Dagger or IBC, h is defined as the teacher action
            # also save the negative action, but should also works for the code that without having negative action
            self.buffer.add([observation, h, action])
            self.latested_data_pair = [observation, h, action]
        
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                batch[-1] = self.latested_data_pair
                self.action_value_batch_update_ibc(batch)

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self.action_value_batch_update_ibc(batch)


        # if self.buffer.initialized() and (t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done)):
        if self.buffer.initialized() and ( (self.train_end_episode and done)):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print("train ibc")
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length())

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.action_value_batch_update_ibc(batch)

    def save_model(self):
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        self.action_value_model.save(network_saved_dir + 'IBC_model.h5')

    def loads_model(self, neural_network):
        network_load_dir = self.load_dir + 'network_params/'
        if not os.path.exists(network_load_dir):
            self.action_value_model = neural_network.action_value_function_model()
            print("The load path does not exist: ", network_load_dir)
        else:
            model_filename = network_load_dir + 'IBC_model.h5'
            if os.path.exists(model_filename):
                self.action_value_model = load_model(network_load_dir + 'IBC_model.h5')
                print("load policy model successfully! ")
            else:
                self.action_value_model = neural_network.action_value_function_model()
                print(f"Model file '{model_filename}' not found. Skipping model loading.")
