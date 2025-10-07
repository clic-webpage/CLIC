import numpy as np
from tools.buffer import Buffer_uniform_sampling

import tensorflow as tf
import os

from tensorflow.keras.models import load_model

"""
Implementation of HG-DAgger 
"""

class HG_DAGGER:
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
        self.e = np.zeros((3,3)) # not used in HG-Dagger
        self.policy_model_learning_rate = policy_model_learning_rate

        self.use_CLIC_algorithm = False # not used in HG-Dagger
        # Initialize HG_DAgger buffer 
        self.buffer = Buffer_uniform_sampling(min_size=self.buffer_min_size, max_size=self.buffer_max_size)

        self.saved_dir = saved_dir  # used to save for the buffer & network models
        self.load_dir = load_dir
        self.load_policy_flag  = load_policy

        self.latested_data_pair = None # used to save the latest data pair for online training

        from agents.DP_model.neural_network import NeuralNetwork
        neural_network = NeuralNetwork(dim_a=self.dim_a,
                                dim_a_used=self.dim_a,
                                dim_o=self.dim_o)
        self.policy_model = neural_network.policy_model()
        self.optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)
        

    def action(self, state_representation):
        self.count += 1
        action = self.policy_model(state_representation)
        action = action.numpy()
        out_action = np.clip(action, -1, 1)
        return out_action.reshape(-1)  # (dim_a, )
    

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Policy(last_action, t, done, i_episode, h, obs_proc)

    def TRAIN_Policy(self, action, t, done, i_episode, h, observation):
        if np.any(h):  # if human teleoperates, also update the policy model
            # save the data pair to the buffer
            self.buffer.add([observation, h])
            self.latested_data_pair = [observation, h]
        
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                batch[-1] = self.latested_data_pair
                self._batch_update(batch)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._batch_update(batch)

        # if self.buffer.initialized() and (t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done)):
        if self.buffer.initialized() and ( (self.train_end_episode and done)):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length())

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self._batch_update(batch)

    def _batch_update(self, batch):
        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_batch_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [self.buffer_sampling_size, self.dim_o]),
                                                                  dtype=tf.float32)
        action_label_batch = [np.array(pair[1]) for pair in batch]
        # print(" action_label_batch: ",  action_label_batch)
        action_label_batch  = tf.convert_to_tensor(np.reshape(action_label_batch , [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        self._single_update(observations_batch_reshaped_tensor, action_label_batch)
    
    def _single_update(self, state_representation, policy_label):
        # TRAIN policy model
        with tf.GradientTape() as tape_policy:
            policy_output = self.policy_model([state_representation])
            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - policy_label))  # simple BC loss for explicit policies
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        return

    def save_model(self):
        self.save_policy()

    def save_policy(self, version = 0):
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        filename = f'policy_v{version}.h5'
        self.policy_model.save(os.path.join(network_saved_dir, filename))

        
    def load_policy_model(self, version_id = 0):
        network_load_dir = self.load_dir + 'network_params/'
        if not os.path.exists(network_load_dir):
            print("The load path does not exist: ", network_load_dir)
        else:
            filename = f'policy_v{version_id}.h5'
            if os.path.exists(os.path.join(network_load_dir, filename)):
                self.policy_model = load_model(os.path.join(network_load_dir, filename))
                print("Load policy model successfully: ", filename)
            else:
                print("No model found with version ID: ", version_id)