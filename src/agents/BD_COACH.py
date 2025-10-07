import numpy as np

import tensorflow as tf

import os

from tools.buffer import Buffer_uniform_sampling

"""
D-COACH implementation
"""

import tensorflow as tf
from tensorflow.keras.models import load_model

class BD_COACH:
    def __init__(self, dim_a, dim_o, action_upper_limits, action_lower_limits, e_matrix, loss_weight_inverse_e, 
                buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, train_end_episode, policy_model_learning_rate, human_model_learning_rate, 
                 saved_dir, load_dir, load_policy, number_training_iterations):
        # Initialize variables
        self.h = None
        self.state_representation = None
        self.policy_action_label = None
        #self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a
        self.dim_o = dim_o
        self.action_lower_limits = action_lower_limits
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.train_end_episode = train_end_episode
        self.policy_model_learning_rate = policy_model_learning_rate
        self.human_model_learning_rate = human_model_learning_rate
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        self.use_CLIC_algorithm = False 
        
        self.e = np.diag(e_matrix)
        self.loss_weight_inverse_e = np.diag(loss_weight_inverse_e)

        self.buffer = Buffer_uniform_sampling(min_size=self.buffer_min_size, max_size=self.buffer_max_size)
        self.policy_loss_list = []

        self.test_New_value_function_idea = False

        self.saved_dir = saved_dir  # save buffer data & network models
        self.load_dir = load_dir
        self.load_policy_flag = load_policy  # whether load the previous buffer & network models
        self.number_training_iterations = number_training_iterations
        self.time_step = None
        self.evaluation = False

        self.list_action_tobeQueried = []
        self.last_action = True

        # used for visualization in toy example
        self.sampled_action_last = None

        from agents.DP_model.neural_network import NeuralNetwork
        neural_network = NeuralNetwork(dim_a=self.dim_a,
                                dim_a_used=self.dim_a,
                                dim_o=self.dim_o)
        self.policy_model = neural_network.policy_model()
        self.optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)
        self.Human_model = neural_network.Human_model()
        self.optimizer_Human_model = tf.keras.optimizers.Adam(learning_rate=self.human_model_learning_rate)

    def _generate_policy_label(self, action, h):
        error =  np.matmul(self.e, h)
        error = np.array(error).reshape(1, self.dim_a)
        policy_action_label = []

        for i in range(self.dim_a):
            policy_action_label.append(np.clip(action[i] + error[0, i], -1, 1))

        policy_action_label = np.array(policy_action_label).reshape(1, self.dim_a)

        return policy_action_label

    def _generate_batch_policy_label(self, action_batch, h_predicted_batch):
        multi = np.matmul(np.asarray(h_predicted_batch), self.e)
        error = multi.reshape(-1, self.dim_a)
        a_target_batch = []

        for i in range(self.buffer_sampling_size):

            a_target_batch.append(np.clip(action_batch[i] + error[i], -1, 1))

        a_target_batch = np.array(a_target_batch).reshape(-1, self.dim_a)
        return a_target_batch


    def _single_update(self, state_representation, policy_label):
        with tf.GradientTape() as tape_policy:

            policy_output = self.policy_model([state_representation])
            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - policy_label))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        self.optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))
        return


    def _batch_update(self, batch, i_episode, t):
        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_batch_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [self.buffer_sampling_size, self.dim_o]),
                                                                  dtype=tf.float32)
        action_label_batch = [np.array(pair[1]) for pair in batch]
        action_label_batch  = tf.convert_to_tensor(np.reshape(action_label_batch , [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)

        self._single_update(observations_batch_reshaped_tensor, action_label_batch)


    def _policy_batch_update_with_HM(self, batch):
        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [self.buffer_sampling_size, self.dim_o]),
                                                            dtype=tf.float32)
        actions_batch_no_gradient = self.policy_model([observations_reshaped_tensor])
        with tf.GradientTape() as tape_policy:
            # policy_output = self.policy_model([state_representation])
            actions_batch = self.policy_model([observations_reshaped_tensor])

            # 5. Get bath of h predictions from Human model
            h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch_no_gradient])

            # h_predicted_batch = self.discretize_feedback(h_predicted_batch)

            # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
            a_target_batch = self._generate_batch_policy_label(actions_batch_no_gradient, h_predicted_batch)

            # 7. Update policy indirectly from Human model

            policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        self.optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))

    # current feedback are used to update the policy together with other data sampled from the batch
    # current feedback = (state, action, action_label)
    def _policy_batch_update_with_HM_and_currentFB(self, batch, current_feedback):

        batch_for_actor = batch + [current_feedback]
        observations_batch = [np.array(pair[0]) for pair in batch_for_actor]  # state(t) sequence
        observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [self.buffer_sampling_size + 1, self.dim_o]),
                                                            dtype=tf.float32)

        action_label = current_feedback[-1]

        with tf.GradientTape() as tape_policy:
            # policy_output = self.policy_model([state_representation])
            actions_batch = self.policy_model([observations_reshaped_tensor])

            # 5. Get bath of h predictions from Human model
            h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch])

            # h_predicted_batch = self.discretize_feedback(h_predicted_batch)

            # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
            a_target_batch = self._generate_batch_policy_label(actions_batch, h_predicted_batch)
            a_target_batch = np.vstack((a_target_batch, action_label))
            # 7. Update policy indirectly from Human model

            policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        self.optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))


    def Human_single_update(self, observation, action, h_human):
        # TRAIN Human model
        with tf.GradientTape() as tape_policy:

            h_predicted = self.Human_model([observation, action])
            #print("h_human: ", h_human)
            #print("h_predicted: ", h_predicted)
            policy_loss = 0.5 * tf.reduce_mean(tf.square(h_human- h_predicted))
            grads = tape_policy.gradient(policy_loss, self.Human_model.trainable_variables)

        self.optimizer_Human_model.apply_gradients(zip(grads, self.Human_model.trainable_variables))

        return

    def Human_batch_update(self, batch):
        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]  # last
        #print("h_human_batch: ",h_human_batch)

        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)

        self.Human_single_update(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor)
   
    def action(self, state_representation):
        action = self.policy_model(state_representation)
        action = action.numpy()
        out_action = np.clip(action, -1, 1)
        return out_action.reshape(-1)  # (dim_a, )

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Policy(last_action, t, done, i_episode, h, obs_proc)

    def TRAIN_Policy(self, action,t, done,i_episode, h, observation):
        if np.any(h):  # if any element is not 0
            # 1. append  (o_t, a_t, h_t) to D
            print("h: ", h)
            self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]), dtype=tf.float32)
            self.buffer.add([observation, action, self.h_to_buffer])
            # 2. Generate a_target
            action_label = self._generate_policy_label(action, h)

            # # # 3. Update policy with current observation and a_target
            # remove the soft start of the policy seems make BD-COACH converge a little bit faster with simulated teacher
            if not self.buffer.initialized():
                self._single_update(observation, action_label)

            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                # include the new data in this batch
                batch[-1] = [observation, action, self.h_to_buffer]
                self.Human_batch_update(batch)

                # 4. Batch update of the policy with the Human Model
                self._policy_batch_update_with_HM_and_currentFB(batch, [observation, action, action_label])  # use this seems more stable for learning (but the time to converge is quite similar)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            print("train BD-COACH")
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self.Human_batch_update(batch)

            # Batch update of the policy with the Human Model
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._policy_batch_update_with_HM(batch)

        if self.buffer.initialized() and (self.train_end_episode and done):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length())
                for i in range(3):
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.Human_batch_update(batch)

                # Batch update of the policy with the Human Model
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self._policy_batch_update_with_HM(batch)
    
    def save_model(self):
        self.save_policy()
        self.save_human_model()

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

    def save_human_model(self):
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        self.Human_model.save(network_saved_dir + 'human_model.h5')

    def load_human_model(self):
        network_load_dir = self.load_dir + 'network_params/'
        if not os.path.exists(network_load_dir):
            print("The load path does not exist: ", network_load_dir)
        else:
            self.Human_model = load_model(network_load_dir + 'human_model.h5')
            print("load policy model successfully! ")

    
    

    


