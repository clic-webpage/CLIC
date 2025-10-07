import numpy as np

import tensorflow as tf 
from tensorflow.keras.models import load_model

import os
from tools.buffer import Buffer_uniform_sampling
"""
Tensorflow implementation of CLIC-Explicit
"""
class CLIC_Explicit:
    def __init__(self, dim_a, dim_o, action_upper_limits, action_lower_limits, e_matrix, loss_weight_inverse_e, 
                 sphere_alpha, sphere_gamma, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, train_end_episode, policy_model_learning_rate, 
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
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        self.use_CLIC_algorithm = True # use CLIC method
        
        self.e = np.diag(e_matrix)
        self.loss_weight_inverse_e = np.diag(loss_weight_inverse_e)

        self.sphere_alpha = sphere_alpha    # parameter that adjusts where to sample counterexamples on the sphere
        self.sphere_gamma = sphere_gamma    # parameter that adjust the center of the sphere to samle counterexamples

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


    def _single_update(self, state_representation, policy_label):
        # TRAIN policy model with BC loss
        with tf.GradientTape() as tape_policy:
            policy_output = self.policy_model([state_representation])
            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - policy_label))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        self.optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        return

    # Train policy model with CLIC 
    # The effect of this loss is to push the policy output towards the desired action space
    def _policy_single_update_contrastive_sphere(self, observation, action, h_human, ee_pose = None, observation_2=None):
        sample_h_size = 256  # Number of sampled contrastive action pairs
        alpha = self.sphere_alpha * np.pi / 180
        sphere_center_gamma = self.sphere_gamma

        with tf.GradientTape() as tape_policy:

            action_output = self.policy_model([observation])
            action_achor = action + sphere_center_gamma * tf.matmul(h_human, self.e)
            log_prob_pi_a = -tf.reduce_sum(tf.matmul((action_output - action) ** 2 , self.loss_weight_inverse_e), axis=1, keepdims=True) 
            log_prob_pi_a_plus_eh = -tf.reduce_sum(tf.matmul((action_output - action - tf.matmul(h_human, self.e)) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)

            log_prob_pi_a_plus_eh_gamma = -tf.reduce_sum(tf.matmul((action_output - action - sphere_center_gamma* tf.matmul(h_human, self.e)) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)

            policy_loss = 0.2 * tf.maximum(0.0, log_prob_pi_a - log_prob_pi_a_plus_eh_gamma) 

            if self.dim_a > 1:
                # Sample contrastive action pairs: Build Polytope Desired action spaces via multiple Desired Half-spaces
                v_combined = self.sample_vectors_from_sphere(h_human, sample_h_size, alpha)

                normalized_random_hs = tf.reshape(v_combined, [-1, h_human.shape[-1]]) * (1- sphere_center_gamma) * 2

                tiled_action_output = tf.tile(action_output, [sample_h_size , 1])
                tiled_action_achor= tf.tile(action_achor, [sample_h_size, 1])

                log_prob_a_counterfactuals =  -tf.reduce_sum(tf.matmul((tiled_action_output - tiled_action_achor -  tf.matmul(normalized_random_hs, self.e)) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)
                tiled_log_prob_pi_a_plus_eh= tf.tile(log_prob_pi_a_plus_eh, [sample_h_size, 1])
                policy_loss += 0.02 * (128.0 / sample_h_size) * tf.reduce_sum(tf.maximum(0.0, log_prob_a_counterfactuals - tiled_log_prob_pi_a_plus_eh))               

            policy_loss = tf.reduce_mean(policy_loss)
            print("policy loss: ", policy_loss)  
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        self.optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))


    # combine the CLIC-D and CLIC-C, the e is also saved into the buffer (being selected by the human teacher)
    # correction_e is 0.1 for CLIC-C and 1 for CLIC-D
    def _policy_single_update_contrastive_sphere_combined_relative_absolute(self, observation, action, h_human, correction_e):
        # sample negative samples from the sphere, with the radius of 0.5*||h_human||*e, the center is a + 0.5 * h_human
        # the uncertainty of the optimality of the correction is given by hyperparameter alpha, which is the angle between the correction and the negative sample-action
        # log_prob_pi_a = -(Pi(s) - a)^2
        sample_h_size = 128
        # sample_h_size = 32  # dim_a = 4
        # sample_h_size = 6  # for testing
        alpha = self.sphere_alpha * np.pi / 180
        sphere_center_gamma = self.sphere_gamma
        # print("self.loss_weight_inverse_e: ", self.loss_weight_inverse_e)
        v_combined = self.sample_vectors_from_sphere(h_human, sample_h_size, alpha)
        
        with tf.GradientTape() as tape_policy:
            action_output = self.policy_model([observation])
            # max(0, (Pi(s) - a)^2
            # to do (1) remove the minus one
            # (2) add a margin to the loss, the larger the margion,, the closer to the achor point

            # action: [batch, dim_a]; e: [dim_a, dim_a]; h_human: [batch, dim_a]
            action_achor = action + sphere_center_gamma * tf.multiply(h_human, correction_e)
            # print("tf.multiply(h_human, correction_e): ", tf.multiply(h_human, correction_e), " h_human: ", h_human)
            # log_prob_pi_a_achor = -tf.reduce_sum(tf.matmul((action_output - action_achor) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)  # [batch, 1]
            log_prob_pi_a = -tf.reduce_sum(tf.matmul((action_output - action) ** 2 , self.loss_weight_inverse_e), axis=1, keepdims=True) 
            log_prob_pi_a_plus_eh = -tf.reduce_sum(tf.matmul((action_output - action - tf.multiply(h_human, correction_e)) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)
            log_prob_pi_a_minus_eh = -tf.reduce_sum(tf.matmul((action_output - action + tf.multiply(h_human, correction_e)) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)
            # add a margin to the loss, which is equal to ||e* h||^2
            margin = tf.reduce_sum((tf.multiply(h_human, correction_e)) ** 2, axis=1, keepdims=True)
            
            log_prob_pi_a_plus_eh_gamma = -tf.reduce_sum(tf.matmul((action_output - action - sphere_center_gamma* tf.multiply(h_human, correction_e)) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)

            policy_loss = 0.02 * tf.maximum(0.0, log_prob_pi_a - log_prob_pi_a_plus_eh_gamma) 
            # reshape v_combined to [sample_h_size * 2, dim_a]
            normalized_random_hs = tf.reshape(v_combined, [-1, h_human.shape[-1]]) * (1- sphere_center_gamma) * 2

            tiled_action_output = tf.tile(action_output, [sample_h_size , 1])
            tiled_action_achor= tf.tile(action_achor, [sample_h_size, 1])
            tiled_correction_e = tf.tile(correction_e, [sample_h_size, 1])

            log_prob_a_counterfactuals =  -tf.reduce_sum(tf.matmul((tiled_action_output - tiled_action_achor -  tf.multiply(normalized_random_hs, tiled_correction_e)) ** 2, self.loss_weight_inverse_e), axis=1, keepdims=True)
            tiled_log_prob_pi_a_plus_eh= tf.tile(log_prob_pi_a_plus_eh, [sample_h_size, 1])

            policy_loss += 0.02 * tf.reduce_sum(tf.maximum(0.0, log_prob_a_counterfactuals - tiled_log_prob_pi_a_plus_eh))

            policy_loss = tf.reduce_mean(policy_loss)
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        self.optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        
    def sample_equally_on_sphere(self, sample_h_size):  # only for dim_a - 1 = 3
        indices = tf.range(0, sample_h_size, dtype=tf.float32)
        phi = tf.math.acos(1 - 2 * (indices + 0.5) / sample_h_size)
        golden_ratio = (1 + tf.sqrt(tf.constant(5, dtype=tf.float32))) / 2
        theta = 2 * np.pi * indices / golden_ratio

        x = tf.sin(phi) * tf.cos(theta)
        y = tf.sin(phi) * tf.sin(theta)
        z = tf.cos(phi)

        vectors = tf.stack([x, y, z], axis=-1)
        vectors = tf.reshape(vectors, (sample_h_size, 3))
        return vectors
    

    def sample_equally_on_1d(self,sample_h_size):
        vectors = np.array([[1.0], [-1.0]])
        
        # Select a specific number of vectors to match the sample_h_size
        vectors = np.tile(vectors, (sample_h_size // 2, 1))
        
        # If sample_h_size is odd, append one more vector to match the required size
        if sample_h_size % 2 != 0:
            vectors = np.vstack([vectors, np.array([[1.0]])])
        return tf.convert_to_tensor(vectors, dtype=np.float32)


    def sample_orthogonal_vectors(self, h_human, sample_h_size):
        # Get the shape of h_human
        batch_human, dim_a = h_human.shape.as_list()

        # Normalize h_human to make each vector a unit vector
        h_human_normalized = tf.nn.l2_normalize(h_human, axis=-1)

        # Initialize an empty list to hold the basis vectors
        basis_vectors = []

        # Use Gram-Schmidt process to generate basis vectors
        for i in range(dim_a - 1):  # We need dim_a - 1 vectors to define the plane
            # Generate a random vector
            random_vector = tf.random.normal(shape=(batch_human, dim_a))
            # Subtract the component that is in the direction of h_human
            projection = tf.reduce_sum(random_vector * h_human_normalized, axis=1, keepdims=True)
            random_vector -= projection * h_human_normalized
            # Orthonormalize with respect to the existing basis vectors
            for basis_vector in basis_vectors:
                projection = tf.reduce_sum(random_vector * basis_vector, axis=1, keepdims=True)
                random_vector -= projection * basis_vector
            # Normalize the new basis vector
            random_vector = tf.nn.l2_normalize(random_vector, axis=-1)
            # Add the new basis vector to the list
            basis_vectors.append(random_vector)

        # Stack the basis vectors
        basis_vectors = tf.stack(basis_vectors, axis=1)

        # Now, sample uniformly from the space spanned by these basis vectors
        # shape of coefficients:  (10, 16, 2)  shape of basis_vectors:  (10, 2, 3)
        coefficients = tf.random.normal([sample_h_size, dim_a - 1])
        coefficients = tf.linalg.normalize(coefficients, axis=1)[0]
        coefficients = tf.reshape(coefficients, (1, sample_h_size, dim_a - 1))
        coefficients = tf.tile(coefficients, [batch_human, 1, 1])
        # coefficients = self.sample_equally_on_sphere(batch_human, sample_h_size)

        # Combine them to get the vectors in the orthogonal space
        # print("shape of coefficients: ", coefficients.shape, " shape of basis_vectors: ", basis_vectors.shape)
        sampled_vectors = tf.matmul(coefficients, basis_vectors)

        # sampled_vectors = tf.reshape(sampled_vectors, [sample_h_size,batch_human, dim_a])
        sampled_vectors = tf.transpose(sampled_vectors, perm=[1, 0, 2])  # important, do not use reshape!  [sample_h_size,batch_human, dim_a]

        h_human_expanded = tf.expand_dims(h_human, 0)  # Expand dims to [1, batch_human, dim_a]
        h_human_tiled = tf.tile(h_human_expanded, [sample_h_size, 1, 1])  # Now tile it

        norm_h_human_tiled = tf.norm(h_human_tiled, axis=-1, keepdims=True)
        sampled_vectors = tf.nn.l2_normalize(sampled_vectors, axis=-1) * norm_h_human_tiled
        sampled_vectors = tf.reshape(sampled_vectors, [-1, h_human_tiled.shape[-1]])
        return sampled_vectors
    
    def generate_orthogonal_basis_if_demonstrationData(self, h_human, sample_h_size, sphere_center_gamma):
            # Get the shape of h_human
        batch_human, dim_a = h_human.shape.as_list()
        # Normalize h_human to make each vector a unit vector
        h_human_normalized = tf.nn.l2_normalize(h_human, axis=-1)
        # Initialize an empty list to hold the basis vectors
        basis_vectors = []

        # Use Gram-Schmidt process to generate basis vectors
        for i in range(dim_a - 1):  # We need dim_a - 1 vectors to define the plane
            # Generate a random vector
            random_vector = tf.random.normal(shape=(batch_human, dim_a))
            # Subtract the component that is in the direction of h_human
            projection = tf.reduce_sum(random_vector * h_human_normalized, axis=1, keepdims=True)
            random_vector -= projection * h_human_normalized
            # Orthonormalize with respect to the existing basis vectors
            for basis_vector in basis_vectors:
                projection = tf.reduce_sum(random_vector * basis_vector, axis=1, keepdims=True)
                random_vector -= projection * basis_vector
            # Normalize the new basis vector
            random_vector = tf.nn.l2_normalize(random_vector, axis=-1)
            # Add the new basis vector to the list
            basis_vectors.append(random_vector)
        basis_vectors.append(h_human_normalized)

        h_human_expanded = tf.expand_dims(h_human, 0)  # Expand dims to [1, batch_human, dim_a]
        h_human_tiled = tf.tile(h_human_expanded, [sample_h_size, 1, 1])  # Now tile it
        norm_h_human_tiled = tf.norm(h_human_tiled, axis=-1, keepdims=True)
        basis_vectors = tf.stack(basis_vectors, axis=1) 
        combined_basis_vectors = tf.concat([basis_vectors, -basis_vectors], axis=1) 
        combined_basis_vectors = tf.transpose(combined_basis_vectors, perm=[1, 0, 2])
        combined_basis_vectors = combined_basis_vectors * norm_h_human_tiled
        # shift the center to a minus
        combined_basis_vectors = combined_basis_vectors * sphere_center_gamma + h_human_tiled
        return combined_basis_vectors

    def sample_vectors_from_sphere(self, h_human, sample_h_size, alpha):
        # Get the shape of h_human
        batch_human, dim_a = h_human.shape.as_list()

        if dim_a == 1:
            raise NotImplementedError("The case where dim_a == 1 is not implemented.")

        # Normalize h_human to make each vector a unit vector
        h_human_normalized = tf.nn.l2_normalize(h_human, axis=-1)

        # Initialize an empty list to hold the basis vectors
        basis_vectors = []

        # Use Gram-Schmidt process to generate basis vectors
        for i in range(dim_a - 1):  # We need dim_a - 1 vectors to define the plane
            # Generate a random vector
            # random_vector = tf.random.normal(shape=(batch_human, dim_a))
            random_vector = tf.one_hot(i, dim_a, dtype=tf.float32)
            random_vector = tf.tile(tf.expand_dims(random_vector, 0), [batch_human, 1])
            # Subtract the component that is in the direction of h_human
            projection = tf.reduce_sum(random_vector * h_human_normalized, axis=1, keepdims=True)
            random_vector -= projection * h_human_normalized
            # Orthonormalize with respect to the existing basis vectors
            for basis_vector in basis_vectors:
                projection = tf.reduce_sum(random_vector * basis_vector, axis=1, keepdims=True)
                random_vector -= projection * basis_vector
            # Normalize the new basis vector
            random_vector = tf.nn.l2_normalize(random_vector, axis=-1)
            # Add the new basis vector to the list
            basis_vectors.append(random_vector)
        # Stack the basis vectors
        basis_vectors = tf.stack(basis_vectors, axis=1)

        # Now, sample uniformly from the space spanned by these basis vectors
        if dim_a - 1 == 3:
            coefficients = self.sample_equally_on_sphere(sample_h_size)
        elif dim_a - 1 == 1:
            coefficients = self.sample_equally_on_1d(sample_h_size)
        else:
            coefficients = tf.random.normal([sample_h_size, dim_a - 1])
            coefficients = tf.linalg.normalize(coefficients, axis=1)[0]

        coefficients = tf.reshape(coefficients, (1, sample_h_size, dim_a - 1))
        coefficients = tf.tile(coefficients, [batch_human, 1, 1])

        sampled_vectical_vectors = tf.matmul(coefficients, basis_vectors)

        sampled_vectical_vectors = tf.transpose(sampled_vectical_vectors, perm=[1, 0, 2])  # important, do not use reshape!  [sample_h_size,batch_human, dim_a]

        h_human_expanded = tf.expand_dims(h_human, 0)  # Expand dims to [1, batch_human, dim_a]
        h_human_tiled = tf.tile(h_human_expanded, [sample_h_size, 1, 1])  # Now tile it

        norm_h_human_tiled = tf.norm(h_human_tiled, axis=-1, keepdims=True)
        sampled_vectical_vectors = tf.nn.l2_normalize(sampled_vectical_vectors, axis=-1) * norm_h_human_tiled

        sampled_vectors = 0.5 * sampled_vectical_vectors * tf.math.sin(alpha) + 0.5 * h_human_tiled * tf.math.cos(alpha)

        return sampled_vectors


    def _policy_batch_update_contrastive(self, batch):
        state_batch = [np.array(pair[0]) for pair in batch]
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]

        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)

        self._policy_single_update_contrastive(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor)


    def _policy_batch_update_contrastive_sphere(self, batch):
        state_batch = [np.array(pair[0]) for pair in batch]
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]

        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        self._policy_single_update_contrastive_sphere(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor)
        
    def _generate_policy_label(self, action, h):
        error =  np.matmul(self.e, h)
        error = np.array(error).reshape(1, self.dim_a)
        policy_action_label = []

        for i in range(self.dim_a):
            policy_action_label.append(np.clip(action[i] + error[0, i], -1, 1))

        policy_action_label = np.array(policy_action_label).reshape(1, self.dim_a)

        return policy_action_label
    
    def action(self, state_representation):
        action = self.policy_model(state_representation)
        action = action.numpy()
        out_action = np.clip(action, -1, 1)
        return out_action.reshape(-1)  # (dim_a, )
       

    def Train_policy_contrastive_sphere_combined_relative_absolute(self, action, h, observation, t, done, correction_e = 0.1):
        if np.any(h):
            # 1. append  (o_t, a_t, h_t) to D
            self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]), dtype=tf.float32)
            action_tf = tf.convert_to_tensor(np.reshape(action, [1, self.dim_a]), dtype=tf.float32)
            self.buffer.add([observation, action_tf, self.h_to_buffer, correction_e])

            if not self.buffer.initialized():
                action_label = self._generate_policy_label(action, h)
                self._single_update(observation, action_label)

            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                # include the new data in this batch
                batch[-1] = [observation, action_tf, self.h_to_buffer, correction_e]
                self._policy_batch_update_contrastive_sphere(batch)
            
        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._policy_batch_update_contrastive_sphere(batch)

        if self.buffer.initialized() and (self.train_end_episode and done):
            print("train contrastive policy sphere")
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print("alpha: ", self.sphere_alpha, " gamma", self.sphere_gamma)
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length())

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self._policy_batch_update_contrastive_sphere(batch)

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.Train_policy_contrastive_sphere(last_action, h, obs_proc, t, done)

    def Train_policy_contrastive_sphere(self, action, h, observation, t, done):
        if np.any(h):
            # 1. append  (o_t, a_t, h_t) to D
            self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]), dtype=tf.float32)
            action_tf = tf.convert_to_tensor(np.reshape(action, [1, self.dim_a]), dtype=tf.float32)
            self.buffer.add([observation, action_tf, self.h_to_buffer])

            if not self.buffer.initialized():
                action_label = self._generate_policy_label(action, h)
                self._single_update(observation, action_label)

            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                # include the new data in this batch
                batch[-1] = [observation, action_tf, self.h_to_buffer]
                self._policy_batch_update_contrastive_sphere(batch)
            
        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._policy_batch_update_contrastive_sphere(batch)

        if self.buffer.initialized() and (self.train_end_episode and done):
            print("train contrastive policy sphere")
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print("alpha: ", self.sphere_alpha, " gamma", self.sphere_gamma)
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length())

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self._policy_batch_update_contrastive_sphere(batch)
    
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


    


