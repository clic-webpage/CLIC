import numpy as np
import tensorflow as tf
import os
from tools.buffer import Buffer_uniform_sampling
from agents import mcmc

import ipdb
"""
Tensorflow implementation of CLIC
"""

import tensorflow as tf
from tensorflow.keras.models import load_model


# reference: ibc 
import tensorflow_probability as tfp
@tfp.experimental.auto_composite_tensor
class MappedCategorical(tfp.distributions.Categorical):
  """Categorical distribution that maps classes to specific values."""
  def __init__(self,
               logits=None,
               probs=None,
               mapped_values=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               name='MappedCategorical'):
    """Initialize Categorical distributions using class log-probabilities.
    """
    self._mapped_values = mapped_values
    super(MappedCategorical, self).__init__(
        logits=logits,
        probs=probs,
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)

  def mode(self, name='mode'):
    """Mode of the distribution."""
    mode = super(MappedCategorical, self).mode(name)
    return tf.gather(self._mapped_values, [mode], batch_dims=0)

  def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
    """Generate samples of the specified shape."""
    # TODO(oars): Fix for complex sample_shapes
    sample = super(MappedCategorical, self).sample(
        sample_shape=sample_shape, seed=seed, name=name, **kwargs)
    return tf.gather(self._mapped_values, [sample], batch_dims=0)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return tfp.distributions.Categorical._parameter_properties(
        dtype=dtype, num_classes=num_classes)


@tf.function
def sample_action_from_Q_function(Q_value_network, state, batch_size, sample_actions_size, evaluation, dim_a):
    action_samples_original = tf.random.uniform([batch_size * sample_actions_size, dim_a], minval=-1, maxval=1)
    state_tiled = tf.tile(state, (sample_actions_size, 1))

    langevin_return = mcmc.langevin_actions_given_obs(
            Q_value_network,
            state_tiled,
            action_samples_original,
            policy_state=(),
            min_actions=-1,
            max_actions=1,
            training=False,
            tfa_step_type=(),
            return_chain=False,  
            grad_norm_type='inf',
            num_action_samples=sample_actions_size)
    # action_samples, chain_data = langevin_return
    action_samples = langevin_return

    langevin_return = mcmc.langevin_actions_given_obs(
            Q_value_network,
            state_tiled,
            action_samples,
            policy_state=(),
            min_actions=-1,
            max_actions=1,
            training=False,
            tfa_step_type=(),
            return_chain=False,  
            grad_norm_type='inf',
            sampler_stepsize_init=1e-1,
            sampler_stepsize_final=1e-5,
            num_action_samples=sample_actions_size)
    action_samples = langevin_return
    
    probs = mcmc.get_probabilities(Q_value_network,
                                     batch_size,
                                     sample_actions_size,
                                     state_tiled,
                                     action_samples,
                                     training=False)

    distribution = MappedCategorical(probs=probs, mapped_values=action_samples)
    action_sample = distribution.sample()
    return action_sample


@tf.function
def make_counter_example_actions(
        Q_value_network, 
      observations,  # B x obs_spec
    #   expanded_actions,  # B x 1 x act_spec
      batch_size,
      dim_a,
      sample_closer_to_optimal_a = False,
      training=False,
      sample_actions_size = 1024,
    #   sample_actions_size = 10
    ):
    """Given observations and true actions, create counter example actions."""
    random_uniform_example_actions = tf.random.uniform([batch_size * sample_actions_size, dim_a], minval=-1, maxval=1)


    # Reshape to put B and num counter examples on same tensor dimenison
    # [B*num_counter_examples x act_spec]
    random_uniform_example_actions = tf.reshape(
        random_uniform_example_actions,
        (batch_size * sample_actions_size, -1))


    # maybe_tiled_obs_n = tf.tile(observations, [sample_actions_size, 1])
    maybe_tiled_obs_n = observations


    lang_opt_counter_example_actions = None
    
    chain_data = None
    return_chain = False
    langevin_return = mcmc.langevin_actions_given_obs(
        Q_value_network,
        maybe_tiled_obs_n,
        random_uniform_example_actions,
        policy_state=(),
        min_actions=-1,
        max_actions=1,
        training=False,
        num_iterations = 25,
        tfa_step_type=(),
        # return_chain=True, 
        return_chain=return_chain,  
        grad_norm_type='inf',
        num_action_samples=sample_actions_size)
    if sample_closer_to_optimal_a:
        action_samples = langevin_return
        langevin_return = mcmc.langevin_actions_given_obs(
                Q_value_network,
                maybe_tiled_obs_n,
                action_samples,
                policy_state=(),
                min_actions=-1,
                max_actions=1,
                training=False,
                tfa_step_type=(),
                return_chain=False,  
                grad_norm_type='inf',
                sampler_stepsize_init=1e-1,
                sampler_stepsize_final=1e-5,
                num_action_samples=sample_actions_size)
    
    if return_chain is False:
        chain_data = None
        lang_opt_counter_example_actions = langevin_return
    else:
        lang_opt_counter_example_actions, chain_data = langevin_return

    counter_example_actions = tf.reshape(
        lang_opt_counter_example_actions, (sample_actions_size, batch_size, -1))

    return counter_example_actions, chain_data


@tf.function
def grad_penalty(energy_network,
                 batch_size,
                #  chain_data,
                 observations,
                 combined_true_counter_actions,
                 grad_norm_type='inf',
                 training=False,
                 only_apply_final_grad_penalty=True,
                 grad_margin=1.0,
                 square_grad_penalty=True,
                 grad_loss_weight=1.0):
  """Calculate losses based on some norm of dE/dactions from mcmc samples."""

  if only_apply_final_grad_penalty:
    de_dact, _ = mcmc.gradient_wrt_act(
        energy_network,
        observations,
        tf.stop_gradient(combined_true_counter_actions),
        training,
        network_state=(),
        tfa_step_type=(),
        apply_exp=False)  # TODO(peteflorence): config this.

    # grad norms should now be shape (b*(n+1))
    grad_norms = mcmc.compute_grad_norm(grad_norm_type, de_dact)
    grad_norms = tf.reshape(grad_norms, (batch_size, -1))


  if grad_margin is not None:
    grad_norms -= grad_margin
    # assume 1e10 is big enough
    grad_norms = tf.clip_by_value(grad_norms, 0., 1e10)

  if square_grad_penalty:
    grad_norms = grad_norms**2

  grad_loss = tf.reduce_mean(grad_norms, axis=1)
  return grad_loss * grad_loss_weight


class TimeBasedDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate):
        super(TimeBasedDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Explicitly cast step to float
        return self.initial_learning_rate / (1 + self.decay_rate * step)


class CLIC:
    def __init__(self, dim_a, dim_o, shape_meta, action_upper_limits, action_lower_limits, e_matrix,
                 loss_weight_inverse_e, sphere_alpha, sphere_gamma, radius_ratio,
                 buffer_min_size, buffer_max_size, buffer_sampling_rate,
                 buffer_sampling_size, train_end_episode,
                 policy_model_learning_rate,
                 saved_dir, load_dir, load_policy,
                 number_training_iterations,
                #  desiredA_type = 'Circular',
                desiredA_type = 'Half',
                 n_obs_steps=1, softmax_temperature=1, 
                 sample_action_number=512):
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
        self.softmax_temperature = softmax_temperature # temperature used in softmax of current policy
        self.sample_action_number = sample_action_number # number of sampled actions during training. Used to estimate the probability of the target policy and the current policy

        
        self.e = np.diag(e_matrix)
        self.loss_weight_inverse_e = np.diag(loss_weight_inverse_e)

        self.sphere_alpha = sphere_alpha    # parameter that adjusts where to sample counterexamples on the sphere
        self.sphere_gamma = sphere_gamma    # parameter that adjust the center of the sphere to samle counterexamples
        self.radius_ratio = radius_ratio    # parameter used in circular desired action space
        
        self.desiredA_type = desiredA_type
        assert self.desiredA_type in ['Half', 'Circular']

        # Data buffer D that saves observed action pairs during online interactions: (state, learner's action, teacher's action)
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
        # Energy-based model: reuse the Q-network and interpret E(s, a) = -Q(s, a) (sign flip only).
        # For more background info, please refer to our CLIC paper (Sec. III.A3)
        self.action_value_model = neural_network.action_value_function_model()
        self.optimizer_action_value_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)


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
        # if dim_a - 1 == 3:
        #     coefficients = self.sample_equally_on_sphere(sample_h_size)
        if dim_a - 1 == 1:
            coefficients = self.sample_equally_on_1d(sample_h_size)
        # elif dim_a - 1 == 6:
        #     loaded_points = np.load('agents/points_on_6dsphere.npy')
        #     coefficients = tf.convert_to_tensor(loaded_points, dtype=tf.float32)
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
    
    def custom_l4_penalty(self,Q_values, lower_bound=-1.0, upper_bound=1.0):
            # Find values outside the [-1, 1] range
            penalty = tf.square(tf.maximum(Q_values - upper_bound, 0.0)) + tf.square(tf.minimum(Q_values - lower_bound, 0.0))
            return tf.reduce_sum(tf.square(penalty))
    
    def process_tensor_sample_h_size_batch(self, tensor, sample_h_size, sample_implicit_size, batch_size):
        tensor = tf.reshape(tensor, [sample_h_size, batch_size, -1])
        tensor = tf.tile(tensor, [1, sample_implicit_size, 1])
        tensor = tf.reshape(tensor, [sample_h_size * sample_implicit_size * batch_size, -1])
        return tensor


    def action_value_single_update_feasible_space_with_bayesian_implicit_areasector(self, observation, action, h_human, next_observation):
        batch_size = h_human.shape[0]
        action_dim = h_human.shape[-1]
        sample_h_size = self.sample_action_number # Number of sampled actions
        sample_implicit_size = 128 

        alpha = self.sphere_alpha * np.pi / 180
        beta = self.sphere_alpha * 0.5 * np.pi / 180  # Angle threshold in radians
        sphere_center_gamma = self.sphere_gamma

        softmax_temperature = self.softmax_temperature 

        # ---------------- Draw actions from current energy-based model ---------------- #
        observation_tiled = tf.tile(observation, [sample_h_size, 1])
        sampled_action, _ = make_counter_example_actions(self.action_value_model, observation_tiled, batch_size, action_dim, sample_actions_size = sample_h_size)
        sampled_action = tf.concat([tf.expand_dims(tf.identity(action), 0), 
                            tf.expand_dims(tf.identity(action) + tf.matmul(tf.identity(h_human), self.e), 0),
                            sampled_action[2:, :, :]], axis=0)   # (sample_a_size, Batch_size, dim_a)
        sampled_action = tf.reshape(sampled_action, [sample_h_size*batch_size, -1]) # (sampled_a_size*B, dim_a)

        self.sampled_action_last = sampled_action # saved for visualization in 2D toy example (batch = 1)
        
        Q_s_a_target = self.action_value_model([observation_tiled, sampled_action]) 
        Q_s_a_target = tf.stop_gradient(Q_s_a_target)
        Q_s_a_target = tf.transpose(tf.reshape(Q_s_a_target, [sample_h_size, -1]))   # (B, sampled_a_size)
        
        test_bounded2N = False
        # ---------------- Data Augumentation: Draw Contrastive Action Pairs for each observed action pair ---------------- #
        # v_combined can contain NaN values
        if self.dim_a > 1:
            rerun_count = 0
            rerun_limit = 10
            while rerun_count <= rerun_limit:
                if test_bounded2N is False:
                    v_combined = self.sample_vectors_from_sphere(h_human, sample_implicit_size, alpha)
                if tf.reduce_any(tf.math.is_nan(v_combined)):
                    rerun_count += 1
                    print(f"NaN detected, rerunning the function... (Attempt {rerun_count})")
                else:
                    break
            if tf.reduce_any(tf.math.is_nan(v_combined)):
                print("NaN fails")
                ipdb.set_trace()
                return None

            normalized_random_hs = tf.reshape(v_combined, [-1, h_human.shape[-1]]) * 2.0   # (B*sample_implicit_size, dim_a)

        with tf.GradientTape() as tape_policy:
            # ---------------- Observation model of Desired Action Spaces ---------------- #
            action_tiled = tf.tile(action, [sample_h_size, 1])
            h_human_tiled = tf.tile(h_human, [sample_h_size, 1])
            sampled_action_reshaped = sampled_action # (sampled_a_size*B, dim_a)
            sampled_action_diff = sampled_action_reshaped - action_tiled  # (sampled_a_size * B, dim_a)

            coefficient_exp = 0.1 
            area_sector = -tf.norm(sampled_action_diff, axis=-1, keepdims=True) + tf.norm(sampled_action_diff- self.e[0, 0] * sphere_center_gamma *  h_human_tiled, axis=-1, keepdims=True) 
            area_sector = tf.math.log(1.0 / (1.0 + tf.math.exp(area_sector / coefficient_exp )))
            area_sector = tf.transpose(tf.reshape(area_sector, [sample_h_size, -1]))

            if self.dim_a > 1:
                # ---------------- Observation model of Desired Half-space (created from Contrastive Action pairs)---------------- #
                action_tiled_for_implicit = tf.tile(action, [sample_implicit_size , 1])
                action_tiled_for_implicit_negative = tf.tile(action + self.e[0, 0] * sphere_center_gamma * h_human, [sample_implicit_size , 1]) # (sample_implicit_size * B , dim_a)                    
                implicit_actions_negative  = action_tiled_for_implicit_negative +  self.e[0, 0] * (1 - sphere_center_gamma) * normalized_random_hs  # (sample_implicit_size * B, dim_a)
                action_tiled_for_implicit_reshaped = tf.tile(action_tiled_for_implicit, [sample_h_size, 1])
                implicit_actions_negative_reshaped = tf.tile(implicit_actions_negative, [sample_h_size, 1])  # (sampled_h_size * sample_implicit_size * B, dim_a)
                sampled_action_reshaped_tiled = self.process_tensor_sample_h_size_batch(sampled_action_reshaped, sample_h_size, sample_implicit_size, batch_size)
            
                area_sector_implicit = -tf.norm(sampled_action_reshaped_tiled- implicit_actions_negative_reshaped, axis=-1, keepdims=True) + tf.norm(sampled_action_reshaped_tiled- action_tiled_for_implicit_reshaped - self.e[0,0] * self.process_tensor_sample_h_size_batch(h_human_tiled, sample_h_size, sample_implicit_size, batch_size), axis=-1, keepdims=True) 

                area_sector_implicit = tf.reshape(area_sector_implicit, [sample_h_size, sample_implicit_size, batch_size ])
                area_sector_implicit = tf.math.log(1.0 / (1.0 + tf.math.exp(area_sector_implicit / coefficient_exp)))
                sum_area_sector_implicit = tf.reduce_sum(area_sector_implicit, axis=1) 
                sum_area_sector_implicit = tf.transpose(sum_area_sector_implicit) 
                weight_angle_sector = 0.3 * sample_implicit_size

                mean_area_sector = (sum_area_sector_implicit + weight_angle_sector * area_sector) / (weight_angle_sector + sample_implicit_size)
            else:
                mean_area_sector = area_sector
            
            if test_bounded2N:
                combined_condition_value = sum_area_sector_implicit / sample_implicit_size
            else:
                combined_condition_value = mean_area_sector 

            # ---------------- Traget Policy ---------------- #
            labels_dynamic = Q_s_a_target/softmax_temperature + combined_condition_value/ (0.1)
            # labels_dynamic = Q_s_a_target/softmax_temperature + combined_condition_value/ (1.0) # bad
            ''' test with direct label, without bayesian '''
            # labels_dynamic =  combined_condition_value/ (0.1)
            labels_dynamic = tf.nn.softmax(labels_dynamic, axis=-1)  # [batch_size, sampled_h_size]

            # ---------------- Current Policy ---------------- #
            Q_s_a = self.action_value_model([observation_tiled, sampled_action])  # size: [sample_h_size * batch, 1]
            predictions = Q_s_a
            predictions = tf.reshape(predictions, [sample_h_size, -1]) # [sample_h_size, batch]
            softmaxed_predictions = tf.nn.softmax(predictions / softmax_temperature, axis=0)
            softmaxed_predictions = tf.transpose(softmaxed_predictions)  # [batch_size, sampled_h_size]
            
            # ---------------- Caculate KL loss and gradient pently ---------------- #
            kl_divergence = tf.keras.losses.KLDivergence()
            per_example_loss = kl_divergence(labels_dynamic, softmaxed_predictions)

            grad_penalty_loss = grad_penalty(self.action_value_model, batch_size, observation_tiled, sampled_action)
            grad_penalty_loss = tf.reduce_mean(grad_penalty_loss)
            print(self.desiredA_type, " grad_penalty_loss: ", grad_penalty_loss, " per_example_loss: ", per_example_loss)
            per_example_loss = grad_penalty_loss + per_example_loss #+ q_value_penalty# + regularization_loss_Qold          

            # Backpropagate the loss
            grads = tape_policy.gradient(per_example_loss, self.action_value_model.trainable_variables)
            
        # Optionally, clip the gradients
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        
        self.optimizer_action_value_model.apply_gradients(zip(grads, self.action_value_model.trainable_variables))


    def action_value_single_update_Circular(self, observation, h_human, action_negative = None):
        sample_h_size = self.sample_action_number  # Number of sampled actions
        observation_tiled = tf.tile(observation, [sample_h_size, 1])
       
        # ---------------- Draw actions from current energy-based model ---------------- #
        sampled_action, _ = make_counter_example_actions(self.action_value_model, observation_tiled, h_human.shape[0], h_human.shape[-1], sample_actions_size=sample_h_size)
        action_positive = action_negative + h_human
        if action_negative is not None:
            sampled_action = tf.concat([tf.expand_dims(action_positive, 0),   #  adding a+ is bad
                                        #  tf.expand_dims(action_negative, 0),  # adding this is bad
                            sampled_action[1:, :, :]], axis=0)  
        else:
            sampled_action = tf.concat([tf.expand_dims(action_positive, 0), 
                                sampled_action[1:, :, :]], axis=0)
        sampled_action = tf.reshape(sampled_action, [sample_h_size*h_human.shape[0], -1])
        self.sampled_action_last = sampled_action

        Q_s_a_target = self.action_value_model([observation_tiled, sampled_action])
        Q_s_a_target = tf.stop_gradient(Q_s_a_target)

        with tf.GradientTape() as tape_policy:
            action_positive_tiled = tf.tile(action_positive, [sample_h_size, 1])
            Q_s_a = self.action_value_model([observation_tiled, sampled_action])  # size: [sample_h_size * batch, 1]
            predictions = Q_s_a

            # ---------------- Observation model of Desired Action Spaces ---------------- #
            coefficient_exp = 0.05  # better than 0.025 and 0.1
            distance_a_a_plus =tf.norm((sampled_action - action_positive_tiled) , axis=1, keepdims=True)  

            if action_negative is not None:
                action_negative_tiled = tf.tile(action_negative, [sample_h_size, 1])

                sigma = self.radius_ratio * tf.norm((action_negative_tiled - action_positive_tiled), axis = 1, keepdims=True)

            area_sector =  distance_a_a_plus -  sigma
            # area_sector = (distance_a_a_plus -  sigma) / sigma
            combined_condition_value = tf.math.log(1.0 / (1.0 + tf.math.exp(area_sector / coefficient_exp)))
            
            # ---------------- Traget Policy ---------------- #
            softmax_temperature = self.softmax_temperature 
            labels_dynamic = Q_s_a_target/softmax_temperature + combined_condition_value/0.1  # CT01
            labels_dynamic = tf.reshape(labels_dynamic, [sample_h_size, -1])   # [sample_h_size, batch_size]
            labels_dynamic = tf.nn.softmax(labels_dynamic, axis=0)
            labels_dynamic = tf.transpose(labels_dynamic)

            # ---------------- Current Policy ---------------- #
            predictions = tf.reshape(predictions, [sample_h_size, -1])
            softmaxed_predictions = tf.nn.softmax(predictions / softmax_temperature, axis=0)
            softmaxed_predictions = tf.transpose(softmaxed_predictions)

            # ---------------- Caculate KL loss and gradient pently ---------------- #
            kl_divergence = tf.keras.losses.KLDivergence()
            per_example_loss =  kl_divergence(labels_dynamic, softmaxed_predictions)  #

            grad_penalty_loss = grad_penalty(self.action_value_model, h_human.shape[0], observation_tiled, sampled_action)
            grad_penalty_loss = tf.reduce_mean(grad_penalty_loss)   
            print(self.desiredA_type, " grad_penalty_loss: ", grad_penalty_loss, " per_example_loss: ", per_example_loss)
            per_example_loss = grad_penalty_loss + per_example_loss
            # Backpropagate the loss
            grads = tape_policy.gradient(per_example_loss, self.action_value_model.trainable_variables)
            
        # Optionally, clip the gradients
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]   
        self.optimizer_action_value_model.apply_gradients(zip(grads, self.action_value_model.trainable_variables))

    # Policy shaping via Desired action spaces
    def action_value_batch_update(self, batch):
        state_batch = [pair[0] for pair in batch]
        action_batch = [pair[1] for pair in batch]
        h_human_batch = [pair[2] for pair in batch]
        next_state_batch = [pair[3] for pair in batch]
        # Check for None in action_batch
        action_contains_none = any(tf.reduce_any(tf.math.is_nan(item)) for item in action_batch)
        h_human_contains_none = any(tf.reduce_any(tf.math.is_nan(item)) for item in h_human_batch)
        none_flag = action_contains_none or h_human_contains_none

        if none_flag is False:
            # Reshape and transform to tensor so they can be pass to the model:
            observation_reshaped_tensor = tf.concat(
                [tf.reshape(tf.cast(state, tf.float32), [1, self.dim_o]) for state in state_batch], axis=0
            )
            action_reshaped_tensor = tf.concat(
                [tf.reshape(tf.cast(action, tf.float32), [1, self.dim_a]) for action in action_batch], axis=0
            )
            h_human_reshaped_tensor = tf.concat(
                [tf.reshape(tf.cast(h_human, tf.float32), [1, self.dim_a]) for h_human in h_human_batch], axis=0
            )
            next_observation_reshaped_tensor = tf.concat(
                [tf.reshape(tf.cast(next_state, tf.float32), [1, self.dim_o]) for next_state in next_state_batch], axis=0
            )         

            if self.desiredA_type == 'Half':
                self.action_value_single_update_feasible_space_with_bayesian_implicit_areasector(
                    observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor, next_observation_reshaped_tensor
                    )
            elif self.desiredA_type == 'Circular':
                self.action_value_single_update_Circular(
                    observation_reshaped_tensor, action_negative= action_reshaped_tensor, h_human = h_human_reshaped_tensor
                )   
            else:
                raise ValueError(f"Unsupported desiredA type {self.desiredA_type}")

    def action(self, state_representation):
        action = sample_action_from_Q_function(self.action_value_model, state_representation, 1, 512, True, self.dim_a)
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
        '''
        Input: 
            action: robot action from the robot policy
            h: corrective feedback, can be none
            observation: observation at current state
            next_observation: observation at the next state.
        Notes:
            For CLIC with demonstration, h is defined as corrective signal, which is h = action_human - action_robot, or h = action_positive - action_negative
            For CLIC with relative correction, h is defined as a directional siginal with magnitude equal to 1. Here, action_human = action_robot + agent.e * h
        '''
        if np.any(h):  # if any element is not 0
            # 1. append  (o_t, a_t, h_t) to D
            # self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]), dtype=tf.float32)
            # action_tf = tf.convert_to_tensor(np.reshape(action, [1, self.dim_a]), dtype=tf.float32)

            # self.buffer.add([observation, action, h, next_observation])
            self.buffer.add([observation, tf.convert_to_tensor(action, dtype=tf.float32), tf.convert_to_tensor(h, dtype=tf.float32), next_observation])

            if self.buffer.initialized():

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                # include the new data in this batch
                batch[-1] = [observation, action, h, next_observation]
                self.action_value_batch_update(batch)

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            for i in range(1):  
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.action_value_batch_update(batch)

        if done:
            self.last_action = None

        if self.buffer.initialized() and (self.train_end_episode and done):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length())
                   
                for i in range(1):
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.action_value_batch_update(batch)
   

    def save_model(self):
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        self.action_value_model.save(network_saved_dir + 'Q_value_model.h5')

    def load_Q_value_model(self, neural_network):
        network_load_dir = self.load_dir + 'network_params/'
        if not os.path.exists(network_load_dir):
            print("The load path does not exist: ", network_load_dir)
        else:
            model_filename = network_load_dir + 'Q_value_model.h5'
            if os.path.exists(model_filename):
                self.action_value_model = load_model(network_load_dir + 'Q_value_model.h5')
                print("load policy model successfully! ")
            else:
                self.action_value_model = neural_network.action_value_function_model()
                print(f"Model file '{model_filename}' not found. Skipping model loading.")


    
    

    


