import numpy as np
from tools.buffer import Buffer
from typing import Iterable
import os
from tools.buffer import Buffer_uniform_sampling
import torch
import torch.nn.functional as F
from agents.DP_model.energy_nn import ActionValueFunctionModel
from agents.diffusion_policy import BaseLowdimPolicy

from agents.CLIC_torch import sample_action_from_Q_function, make_counter_example_actions, grad_penalty
"""
Implementation of Implicit BC in torch, adapted from CLIC/src/agents/ibc.py
"""

class ibc(BaseLowdimPolicy):
    def __init__(self,  dim_a, dim_o, 
                 action_upper_limits, action_lower_limits,
                 buffer_min_size, buffer_max_size, buffer_sampling_rate,
                 buffer_sampling_size, train_end_episode,
                 policy_model_learning_rate,
                 saved_dir, load_dir, load_policy,
                 number_training_iterations,
                 sample_action_number,
                 n_action_steps = 1, 
                 n_obs_steps=1):
        
        super().__init__()
        
        # Initialize variables
        self.n_obs_steps = n_obs_steps
        self.h = None
        self.state_representation = None
        self.policy_action_label = None
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
        
        self.device = torch.device("cuda:0")  
        self.action_value_model = ActionValueFunctionModel(dim_a=dim_a, dim_o=self.dim_o* n_obs_steps).to(self.device)
        
        self.optimizer_action_value_model = torch.optim.AdamW(
            self.parameters(),
            lr=self.policy_model_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            weight_decay=1.0e-6
        )
        
        self.buffer = Buffer_uniform_sampling(min_size=self.buffer_min_size,
                                             max_size=self.buffer_max_size)
        self.sample_action_number = sample_action_number

        self.policy_loss_list = []
        self.test_New_value_function_idea = False
        self.saved_dir = saved_dir
        self.load_dir = load_dir
        self.load_policy_flag = load_policy
        self.number_training_iterations = number_training_iterations
        self.time_step = None
        self.evaluation = False  # for the number of action spaces during inference
        self.evaluation_last = False
        self.list_action_tobeQueried = []
        self.last_action = True
        self.sampled_action_last = None
        self.use_CLIC_algorithm = False
        self.e = 0.2 # used for relative correction data

    def save_model(self):
        # Define the directory for saving model parameters
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        
        # Save the model state dictionary
        model_filename = network_saved_dir + 'action_value_model.pth'
        torch.save({'action_value_model_state_dict': self.action_value_model.state_dict()}, model_filename)
        print(f"action_value_model saved at {model_filename}")

    def load_model(self):
        model_dir = os.path.join(self.load_dir, 'network_params')

        # Load policy model
        model_path = os.path.join(model_dir, 'action_value_model.pth')
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.action_value_model.load_state_dict(checkpoint['action_value_model_state_dict'])
            print(f"action_value_model loaded from {model_path}")
        else:
            print(f"action_value_model file not found at {model_path}, skipping.")

        # Load observation encoder
        obs_enc_path = os.path.join(model_dir, 'obs_encoder.pth')
        if os.path.isfile(obs_enc_path):
            checkpoint = torch.load(obs_enc_path, map_location=self.device)
            self.obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
            print(f"Obs encoder loaded from {obs_enc_path}")
        else:
            print(f"Obs encoder file not found at {obs_enc_path}, skipping.")

    def custom_l4_penalty(self, Q_values, lower_bound=-1.0, upper_bound=1.0):
        penalty = torch.clamp(Q_values - upper_bound, min=0.0)**2
        penalty += torch.clamp(Q_values - lower_bound, max=0.0)**2
        return torch.sum(penalty**2)

    def action_value_single_update_InfoNCE(self, observation, action, h_human, next_observation):
        self.train()
        self.action_value_model.training = True

        batch_size = h_human.shape[0]
        action_dim = h_human.shape[-1]  # (Batch, horizon, dim_a)
        sample_h_size = self.sample_action_number

        softmax_temperature = 1.0

        global_cond = observation   # just the observation, used to condition the action

        observation_tiled = global_cond.repeat(sample_h_size, 1)

        sampled_action, _ = make_counter_example_actions(
            self.action_value_model, observation_tiled,
            batch_size, action_dim,
            sample_actions_size=sample_h_size)
        # prepend true action and shifted action
        first = h_human.unsqueeze(0)
        
        sampled_action = torch.cat([first, sampled_action[1:]], dim=0)

        sampled_action = sampled_action.view(-1, action_dim)
        self.sampled_action_last = sampled_action

        h_tiled = h_human.repeat(sample_h_size, 1)
        log_prob = -torch.sum((sampled_action - h_tiled) ** 2, dim=-1, keepdim=True)
            
        threshold = 0.0001
        cond = (-log_prob < threshold).float()  
        cond = cond.view(sample_h_size, batch_size).transpose(0, 1) 
        obs_tiled = observation_tiled
        Q_s_a = self.action_value_model(obs_tiled, sampled_action)

        labels = cond
        labels = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)
            
        # InfoNCE
        preds = Q_s_a.reshape(sample_h_size, batch_size)
        soft_p = F.softmax(preds/softmax_temperature, dim=0).transpose(0,1)
        
        
        loss_kl = F.kl_div(soft_p.log(), labels, reduction='batchmean')

        grad_pen = grad_penalty(self.action_value_model, batch_size, observation_tiled, sampled_action, training=True)
        grad_pen = grad_pen.mean()
        loss = loss_kl + grad_pen
        print("IBC loss_Kl: ", loss_kl, " grad_pen: ", grad_pen)
        self.optimizer_action_value_model.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.action_value_model.parameters(), 1.0)
        self.optimizer_action_value_model.step()


    def action_value_batch_update(self, batch):
        state_batch = [pair[0] for pair in batch]
        action_batch = [np.array(pair[2]) for pair in batch]  # robot action
        h_human_batch = [np.array(pair[1]) for pair in batch]  # human action
        batch_size = len(batch)

        h_human_batch     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.dim_a]), dtype=torch.float32, device=self.device)
        action_batch     = torch.tensor(np.reshape(action_batch, [batch_size, self.dim_a]), dtype=torch.float32, device=self.device)
        obs = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32, device=self.device)
        next_obs = None

        self.action_value_single_update_InfoNCE(
            obs, action_batch, h_human_batch, next_obs
        )

    def action(self, observation):
        observation = torch.tensor(observation, dtype=self.dtype, device=self.device)
        # sample from the EBM to obtain action that minimize this EBM (through MCMC sampling).
        # Note: avoid using 'iterative_dfo' as this function only works in low-dim action space.
        action = sample_action_from_Q_function(self.action_value_model, observation, 1, 512, True, self.dim_a, self.device, 
                                                ).detach()

        action = action.cpu().numpy()
        numpy_action = np.clip(action, -1, 1)
        return numpy_action
    
    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Policy(last_action, t, done, i_episode, h, obs_proc)

    def TRAIN_Policy(self, action, t, done, i_episode, h, observation):
        next_observation = None # not implemented
        if np.any(h):  # if any element is not 0
            # 1. append  (o_t,  h_t, a_T) to D
           # h_t is the optimal action here and a_t is the robot action
            # print("action in TRAIN_Policy: ", action, " h: ", h)
            self.buffer.add([observation, h, action])
            self.latested_data_pair = [observation, h, action]
            
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                # include the new data in this batch
                batch[-1] = self.latested_data_pair
                self.action_value_batch_update(batch)

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self.action_value_batch_update(batch)

        if done:
            self.last_action = None

        if self.buffer.initialized() and (self.train_end_episode and done):

            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))
                    print("buffer size: ", self.buffer.length())
                   
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.action_value_batch_update(batch)
