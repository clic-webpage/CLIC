import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D
from agents.DP_model.diffusion.mask_generator import LowdimMaskGenerator

import os
import numpy as np
from tools.buffer import Buffer, Buffer_uniform_sampling

from agents.DP_model.common.scheduler import CosineAnnealingWarmupRestarts

"""
Implementation of Diffusion policy,  mainly copied from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py 
"""

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    # @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class BaseLowdimPolicy(ModuleAttrMixin):  

    def action(self, state_representation):
        raise NotImplementedError()



class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            # model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            saved_dir, load_dir, load_pretrained_dir,
            load_policy, number_training_iterations,
            buffer_min_size, buffer_max_size, 
            buffer_sampling_size, policy_model_learning_rate,
            n_action_steps = 1, 
            n_obs_steps = 1 ,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=True,
            pred_action_steps_only=False,
            oa_step_convention=False, 
            use_ambient_loss = False,
            ambient_k = 3,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        
        self.device = torch.device("cuda:0")  
       
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        # self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        self.policy_model_learning_rate = policy_model_learning_rate
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        self.buffer_sampling_size = buffer_sampling_size
        
        self.buffer = Buffer_uniform_sampling(min_size=self.buffer_min_size, max_size=self.buffer_max_size )
        self.use_CLIC_algorithm = False # not used in HG-Dagger
        self.dim_o = obs_dim
        self.dim_a = action_dim
        self.train_end_episode = True
        self.buffer_sampling_rate = 5
        self.traning_count = 0
        self.number_training_iterations = number_training_iterations

        self.saved_dir = saved_dir  # used to save for the buffer & network models
        self.load_dir = load_dir
        self.load_pretrained_dir = load_pretrained_dir
        self.load_policy_flag = load_policy

        
        # self.model = ConditionalUnet1D(
        #                     input_dim=action_dim,
        #                     local_cond_dim=None,
        #                     global_cond_dim=obs_dim * n_obs_steps,
        #                     cond_predict_scale=True
        #                 ).to(self.device)   
        from agents.DP_model.diffusion.conditional_unet1d import ConditionalUnet1D, HumanFunctionModel

        self.model = HumanFunctionModel(dim_a=self.dim_a , dim_o=self.dim_o).to(self.device)

        self.e = 0.2 # used for relative correction data

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.policy_model_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-7,
        )

        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=162 * (500 + self.number_training_iterations),
            cycle_mult=1.0,
            max_lr=self.policy_model_learning_rate,
            min_lr=1e-5,
            warmup_steps=10,
            gamma=1.0,
        )

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.evaluation=False

        self.use_ambient_loss = use_ambient_loss
        self.ambient_k = ambient_k

        self.save_save_count = 0
    

    def save_model(self):
        # Define the directory for saving model parameters
        network_saved_dir = self.saved_dir + 'network_params/'
        if not os.path.exists(network_saved_dir):
            os.makedirs(network_saved_dir)
        
        # Save the model state dictionary
        model_filename = network_saved_dir + 'diffusion_model' + '.pth'
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, model_filename)
        
        self.save_save_count = self.save_save_count + 1
        print(f"diffusion model saved at {model_filename}")

    def load_model(self, model_name = None):
        model_dir = os.path.join(self.load_dir, 'network_params')

        # Load policy model
        if model_name is None:
            model_name = 'diffusion_model.pth'
        model_path = os.path.join(model_dir, model_name)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Policy model loaded from {model_path}")
        else:
            print(f"Policy model file not found at {model_path}, skipping.")

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        # print("condition data device: ", condition_data.device)
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t.to(self.device), 
                local_cond=local_cond, global_cond=global_cond.to(self.device))

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def action(self, state_representation):
        if self.evaluation:
            self.model.eval()
            self.model.training = False
        with torch.no_grad():
            state_representation = torch.tensor(state_representation, dtype=self.dtype)
            state_representation = state_representation.unsqueeze(0)
            nobs = state_representation
            B, _, Do = nobs.shape
            To = self.n_obs_steps
            T = self.horizon
            Da = self.action_dim

            # build input
            device = self.device
            # print("device: ", device)
            dtype = self.dtype

            # handle different ways of passing observation
            local_cond = None
            global_cond = nobs.reshape(nobs.shape[0], -1)
            
            shape = (B, T, Da)
            # shape = (B, Da)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            # run sampling
            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
            
            # unnormalize prediction
            naction_pred = nsample[...,:Da]

            if T == 1:
                numpy_action = naction_pred.detach().cpu().numpy().reshape(-1)
            else:
                # get action
                start = To - 1
                end = start + self.n_action_steps
                # action = naction_pred[:,start:end]
                action = naction_pred[:,start:]
                numpy_action = action.detach().cpu().numpy().reshape(T-start, -1)

            # Clip the values within the range [-1, 1]
            numpy_action = np.clip(numpy_action, -1, 1)
        return numpy_action
    
    # USED to generate some action samples for visualization
    def action_evalaution(self, state_representation):
        """
        obs_dict: must include \"obs\" key
        result: must include \"action\" key
        """
        num_of_samples = 512  # Number of action samples for clustering

        with torch.no_grad():
            # state_representation_raw = torch.tensor(state_representation.numpy())
            state_representation = torch.tensor(state_representation)

            # Repeat state for batch sampling
            state_representation = state_representation.unsqueeze(1)
            state_representation = state_representation.repeat(num_of_samples, 1, 1)
            
            nobs = state_representation
            B, _, Do = nobs.shape
            To = self.n_obs_steps
            assert Do == self.obs_dim

            T = self.horizon
            Da = self.action_dim

            device = self.device
            dtype = self.dtype

            global_cond = nobs[:, :To].reshape(B, -1)

            shape = (B, T, Da)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            # Generate action samples
            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=None,
                global_cond=global_cond,
                **self.kwargs)

            naction_pred = nsample[..., :Da]
            action_pred = naction_pred

            # Flatten action trajectories for clustering
            action_numpy = action_pred.detach().cpu().numpy().reshape(num_of_samples, -1)

            
        return action_numpy


    # ========= training  ============
    # def set_normalizer(self, normalizer: LinearNormalizer):
    #     self.normalizer.load_state_dict(normalizer.state_dict())
    def compute_loss(self, batch):
        self.model.train()
        self.model.training = True
        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        h_human_batch = [np.array(pair[1]) for pair in batch]  # last

        batch_size = len(batch)
        # Convert the numpy array to a PyTorch tensor and reshape it
        obs = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32)
        action     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)

        local_cond = None
        global_cond = None
        # action = action.unsqueeze(1)
        trajectory = action.to(self.device)
        global_cond = obs   # just the observation, used to condition the action

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
      
        # Predict the noise residual
        #x: (B,T,input_dim)
        #timestep: (B,) or int, diffusion step
        #local_cond: (B,T,local_cond_dim)
        #global_cond: (B,global_cond_dim)
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond.to(self.device))

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype) 
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()


    def compute_loss_AmbientDiffusion(self, batch):
        self.model.train()
        self.model.training = True

        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        robot_action_batch = [np.array(pair[2]) for pair in batch]
        h_human_batch = [np.array(pair[1]) for pair in batch]  # last

        batch_size = len(batch)
        # Convert the numpy array to a PyTorch tensor and reshape it
        obs = torch.tensor(np.reshape(state_batch, [batch_size,  self.dim_o]), dtype=torch.float32)
        action     = torch.tensor(np.reshape(h_human_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)

        # Attention: in the main, we change the input of action to optimal teacher action. 
        optimal_action     = torch.tensor(np.reshape(robot_action_batch, [batch_size, self.horizon, self.dim_a]), dtype=torch.float32)

        diff = torch.abs(action - optimal_action)               # [batch, dim_a
        # Add a small random perturbation to `diff` if it's all zeros
        if diff.sum() == 0:
            diff = diff + torch.randn_like(diff) * 1e-6  # Adding a tiny random noise
        ## caculate corruption matrix A (batch, dim_a, dim_a), diagonal matrix
        # For partial feedback, (I- A) * action = (I- A) * robot_action

        k = self.ambient_k # number of dimensions you want to “keep”
        _, idx = torch.topk(-diff, k, dim=-1)               # idx: [batch, k]
        A_diag = torch.zeros_like(diff)                    # [batch, dim]
        A_diag.scatter_(dim=-1, index=idx, src=torch.ones_like(idx, dtype=torch.float))                       # set the k smallest‐diff dims to 1
        mask_corruption = A_diag.to(self.device)

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        # action = action.unsqueeze(1)
        trajectory = action.to(self.device)

        global_cond = obs   # just the observation, used to condition the action
      
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond.to(self.device))

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')  # (batch, T, dim_a)
        # import pdb; pdb.set_trace()
        loss = loss * mask_corruption.type(loss.dtype) 
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def collect_data_and_train(self, last_action, h, obs_proc, next_obs, t, done, agent_algorithm=None, agent_type=None, i_episode=None):
        """Unified entry point used by main_IIL.py."""
        return self.TRAIN_Policy(last_action, t, done, i_episode, h, obs_proc)

    def TRAIN_Policy(self, action, t, done, i_episode, h, observation):
        if np.any(h):  
            # save the data pair to the buffer

            # h is defined as the teacher action
            self.buffer.add([observation, h, action])
            self.latested_data_pair = [observation, h, action]

            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                batch[-1] = self.latested_data_pair

                if self.use_ambient_loss:
                    self.compute_loss_AmbientDiffusion(batch)
                else:
                    self.compute_loss(batch)

        # Train policy every k time steps from buffer
        elif self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.buffer.initialized() and self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            if self.use_ambient_loss:
                self.compute_loss_AmbientDiffusion(batch)
            else:
                self.compute_loss(batch)

        if len(self.buffer.buffer) < self.buffer_sampling_size:
            self.traning_count = 0
        
        number_training_iterations_ = self.number_training_iterations

        if len(self.buffer.buffer) > self.buffer_sampling_size and ( (self.train_end_episode and done)):
            for i in range(number_training_iterations_):
                if i % (number_training_iterations_ / 20) == 0:
                    print("number_training_iterations: ", number_training_iterations_)
                    print("train diffusion")
                    print('Progress Policy training: %i %%' % (i / number_training_iterations_ * 100))
                    print("buffer size: ", self.buffer.length())

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                if self.use_ambient_loss:
                    self.compute_loss_AmbientDiffusion(batch)
                else:
                    self.compute_loss(batch)