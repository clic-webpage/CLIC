import torch 
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
"""
Functions that selects the agent
"""
from omegaconf import DictConfig

def agent_selector(agent_type: str, config_agent: DictConfig):
    # CLIC family
    algorithm_type = config_agent.algorithm
    if agent_type == 'CLIC':  # CLIC with an Energy-based model (EBM) as policy representation
    
        if config_agent.use_tensorflow_version:
            from agents.CLIC_tf import CLIC
        else:
            from agents.CLIC_torch import CLIC

        return CLIC(
            shape_meta=config_agent.shape_meta,
            dim_a=config_agent.dim_a,dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_max_size=config_agent.buffer_max_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            e_matrix=config_agent.e,
            loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
            sphere_alpha=config_agent.sphere_alpha,
            sphere_gamma=config_agent.sphere_gamma,
            radius_ratio = config_agent.radius_ratio,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy,
            number_training_iterations=config_agent.number_training_iterations,
            desiredA_type=config_agent.desiredA_type,
            softmax_temperature= config_agent.softmax_temperature,
            sample_action_number = config_agent.sample_action_number,
        )
    
    elif agent_type == 'CLIC_Explicit': # CLIC with Gaussian-parameterized policy
        from agents.CLIC_Explicit_tf import CLIC_Explicit
        return CLIC_Explicit(
            dim_a=config_agent.dim_a,dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_max_size=config_agent.buffer_max_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            e_matrix=config_agent.e,
            loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
            sphere_alpha=config_agent.sphere_alpha,
            sphere_gamma=config_agent.sphere_gamma,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy,
            number_training_iterations=config_agent.number_training_iterations,
        )

    elif agent_type == 'BD_COACH':  # A baseline that learns from relative corrections, with a Gaussian-parameterized policy
        from agents.BD_COACH import BD_COACH
        return BD_COACH(
            dim_a=config_agent.dim_a,dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_max_size=config_agent.buffer_max_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            human_model_learning_rate = config_agent.human_model_learning_rate,
            e_matrix=config_agent.e,
            loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy,
            number_training_iterations=config_agent.number_training_iterations,
        )

    # HG-DAgger: A baseline that learns from human interventions, with a Gaussian-parameterized policy
    elif agent_type == 'HG_DAgger':
        from agents.HG_DAgger import HG_DAGGER
        return HG_DAGGER(
            dim_a=config_agent.dim_a,
            dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_max_size=config_agent.buffer_max_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            number_training_iterations=config_agent.number_training_iterations,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy
        )

    # implicit Behavior cloning: policy represented as an EBM 
    elif agent_type == 'Implicit_BC':
        if not config_agent.use_tensorflow_version:
            from agents.ibc_torch import ibc
            return ibc(
                dim_a=config_agent.dim_a,dim_o=config_agent.dim_o,
                action_upper_limits=config_agent.action_upper_limits,
                action_lower_limits=config_agent.action_lower_limits,
                buffer_min_size=config_agent.buffer_min_size,
                buffer_max_size=config_agent.buffer_max_size,
                buffer_sampling_rate=config_agent.buffer_sampling_rate,
                buffer_sampling_size=config_agent.buffer_sampling_size,
                train_end_episode=config_agent.train_end_episode,
                policy_model_learning_rate=config_agent.policy_model_learning_rate,
                saved_dir=config_agent.saved_dir,
                sample_action_number = config_agent.sample_action_number,
                load_dir=config_agent.load_dir,
                load_policy=config_agent.load_policy,
                number_training_iterations=config_agent.number_training_iterations,
            )
        else:
            from agents.ibc import ibc
            return ibc(dim_a=config_agent.dim_a,
                         dim_o=config_agent.dim_o,
                         action_upper_limits=config_agent.action_upper_limits,
                         action_lower_limits=config_agent.action_lower_limits,
                         buffer_min_size=config_agent.buffer_min_size,
                         buffer_max_size=config_agent.buffer_max_size,
                         buffer_sampling_rate=config_agent.buffer_sampling_rate,
                        buffer_sampling_size=config_agent.buffer_sampling_size,
                         number_training_iterations = config_agent.number_training_iterations,
                         train_end_episode=config_agent.train_end_episode,
                         policy_model_learning_rate=config_agent.policy_model_learning_rate,
                        saved_dir=config_agent.saved_dir,
                         load_dir=config_agent.load_dir,
                        load_policy=config_agent.load_policy,)

    # PVP: policy represented as an EBM 
    elif agent_type == 'PVP':
        from agents.PVP import pvp
        return pvp(
            dim_a=config_agent.dim_a,
            dim_o=config_agent.dim_o,
            action_upper_limits=config_agent.action_upper_limits,
            action_lower_limits=config_agent.action_lower_limits,
            buffer_min_size=config_agent.buffer_min_size,
            buffer_sampling_rate=config_agent.buffer_sampling_rate,
            buffer_sampling_size=config_agent.buffer_sampling_size,
            train_end_episode=config_agent.train_end_episode,
            policy_model_learning_rate=config_agent.policy_model_learning_rate,
            e_matrix=config_agent.e,
            loss_weight_inverse_e=config_agent.loss_weight_inverse_e,
            buffer_max_size=config_agent.buffer_max_size,
            saved_dir=config_agent.saved_dir,
            load_dir=config_agent.load_dir,
            load_policy=config_agent.load_policy,
            number_training_iterations=config_agent.number_training_iterations
        )

    # Diffusion-based policies
    elif agent_type.startswith('Diffusion'):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from agents.diffusion_policy import DiffusionUnetLowdimPolicy
        if agent_type == 'Diffusion':
            return DiffusionUnetLowdimPolicy(
                noise_scheduler=DDPMScheduler(
                    num_train_timesteps=config_agent.DDPM_num_train_timesteps, beta_start=0.0001, beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2', variance_type='fixed_small',
                    clip_sample=True, prediction_type='epsilon'
                ),
                obs_dim=config_agent.dim_o,
                action_dim=config_agent.dim_a,
                horizon=config_agent.Ta,
                saved_dir=config_agent.saved_dir,
                load_dir=config_agent.load_dir,
                load_pretrained_dir=config_agent.load_pretrained_dir,
                load_policy=config_agent.load_policy,
                number_training_iterations=config_agent.number_training_iterations,
                buffer_min_size=config_agent.buffer_min_size,
                buffer_max_size=config_agent.buffer_max_size,
                buffer_sampling_size=config_agent.buffer_sampling_size,
                policy_model_learning_rate=config_agent.policy_model_learning_rate,
                use_ambient_loss = config_agent.use_ambient_loss,
                ambient_k = config_agent.ambient_k,
            )
    else:
        raise NameError(f'Unknown agent type: {agent_type}')

