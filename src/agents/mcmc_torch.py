# """MCMC algorithms to optimize samples from EBMs."""

import collections
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

# Global range alias for debugging
my_range = range


def gradient_wrt_act(
    energy_network,
    observations,
    actions,
    training,
    network_state,
    tfa_step_type,
    apply_exp,
    obs_encoding=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dE(obs,act)/dact, also return energy."""
    # Ensure actions require gradient
    actions = actions.clone().detach().requires_grad_(True)

    # Forward pass
    if obs_encoding is not None:
        energies, _ = energy_network(
            (observations, actions),
            training=training,
            network_state=network_state,
            step_type=tfa_step_type,
            observation_encoding=obs_encoding
        )
    else:
        if training:
            energy_network.train()
        else:
            energy_network.eval()
        energies = energy_network(observations, actions)

    if apply_exp:
        energies = torch.exp(energies)

    # Gradients
    denergies_dactions = torch.autograd.grad(
        outputs=energies.sum(),
        inputs=actions,
        create_graph=True,
        retain_graph=True
    )[0] * -1.0

    return denergies_dactions, energies.detach()


def compute_grad_norm(grad_norm_type, de_dact):
    """Given de_dact and the type, compute the norm."""
    if grad_norm_type is not None:
        grad_type = {'1': 1, '2': 2, 'inf': float('inf')}[grad_norm_type]
        grad_norms = torch.norm(de_dact, p=grad_type, dim=1)
    else:
        grad_norms = torch.zeros(de_dact.size(0), device=de_dact.device)
    return grad_norms


def langevin_step(
    energy_network,
    observations,
    actions,
    training,
    policy_state,
    tfa_step_type,
    noise_scale,
    grad_clip,
    delta_action_clip,
    stepsize,
    apply_exp,
    min_actions,
    max_actions,
    grad_norm_type,
    obs_encoding
):
    """Single step of Langevin update."""
    l_lambda = 1.0
    de_dact, energies = gradient_wrt_act(
        energy_network,
        observations,
        actions,
        training,
        policy_state,
        tfa_step_type,
        apply_exp,
        obs_encoding
    )

    # Scale delta clip
    delta_clip_scaled = delta_action_clip * 0.5 * (max_actions - min_actions)

    unclipped = de_dact.clone()
    grad_norms = compute_grad_norm(grad_norm_type, unclipped)

    if grad_clip is not None:
        de_dact = torch.clamp(de_dact, -grad_clip, grad_clip)

    de_dact = 0.5 * l_lambda * de_dact + torch.randn_like(actions) * l_lambda * noise_scale
    delta_actions = stepsize * de_dact

    # Clip and update
    delta_actions = torch.clamp(delta_actions, -delta_clip_scaled, delta_clip_scaled)
    actions = actions - delta_actions
    actions = actions.clamp(min=min_actions, max=max_actions)

    return actions.detach(), energies, grad_norms


class ExponentialSchedule:
    def __init__(self, init, decay):
        self._decay = decay
        self._latest = init

    def get_rate(self, index):
        self._latest *= self._decay
        return self._latest


class PolynomialSchedule:
    def __init__(self, init, final, power, num_steps):
        self._init = init
        self._final = final
        self._power = power
        self._num_steps = num_steps

    def get_rate(self, index):
        frac = 1.0 - (index / float(self._num_steps - 1))
        return ((self._init - self._final) * (frac ** self._power)) + self._final


def update_chain_data(
    num_iterations,
    step_index,
    actions,
    energies,
    grad_norms,
    full_chain_actions,
    full_chain_energies,
    full_chain_grad_norms
):
    """Helper to record data during MCMC."""
    mask = F.one_hot(torch.full_like(energies, step_index, dtype=torch.long), num_iterations).float()
    full_chain_energies += energies * mask
    full_chain_grad_norms += grad_norms * mask

    mask_actions = mask.unsqueeze(-1).expand(-1, -1, actions.size(-1))
    actions_exp = actions.unsqueeze(0).expand_as(mask_actions)
    full_chain_actions += actions_exp * mask_actions

    return full_chain_actions, full_chain_energies, full_chain_grad_norms


def langevin_actions_given_obs(
    energy_network,
    observations,
    action_samples,
    policy_state,
    min_actions,
    max_actions,
    num_action_samples,
    num_iterations=25,
    training=False,
    tfa_step_type=(),
    sampler_stepsize_init=1e-1,
    sampler_stepsize_decay=0.8,
    noise_scale=1.0,
    grad_clip=None,
    delta_action_clip=0.1,
    stop_chain_grad=True,
    apply_exp=False,
    use_polynomial_rate=True,
    sampler_stepsize_final=1e-5,
    sampler_stepsize_power=2.0,
    return_chain=False,
    grad_norm_type='inf',
    late_fusion=False
):
    """Perform Langevin MCMC sampling of actions given observations."""
    stepsize = sampler_stepsize_init
    if use_polynomial_rate:
        schedule = PolynomialSchedule(sampler_stepsize_init, sampler_stepsize_final,
                                      sampler_stepsize_power, num_iterations)
    else:
        schedule = ExponentialSchedule(sampler_stepsize_init, sampler_stepsize_decay)

    actions = action_samples.clone()
    b_times_n = actions.size(0)
    act_dim = actions.size(-1)

    full_chain_actions = torch.zeros((num_iterations, b_times_n, act_dim), device=actions.device)
    full_chain_energies = torch.zeros((num_iterations, b_times_n), device=actions.device)
    full_chain_grad_norms = torch.zeros((num_iterations, b_times_n), device=actions.device)

    for step_index in my_range(num_iterations):
        actions, energies, grad_norms = langevin_step(
            energy_network,
            observations,
            actions,
            training,
            policy_state,
            tfa_step_type,
            noise_scale,
            grad_clip,
            delta_action_clip,
            stepsize,
            apply_exp,
            min_actions,
            max_actions,
            grad_norm_type,
            None
        )
        if stop_chain_grad:
            actions = actions.detach()
        stepsize = schedule.get_rate(step_index + 1)

        if return_chain:
            (full_chain_actions, full_chain_energies,
             full_chain_grad_norms) = update_chain_data(
                num_iterations, step_index,
                actions, energies, grad_norms,
                full_chain_actions, full_chain_energies,
                full_chain_grad_norms
            )

    if return_chain:
        ChainData = collections.namedtuple('ChainData', ['actions', 'energies', 'grad_norms'])
        chain_data = ChainData(full_chain_actions, full_chain_energies, full_chain_grad_norms)
        return actions, chain_data
    return actions


def get_probabilities(
    energy_network,
    batch_size,
    num_action_samples,
    observations,
    actions,
    training,
    temperature=0.1
):
    """Get probabilities to post-process Langevin results."""
    # net_logits = energy_network([observations, actions], training=training)
    if training:
        energy_network.train()
    else:
        energy_network.eval()
    net_logits = energy_network(observations, actions)
    net_logits = net_logits.view(batch_size, num_action_samples)
    probs = F.softmax(net_logits / temperature, dim=1)
    return probs.view(-1)