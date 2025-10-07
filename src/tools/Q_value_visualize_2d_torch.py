import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_normalized_q_values_torch_2d_contour(
    model,
    state,
    a_negative,
    a_positive,
    saved_path,
    saved_data_path=None,
    sampled_action=None,
    resolution=200,
    draw_desiredA=True,
    temperature=0.2,
):
    """
    Visualizes the normalized Q-values for a given state over a 2D action space,
    using a contour map, but implemented in PyTorch.

    Parameters:
    - model: a PyTorch nn.Module, mapping (state, action) -> Q-value
    - state: 1D tensor of shape [state_dim]
    - a_negative: tuple/list of length 2 (robot action)
    - a_positive: tuple/list of length 2 (teacher action)
    - saved_path: filepath to save the figure (e.g. 'q_contour.png')
    - saved_data_path: optionally .npz file to dump grid & Q-values
    - sampled_action: optional Tensor [N,2] of actions to scatter
    - resolution: number of points per axis
    - draw_desiredA: whether to overlay a_negative/a_positive + perpendicular line
    - temperature: softmax temperature
    """
    device = next(model.parameters()).device
    model.eval()

    # 1) build action grid
    x = torch.linspace(-1, 1, resolution, device=device)
    y = torch.linspace(-1, 1, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    flat_actions = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [R*R, 2]

    # 2) repeat state
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float()
    state = state.to(device)#.unsqueeze(0)                  # [1, state_dim]
    state_batch = state.repeat(flat_actions.size(0), 1)    # [R*R, state_dim]

    # 3) forward pass
    with torch.no_grad():
        # adjust this if your model takes concatenated inputs or two args
        q_values = model(state_batch, flat_actions)        # [R*R] or [R*R,1]
        q_values = q_values.view(-1)

        # 4) softmax normalization
        q_norm = torch.softmax(q_values / temperature, dim=0)
        q_min, q_max = q_norm.min(), q_norm.max()
        q_norm = (q_norm - q_min) / (q_max - q_min)

    Q = q_norm.cpu().numpy().reshape(resolution, resolution)
    X = grid_x.cpu().numpy()
    Y = grid_y.cpu().numpy()

    # 5) plot
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contour(X, Y, Q, levels=5, cmap='viridis', linewidths=2.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Q-value', rotation=270, labelpad=15)

    # 6) desired actions + perpendicular line
    if draw_desiredA:
        an = np.array(a_negative)
        ap = np.array(a_positive)
        ax.scatter(*an, color='black', marker='o', s=100, label='Robot Action')
        ax.scatter(*ap, color='red',   marker='*', s=100, label='Teacher Action')

        mid = (an + ap) / 2
        v = ap - an
        perp = np.array([-v[1], v[0]])
        perp = perp / np.linalg.norm(perp)
        scale = 1.5
        start = mid - scale * perp
        end   = mid + scale * perp
        ax.plot([start[0], end[0]], [start[1], end[1]],
                'blue', linestyle='--', linewidth=1, label='Perpendicular Line')

    # 7) sampled actions
    if sampled_action is not None:
        sa = sampled_action.cpu().numpy()
        ax.scatter(sa[:, 0], sa[:, 1],
                   color='green', marker='x', s=50, label='Sampled Actions')

    # 8) formatting
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Action Dimension 1')
    ax.set_ylabel('Action Dimension 2')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.grid(True)

    # 9) save
    plt.savefig(saved_path)
    plt.close(fig)

    if saved_data_path is not None:
        np.savez(
            saved_data_path,
            x=X,
            y=Y,
            q_values=Q,
            a_negative=a_negative,
            a_positive=a_positive,
            sampled_action=(sampled_action.cpu().numpy() 
                            if sampled_action is not None else None)
        )
