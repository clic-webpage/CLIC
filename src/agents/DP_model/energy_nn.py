import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

from agents.DP_model.diffusion.conditional_unet1d_original import ConditionalResidualBlock1D

from agents.DP_model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
    
class ActionValueFunctionModel(nn.Module):
    """
    PyTorch implementation of the TF Keras action-value function model.
    Takes state and action inputs, concatenates them, and passes through
    successive Linear -> ReLU -> LayerNorm layers, ending in a scalar Q-value.
    """
    def __init__(self, dim_o: int, dim_a: int):
        super(ActionValueFunctionModel, self).__init__()
        self.dim_o = dim_o
        # self.dim_o = 2083
        self.dim_a = dim_a

        # Define network dimensions
        input_dim = dim_o + dim_a
        # input_dim =2083
        hidden_dims = [512, 512, 512, 256]

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        # Layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        # Layer 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.ln3 = nn.LayerNorm(hidden_dims[2])
        # Layer 4
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.ln4 = nn.LayerNorm(hidden_dims[3])
        # Output layer: scalar Q-value
        self.fc_out = nn.Linear(hidden_dims[3], 1)

        # Initialize output layer weights and bias to zero for stable start
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param state: Tensor of shape (batch_size, dim_o)
        :param action: Tensor of shape (batch_size, dim_a)
        :return: Q-value tensor of shape (batch_size, 1)
        """
        # Concatenate state and action along last dim
        x = torch.cat((state, action), dim=1)
        # print("x ", x.shape)

        # Hidden layers with ReLU then LayerNorm
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = F.relu(self.fc3(x))
        x = self.ln3(x)
        x = F.relu(self.fc4(x))
        x = self.ln4(x)

        # Output Q-value
        q_value = self.fc_out(x)
        return q_value

class UnetEncoder_QModel(nn.Module):
    """
    Q-network that reuses the encoder (down + mid blocks) of a UNet
    to process an H-step action chunk, then collapses temporally
    via purely CNN-based pooling, fuses with a state embedding,
    and outputs a scalar Q-value.
    """
    def __init__(self, 
        input_dim:          int,
        # if you want to pass per-time local conditioning (else leave None)
        local_cond_dim:     int    = None,
        # UNet encoder parameters
        down_dims:          list   = [256, 512, 1024],
        kernel_size:        int    = 3,
        n_groups:           int    = 8,
        cond_predict_scale: bool   = False,
        # intermediate channel dim for collapse
        attn_dim:           int    = 256,
        # state embedding + MLP
        dim_o:              int    = 100,
        state_emb_dim:      int    = 256,
        mlp_dims:           list   = [512, 256],
    ):
        super().__init__()
        self.input_dim = input_dim
        all_dims = [input_dim] + down_dims
        cond_dim = state_emb_dim  # use state embedding as “global cond”

        # ——— local-condition encoder (optional) ———
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            in_dim, out_dim = all_dims[0], all_dims[1]
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    local_cond_dim, out_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock1D(
                    local_cond_dim, out_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
            ])

        # ——— build the down-path ———
        self.down_modules = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = (i == len(all_dims)-2)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    c_in, c_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                ConditionalResidualBlock1D(
                    c_out, c_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                ),
                Downsample1d(c_out) if not is_last else nn.Identity()
            ]))

        # ——— mid-blocks ———
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
            for _ in range(2)
        ])

        # ——— reduce channels for temporal‐CNN pooling ———
        C = down_dims[-1]
        self.dim_reducer   = Conv1dBlock(C, attn_dim, kernel_size=1)
        # self.collapse_conv = Conv1dBlock(attn_dim, attn_dim, kernel_size=3)

        # ——— state embedding + final MLP ———
        self.state_fc  = nn.Linear(dim_o, state_emb_dim)
        self.ln_state  = nn.LayerNorm(state_emb_dim)

        joint_dim     = state_emb_dim + attn_dim
        self.fc1      = nn.Linear(joint_dim, mlp_dims[0])
        self.ln1      = nn.LayerNorm(mlp_dims[0])
        self.fc2      = nn.Linear(mlp_dims[0], mlp_dims[1])
        self.ln2      = nn.LayerNorm(mlp_dims[1])
        self.fc_out   = nn.Linear(mlp_dims[1], 1)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        
        # self._modules

        # print(f"UnetEncoder_QModel parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        # import pdb; pdb.set_trace()


    def forward(self, 
        state:        torch.Tensor,  # (B, dim_o)
        action_chunk: torch.Tensor,  # (B, H, input_dim)
        local_cond:   torch.Tensor = None
    ) -> torch.Tensor:
        B, H, D = action_chunk.shape
        assert D == self.input_dim, f"Expected action_dim={self.input_dim}, got {D}"

        # ——— 1) global cond from state ———
        s = F.relu(self.state_fc(state))
        s = self.ln_state(s)             # (B, state_emb_dim)
        global_feature = s               # passed into each ResBlock

        # ——— 2) optional local-condition encoding ———
        local_feats = None
        if self.local_cond_encoder and local_cond is not None:
            lc = local_cond.transpose(1, 2)  # (B, local_cond_dim, H)
            down_loc, up_loc = self.local_cond_encoder
            local_feats = [
                down_loc(lc, global_feature),
                up_loc(lc, global_feature)
            ]

        # ——— 3) UNet down-path ———
        x = action_chunk.transpose(1, 2)  # (B, input_dim, H)
        for idx, (res1, res2, downsample) in enumerate(self.down_modules):
            x = res1(x, global_feature)
            if idx == 0 and local_feats is not None:
                x = x + local_feats[0]
            x = res2(x, global_feature)
            x = downsample(x)

        # ——— 4) mid-blocks ———
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        
        # import pdb; pdb.set_trace()
        # x is now (B, C, T′). Reduce channels, then pooling via CNN+max:
        x = self.dim_reducer(x)              # (B, attn_dim, T′)
        # x = F.relu(self.collapse_conv(x))    # (B, attn_dim, T′)
        chunk_emb = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, attn_dim)
        # import pdb; pdb.set_trace()
        # ——— 5) fuse with state and predict Q ———
        j = torch.cat([s, chunk_emb], dim=1)
        j = F.relu(self.fc1(j)); j = self.ln1(j)
        j = F.relu(self.fc2(j)); j = self.ln2(j)
        q = self.fc_out(j)                  # (B, 1)
        return q