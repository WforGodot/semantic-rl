# dreamerv2/models/slot_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    """
    A Slot-Attention module that saves its last attention map to self.attn.
    """
    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = None
    ):
        super().__init__()
        # --- BEGIN FIX: initialize cache attributes ----------------------------
        self.attn = None              # soft-assignment maps (batch, slots, patches)
        self.last_attn = None         # will hold the last attn for logging
        self.last_Hp_Wp = None        # placeholder for spatial dims (set by ObsEncoder)
        # --- END FIX ------------------------------------------------------------

        self.num_slots = num_slots
        self.dim       = dim
        self.iters     = iters
        self.eps       = eps
        hidden_dim = hidden_dim or dim

        # slot initialisation params
        self.slots_mu    = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, num_slots, dim))

        # layers for attention
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.project_q   = nn.Linear(dim, dim, bias=False)
        self.project_k   = nn.Linear(dim, dim, bias=False)
        self.project_v   = nn.Linear(dim, dim, bias=False)

        # GRU-based slot update
        self.gru      = nn.GRUCell(dim, hidden_dim)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, inputs: torch.Tensor):
        """
        inputs: (batch, num_patches, dim)
        returns: slots (batch, num_slots, dim)
        and stores self.attn = (batch, num_slots, num_patches)
        """
        b, n, d = inputs.shape
        # 1) initialize slots
        mu    = self.slots_mu.expand(b, -1, -1)
        sigma = self.slots_sigma.expand(b, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # 2) norm inputs once
        inputs = self.norm_inputs(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            # project
            q = self.project_q(slots_norm)    # (b, num_slots, dim)
            k = self.project_k(inputs)        # (b, n, dim)
            v = self.project_v(inputs)        # (b, n, dim)

            # compute attention
            attn_logits = torch.matmul(k, q.transpose(-1, -2))  # (b, n, num_slots)
            attn        = F.softmax(attn_logits, dim=-1) + self.eps
            attn        = attn / attn.sum(dim=-1, keepdim=True)  # (b, n, num_slots)

            # store for logging
            # --- BEGIN FIX: cache soft-attention maps ------------------------------
            self.attn       = attn.transpose(-1, -2)  # (b, num_slots, n_patches)
            self.last_attn  = self.attn               # mirror for external logging
            # --- END FIX ------------------------------------------------------------

            # weighted mean
            updates = torch.matmul(self.attn, v)  # (b, num_slots, dim)

            # GRU update
            updates_flat = updates.view(b * self.num_slots, d)
            slots_flat   = slots_prev.view(b * self.num_slots, d)
            slots        = self.gru(updates_flat, slots_flat)
            slots        = slots.view(b, self.num_slots, d)
            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots
