import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from dreamerv2.models.slot_attention import SlotAttention



class ObsEncoder(nn.Module):
    """
    Slot‐Attention based encoder.
    Takes (B, C, H, W) → K slot vectors → flatten → embedding_size.
    """
    def __init__(self, input_shape, embedding_size, info):
        super().__init__()
        # 1) Conv backbone (same channels as before)
        activation = info['activation']
        d          = info['depth']
        k          = info['kernel']
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],     d, k), activation(),
            nn.Conv2d(d, 2*d, k),                activation(),
            nn.Conv2d(2*d, 4*d, k),              activation(),
        )

        # 2) Slot‐Attention module
        num_slots = info.get('num_slots', 7)
        slot_dim  = info.get('slot_dim', 64)
        iters     = info.get('slot_iters', 3)
        # Map each spatial feature to slot_dim
        self.mapper     = nn.Linear(4*d, slot_dim)
        self.slot_attn  = SlotAttention(num_slots=num_slots,
                                         dim=slot_dim,
                                         iters=iters)

        # 3) Final projection to your embedding_size
        flat_dim = num_slots * slot_dim
        if flat_dim == embedding_size:
            self.fc_out = nn.Identity()
        else:
            self.fc_out = nn.Linear(flat_dim, embedding_size)

    def forward(self, obs):
        # obs: (..., C, H, W)   (e.g. [batch, seq_len, C, H, W])
        # 1) collapse all leading dims into one:
        *batch_shape, C, H, W = obs.shape
        x = obs.view(-1, C, H, W)                 # (B_flat, C, H, W)

        # 2) convolutional backbone
        x = self.conv(x)                          # (B_flat, 4d, H', W')
        Bflat, Cd, Hp, Wp = x.shape

        # 3) prepare for slot-attn: (B_flat, N_patches, feat)
        x = x.view(Bflat, Cd, Hp*Wp).permute(0, 2, 1) # (Bf, N_patches, Cd)
        x = self.mapper(x)                        # (B_flat, N_patches, slot_dim)

        # 4) slot-attention
        slots = self.slot_attn(x)                  # (Bf, num_slots, slot_dim)
        # ─── store attention maps for logging ───
        # `self.slot_attn.attn` is shape (Bf, num_slots, N_patches)
        self.slot_attn.last_attn   = self.slot_attn.attn.detach()  # detach to avoid gradients
        self.slot_attn.last_Hp_Wp  = (Hp, Wp)

        # 5) flatten slots  project
        Bf, K, D = slots.shape
        slots = slots.view(Bf, K*D)               # (B_flat, num_slots*slot_dim)
        slots = self.fc_out(slots)                # (B_flat, embedding_size)

        # 6) restore original batch dims
        return slots.view(*batch_shape, -1)       # (..., embedding_size)       

class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, info):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        conv1_shape = conv_out_shape(output_shape[1:], 0, k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, k, 1)
        self.conv_shape = (4*d, *conv3_shape)
        self.output_shape = output_shape
        if embed_size == np.prod(self.conv_shape).item():
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(embed_size, np.prod(self.conv_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*d, 2*d, k, 1),
            activation(),
            nn.ConvTranspose2d(2*d, d, k, 1),
            activation(),
            nn.ConvTranspose2d(d, c, k, 1),
        )

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))



"""


class ObsEncoder_old(nn.Module):
    def __init__(self, input_shape, embedding_size, info):
        super(ObsEncoder, self).__init__()
        self.shape = input_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        self.k = k
        self.d = d
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_shape[0], d, k),
            activation(),
            nn.Conv2d(d, 2*d, k),
            activation(),
            nn.Conv2d(2*d, 4*d, k),
            activation(),
        )
        if embedding_size == self.embed_size:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(self.embed_size, embedding_size)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, self.k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, self.k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, self.k, 1)
        embed_size = int(4*self.d*np.prod(conv3_shape).item())
        return embed_size


        """