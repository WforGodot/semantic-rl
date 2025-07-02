# File: dreamerv2/training/config.py

import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict
import hydra
from omegaconf import MISSING

# Following HPs are not a result of detailed tuning.   

@dataclass
class MinAtarConfig():
    '''default HPs that are known to work for MinAtar envs '''
    #env desc
    env : str = MISSING                                         
    obs_shape: Tuple[int, ...] = MISSING
    action_size: int = MISSING
    pixel: bool = True
    action_repeat: int = 1
    
    #buffer desc
    capacity: int = int(1e5)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(1e5)
    train_every: int = 50                                  #reduce this to potentially improve sample requirements
    collect_intervals: int = 5 
    batch_size: int = 50 
    seq_len: int = 50
    eval_episode: int = 4
    eval_render: bool = True
    save_every: int = int(1e5)
    seed_steps: int = 4000
    model_dir: int = 'results'
    gif_dir: int = 'results'
    
    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 200
    rssm_node_size: int = 200
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':200, 'stoch_size':20, 'class_size':20, 'category_size':20, 'min_std':0.1})
    
    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 10
    lr: Dict = field(default_factory=lambda:{'model':4e-4, 'actor':8e-5, 'critic':2e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':0.1, 'reward':1.0, 'discount':5.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 100
    slow_target_fraction: float = 1.00

    #actor critic
    actor={'layers':2, 'node_size':64, 'dist':'one_hot', 'init_std':3, 'mean_scale':3, 'activation':nn.ELU}
    critic={'layers':2, 'node_size':64, 'dist':'normal', 'activation':nn.ELU}
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':7000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder={'depth':8, 'kernel':3, 'activation':nn.ELU,
                 'num_slots':7, 'slot_dim':64, 'slot_iters':3},
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':3, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})



# ─── New Slot-Attention Bias Weights Config ─────────────────────────────────
@dataclass
class SlotAttentionLossConfig:
    """Weights for slot-attention mask regularizers"""
    box: float = 1e-2       # bounding-box variance loss weight
    tv: float = 1e-2        # spatial continuity (total variation) loss weight
    temporal: float = 1e-2  # temporal smoothness loss weight

@dataclass
class SlotAttentionConfig:
    loss: SlotAttentionLossConfig = field(default_factory=SlotAttentionLossConfig)


# ---------------------------------------------------------------------
# MiniGridConfig – tuned for small-GPU speed & stability
# ---------------------------------------------------------------------
@dataclass
class MiniGridConfig:
    """Dreamer-V2 defaults that work on a 6 GiB GPU.

    – batch × seq_len chosen so batch-update activations fit <4 GiB
    – narrower RSSM + heads to cut parameter & activation size
    – shorter episode timeout (300) for quicker resets
    """

    # ─── Environment ────────────────────────────────────────────────
    env: str
    obs_shape: Tuple
    action_size: int
    id: int = 0
    seed: int = 123
    device: str = "cuda"
    pixel: bool = True          # set False if you switch to FlatObsWrapper
    action_repeat: int = 1
    time_limit: int = 300       # was 1000

    # ─── Replay Buffer ─────────────────────────────────────────────
    capacity: int = 2000    # was 1e5; longer runs need more
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    # ─── Training Loop ─────────────────────────────────────────────
    train_steps: int = int(5e3)   # more total frames but cheaper per update
    train_every: int = 8         # update every 16 env steps (was 5)
    collect_intervals: int = 4
    batch_size: int = 16          # **biggest memory lever** (was 50)
    seq_len: int = 16             # second lever (was 8 → 50)
    eval_episode: int = 5
    eval_render: bool = False
    save_every: int = int(1e3)
    seed_steps: int = 1000        # lower because buffer is smaller
    seed_episodes: int = 5
    model_dir: str = "results"
    gif_dir: str = "results"

    # ─── Latent Space / RSSM ───────────────────────────────────────
    rssm_type: str = "discrete"
    embedding_size: int = 128       # was 100
    rssm_node_size: int = 128       # narrower → faster
    rssm_info: Dict = field(
        default_factory=lambda: {
            "deter_size": 128,      # was 100
            "stoch_size": 32,       # was 256
            "class_size": 16,
            "category_size": 16,
            "min_std": 0.1,
        }
    )

    # ─── Optimisation Objective ───────────────────────────────────
    grad_clip: float = 40.0         # lower clip for narrow nets
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 6
    lr: Dict = field(
        default_factory=lambda: {
            "model": 3e-4,          # +50 % helps smaller nets
            "actor": 1e-4,
            "critic": 1e-4,
        }
    )
    loss_scale: Dict = field(
        default_factory=lambda: {"kl": 2.0, "reward": 1.0, "discount": 5.0, "obs": 1.0}
    )
    kl: Dict = field(
        default_factory=lambda: {
            "use_kl_balance": True,
            "kl_balance_scale": 0.6,
            "use_free_nats": False,
            "free_nats": 1.0
        }
    )
    use_slow_target: bool = True
    slow_target_update: int = 50
    slow_target_fraction: float = 1.0

     # ─── Slot-Attention Bias Weights ───────────────────────────────────────────
    slot_attention: SlotAttentionConfig = field(default_factory=SlotAttentionConfig)

    # ─── Actor / Critic Heads ─────────────────────────────────────
    actor: Dict = field(
        default_factory=lambda: {
            "layers": 3,
            "node_size": 64,        # was 100
            "dist": "one_hot",
            "min_std": 1e-4,
            "init_std": 5,
            "mean_scale": 5,
            "activation": nn.ELU,
        }
    )
    critic: Dict = field(
        default_factory=lambda: {
            "layers": 3,
            "node_size": 64,
            "dist": "normal",
            "activation": nn.ELU,
        }
    )
    expl: Dict = field(
        default_factory=lambda: {
            "train_noise": 0.3,     # slightly less noise with small seq_len
            "eval_noise": 0.0,
            "expl_min": 0.05,
            "expl_decay": 7000.0,   # faster decay
            "expl_type": "epsilon_greedy",
        }
    )
    actor_grad: str = "reinforce"
    actor_grad_mix: float = 0.0
    actor_entropy_scale: float = 1e-3

    # ─── World-Model Heads ────────────────────────────────────────
    obs_encoder: Dict = field(
        default_factory=lambda: {
            "layers": 3,
            "node_size": 64,
            "dist": None,
            "activation": nn.ELU,
            "kernel": 3,
            "depth": 8,             # half of 16
            # ─── Slot‐Attention params ───
            "num_slots": 10,        
            "slot_dim": 32,         # smaller per-slot embedding
            "slot_iters": 3,        # two attention iterations
        }
    )
    obs_decoder: Dict = field(
        default_factory=lambda: {
            "layers": 3,
            "node_size": 64,
            "dist": "normal",
            "activation": nn.ELU,
            "kernel": 3,
            "depth": 8,
        }
    )
    reward: Dict = field(
        default_factory=lambda: {
            "layers": 3,
            "node_size": 64,
            "dist": "normal",
            "activation": nn.ELU,
        }
    )
    discount: Dict = field(
        default_factory=lambda: {
            "layers": 3,
            "node_size": 64,
            "dist": "binary",
            "activation": nn.ELU,
            "use": True,
        }
    )


    
