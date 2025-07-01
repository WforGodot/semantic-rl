#!/usr/bin/env python
"""eval_mg.py — Evaluate a trained Dreamer‑V2 MiniGrid agent.

Usage example (save videos & random frames):
    python test/eval_mg.py --env MiniGrid-Empty-5x5-v0 --model_dir results/MiniGrid-Empty-5x5-v0_0/models --episodes 10 --save_videos --save_frames

By default the script looks for the checkpoint file `models_best.pth`
under --model_dir; if that is absent it falls back to the most recently
modified `models_*.pth` file.
"""

import argparse
import glob
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import gym as gym_old  # classic gym for Discrete wrapper
import numpy as np
import torch
from PIL import Image

from dreamerv2.utils.minigrid_wrapper import GymMiniGrid
from dreamerv2.utils.wrapper import OneHotAction
from dreamerv2.training.config import MiniGridConfig
from dreamerv2.training.trainer import Trainer


def latest_checkpoint(model_dir: Path) -> Path | None:
    """Return the preferred checkpoint path in *model_dir*.
    Priority: models_best.pth > newest models_*.pth."""
    best = model_dir / "models_best.pth"
    if best.exists():
        return best
    cks = sorted(model_dir.glob("models_*.pth"), key=lambda p: p.stat().st_mtime)
    return cks[-1] if cks else None


def build_env(env_name: str, tile_size: int = 11, resize_shape: tuple[int, int] = (84, 84)):
    env = GymMiniGrid(env_name, tile_size=tile_size, resize_shape=resize_shape)
    # One‑hot wrapper expects a classic‑gym Discrete space
    env.action_space = gym_old.spaces.Discrete(env.action_space.n)
    action_size = env.action_space.n  # capture number of discrete actions BEFORE one-hot encoding
    env = OneHotAction(env)
    return env, action_size


def save_frame(array_c_h_w: np.ndarray, path: Path):
    """Save a (C,H,W) uint8 array as PNG under *path*."""
    img_hwc = np.transpose(array_c_h_w, (1, 2, 0))  # → H,W,C
    Image.fromarray(img_hwc).save(path)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # 1. Environment ---------------------------------------------------------
    env, action_size = build_env(args.env)
    obs_shape = env.observation_space.shape

    # 2. Dummy config + Trainer (to host model) ------------------------------
    cfg = MiniGridConfig(
        env=args.env,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=np.uint8,
        action_dtype=np.float32,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
    )
    trainer = Trainer(cfg, device)

    # 3. Load checkpoint -----------------------------------------------------
    ckpt = latest_checkpoint(Path(args.model_dir))
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {args.model_dir}")
    trainer.load_model(str(ckpt))
    trainer.eval()  # eval‑mode
    print(f"Loaded checkpoint: {ckpt}")

    # 4. Optional video/frames ----------------------------------------------
    if args.save_videos:
        vdir = Path(args.output_dir) / "videos"
        vdir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, str(vdir), episode_trigger=lambda e: True)

    fdir = Path(args.output_dir) / "frames"
    if args.save_frames:
        fdir.mkdir(parents=True, exist_ok=True)

    # 5. Evaluation loop -----------------------------------------------------
    returns: list[float] = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        prev_state = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size, device=device)
        done = False
        ep_return = 0.0
        t = 0

        while not done:
            with torch.no_grad():
                tens_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                embed = trainer.ObsEncoder(tens_obs)
                _, post_state = trainer.RSSM.rssm_observe(embed, prev_action, True, prev_state)
                model_state = trainer.RSSM.get_model_state(post_state)
                action, _ = trainer.ActionModel(model_state)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            ep_return += reward

            # save random frame ------------------------------------------------
            if args.save_frames and random.random() < args.frame_prob:
                fname = f"ep{ep:03d}_step{t:04d}.png"
                save_frame(next_obs, fdir / fname)

            obs = next_obs
            prev_state = post_state
            prev_action = action
            t += 1

        returns.append(ep_return)
        print(f"Episode {ep + 1}/{args.episodes}  return = {ep_return:.2f}")

    env.close()
    print("Mean return:", np.mean(returns))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0")
    p.add_argument("--model_dir", type=str, default="results/MiniGrid-Empty-5x5-v0_0/models")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="eval_outputs")
    p.add_argument("--save_videos", action="store_true", help="record MP4 videos of each episode")
    p.add_argument("--save_frames", action="store_true", help="save random individual frames as PNGs")
    p.add_argument("--frame_prob", type=float, default=0.05, help="probability of saving any given frame (0‑1)")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
