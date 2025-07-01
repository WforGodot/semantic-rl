#!/usr/bin/env python3
"""
run_env.py

Generic entry-point for running DreamerV2 on any Gym environment (RajGhugare19/dreamerv2).
Usage: python run_env.py --env CartPole-v1 --device cpu --steps 100000
"""
# Monkey-patch for numpy compatibility: Gym's passive_env_checker expects np.bool8
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import argparse
from dreamerv2.training.config import MinAtarConfig, MiniGridConfig
from dreamerv2.training.trainer import Trainer

def wrap_env(env):
    """Wraps new Gym API (obs, reward, terminated, truncated, info) to old API (obs, reward, done, info)"""
    orig_step = env.step
    def step_compat(action):
        obs, reward, terminated, truncated, info = orig_step(action)
        done = terminated or truncated
        return obs, reward, done, info
    env.step = step_compat
    return env

def make_config(env_id, steps):
    # Instantiate environment to infer specs
    env = wrap_env(gym.make(env_id))
    obs_shape = env.observation_space.shape
    action_space = env.action_space
    # Determine action size for discrete or continuous
    if hasattr(action_space, 'n'):
        action_size = action_space.n
    else:
        action_size = action_space.shape[0]
    # Choose appropriate config class
    if 'MiniGrid' in env_id or 'minigrid' in env_id.lower():
        cfg = MiniGridConfig(env=env_id,
                             obs_shape=obs_shape,
                             action_size=action_size)
    else:
        cfg = MinAtarConfig(env=env_id,
                             obs_shape=obs_shape,
                             action_size=action_size)
        # For non-image envs (vector obs), disable pixel mode
        if len(obs_shape) == 1:
            cfg.pixel = False
    # Override number of training frames if provided
    if steps is not None:
        cfg.train_steps = steps
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, help='Gym environment id')
    parser.add_argument('--device', default='cpu', help='Torch device (cpu or cuda)')
    parser.add_argument('--steps', type=int, default=None, help='Total training frames')
    args = parser.parse_args()

    # Prepare configuration and trainer
    cfg = make_config(args.env, args.steps)
    trainer = Trainer(cfg, device=args.device)

    # Seed episodes for replay buffer
    env = wrap_env(gym.make(args.env))
    trainer.collect_seed_episodes(env)
    frames = 0
    # Main training loop
    while frames < cfg.train_steps:
        metrics = trainer.train_batch({})
        frames += cfg.train_every * cfg.batch_size
        print(f"Frames {frames}/{cfg.train_steps} | Metrics: {metrics}")
    env.close()

if __name__ == '__main__':
    main()
