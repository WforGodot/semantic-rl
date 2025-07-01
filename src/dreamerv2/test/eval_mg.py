# File: test/eval_mg.py
import argparse
import os
import torch
import numpy as np
import gymnasium as gym           # modern Gymnasium API for envs
import gym as gym_old             # classic Gym for Discrete action-space conversion
from dreamerv2.utils.minigrid_wrapper import GymMiniGrid
from dreamerv2.utils.wrapper import OneHotAction
from dreamerv2.utils.reward_wrappers import RewardShapingWrapper
from dreamerv2.training.config import MiniGridConfig
from dreamerv2.training.trainer import Trainer
import imageio
import torch.nn.functional as F


def to_tensor(x, device):
    """Convert numpy obs (C,H,W) to tensor batch (1,C,H,W)."""
    return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


def main(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Choose device
    if torch.cuda.is_available() and args.device == 'cuda':
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Prepare result dirs
    result_dir = os.path.join('results', f"{args.env}_{args.id}")
    model_dir = os.path.join(result_dir, 'models')
    video_dir = os.path.join(result_dir, 'videos')
    image_dir = os.path.join(result_dir, 'images')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Load environment
    env = GymMiniGrid(args.env, tile_size=11, resize_shape=(84, 84))
    discrete_n = env.action_space.n  # before wrappers
    env.action_space = gym_old.spaces.Discrete(discrete_n)
    env = OneHotAction(env)
    env = RewardShapingWrapper(env, args.env)

    # Build config
    config = MiniGridConfig(
        env=args.env,
        obs_shape=env.observation_space.shape,
        action_size=discrete_n,
        model_dir=model_dir
    )

    # Instantiate agent and load model
    trainer = Trainer(config, device)
    ckpt = args.model_path or os.path.join(model_dir, 'models_best.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    trainer.load_model(ckpt)
    trainer.eval()

    # Evaluate episodes and manually record overlays
    for epi in range(args.eval_episodes):
        obs = env.reset()
        done = False
        prev_state = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, discrete_n, device=device)
        frames = []
        saved_imgs = []

        while not done:
            # Agent inference
            tensor_obs = to_tensor(obs, device)
            embed = trainer.ObsEncoder(tensor_obs)
            _, post_state = trainer.RSSM.rssm_observe(embed, prev_action, True, prev_state)
            model_state = trainer.RSSM.get_model_state(post_state)
            action_dist = trainer.ActionModel(model_state)
            action = trainer.ActionModel.add_exploration(action_dist[0], trainer.config.train_steps).detach()
            a_np = action.squeeze(0).cpu().numpy()

            # Step
            result = env.step(a_np)
            if len(result) == 5:
                next_obs, rew, term, trunc, _ = result
                done = term or trunc
            else:
                next_obs, rew, done, _ = result

            # SlotAttention overlay (last slot)
            attn_flat = trainer.ObsEncoder.slot_attn.last_attn
            Hp, Wp = trainer.ObsEncoder.slot_attn.last_Hp_Wp
            masks = attn_flat.view(1, attn_flat.shape[1], Hp, Wp)[0]
            mask = masks[-1].detach().cpu().numpy()
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_up = F.interpolate(mask_tensor, size=(obs.shape[1], obs.shape[2]), mode='bilinear', align_corners=False)[0,0].numpy()
            frame = obs.transpose(1,2,0)
            color = np.zeros_like(frame, dtype=np.uint8)
            color[...,0] = (mask_up * 255).astype(np.uint8)
            overlay = (0.6 * frame + 0.4 * color).astype(np.uint8)

            # Record
            if args.save_video:
                frames.append(overlay)
            if args.save_images and len(saved_imgs) < args.num_images and np.random.rand() < 0.05:
                saved_imgs.append(overlay)

            obs = next_obs
            prev_state = post_state
            prev_action = action

        # Save video at slower playback speed
        if args.save_video:
            vid_path = os.path.join(video_dir, f"{args.env}_{args.id}_ep{epi}.mp4")
            imageio.mimwrite(vid_path, frames, fps=args.video_fps)
            print(f"Saved video (fps={args.video_fps}): {vid_path}")
        # Save images
        if args.save_images:
            for idx, img in enumerate(saved_imgs):
                img_path = os.path.join(image_dir, f"{args.env}_{args.id}_ep{epi}_img{idx}.png")
                imageio.imwrite(img_path, img)
            print(f"Saved {len(saved_imgs)} images to {image_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and record a DreamerV2 agent on MiniGrid')
    parser.add_argument('--env', type=str, required=True, help='MiniGrid environment name')
    parser.add_argument('--id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda', help='Run on CPU or CUDA')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Checkpoint path (default: results/<env>_<id>/models/models_best.pth)')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--save-video', action='store_true', default=True, help='Save episode videos')
    parser.add_argument('--no-video', dest='save_video', action='store_false', help='Disable video saving')
    parser.add_argument('--video-fps', type=int, default=5, help='Playback frames-per-second for saved video')
    parser.add_argument('--save-images', action='store_true', default=True, help='Save random images')
    parser.add_argument('--no-images', dest='save_images', action='store_false', help='Disable image saving')
    parser.add_argument('--num-images', type=int, default=5, help='Random images per episode to save')
    args = parser.parse_args()
    main(args)
