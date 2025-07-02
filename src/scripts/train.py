# dreamerv2/scripts/train.py
import wandb
import argparse
import os
import torch
import numpy as np
import gymnasium as gym           # modern Gymnasium API for envs
import gym as gym_old             # classic Gym for Discrete action-space conversion

from gymnasium.vector import AsyncVectorEnv, VectorWrapper
from dreamerv2.utils.wrapper import OneHotAction


from dreamerv2.training.config import MiniGridConfig
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator
import time

from dreamerv2.utils.minigrid_wrapper import GymMiniGrid

# After adding reward shaping:
from dreamerv2.utils.reward_wrappers import RewardShapingWrapper

import torch.nn.functional as F

# Pick fastest CuDNN kernels for your conv shapes
torch.backends.cudnn.benchmark = True

# On Ampere+ GPUs, allow TensorFloat-32 for matmuls/convs
torch.backends.cuda.matmul.allow_tf32   = True
torch.backends.cudnn.allow_tf32         = True

# —— helper to avoid repeated .to(device) calls ——
def to_tensor(x, device):
    # x: numpy array obs (C,H,W) or (seq,C,H,W) etc.
    return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

def main(args):
    # Login to W&B only if not disabled
    if not args.no_wandb:
        wandb.login()
    env_name = args.env
    exp_id = args.id

    # Directories
    result_dir = os.path.join('results', f'{env_name}_{exp_id}')
    model_dir = os.path.join(result_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Seeds & Device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    make_tensor = lambda x: to_tensor(x, device)

    def make_env(idx: int):
        def _init():
            env = GymMiniGrid(env_name, tile_size=11, resize_shape=(84, 84))

            # ↓↓↓ ADD THIS LINE ↓↓↓ ─────────────────────────────────────────
            env.action_space = gym_old.spaces.Discrete(env.action_space.n)
            # ────────────────────────────────────────────────────────────────

            env = OneHotAction(env)                    # now passes the assert

            # --- BEGIN FIX: convert action_space → Gymnasium Box -----------------
            n_actions = env.action_space.shape[0]  # size after one-hot
            env.action_space = gym.spaces.Box(     # Gymnasium Box so AsyncVectorEnv accepts it
                low=0.0, high=1.0, shape=(n_actions,), dtype=np.float32
            )
            # --- END FIX ---------------------------------------------------------
            env = RewardShapingWrapper(env, env_name)
            env.seed(args.seed + idx)
            return env
        return _init
    
    env_fns = [make_env(i) for i in range(args.num_envs)]

    env = AsyncVectorEnv(env_fns)       # parallel CPU workers

    # --- BEGIN FIX: Vector→Single-env adapter for Dreamer ---
    class VecToSingle(VectorWrapper):
        """
        Run a VectorEnv in parallel, but expose only the first sub-env
        as a classic (obs, reward, done, info) interface for Dreamer.
        """
        def reset(self, *args, **kwargs):
            # Gymnasium reset() → (obs_batch, info_batch)
            obs_b, _ = super().reset(*args, **kwargs)
            return obs_b[0]          # pick first env’s obs

        def step(self, actions):
            # Gymnasium step() → (obs_b, rew_b, term_b, trunc_b, info_b)
            obs_b, rew_b, term_b, trunc_b, info_b = super().step(actions)

            # Combine termination flags
            done_b = np.logical_or(term_b, trunc_b)

            # Select the first env’s data
            obs    = obs_b[0]
            reward = float(rew_b[0])                   # scalar
            done   = bool(done_b[0])                   # scalar
            info   = info_b[0] if isinstance(info_b, (list, tuple)) else info_b

            return obs, reward, done, info             # classic 4-tuple

    raw_c, raw_h, raw_w = env.single_observation_space.shape
    action_size          = env.single_action_space.shape[0]
    obs_shape            = (raw_c, raw_h, raw_w)

    obs_dtype = np.uint8
    action_dtype = np.float32


    # Config
    config = MiniGridConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dir=model_dir,
        train_steps=args.train_steps,
        save_every=max(1, args.train_steps // 10),  # periodic checkpoints
    )

    # Initialize trainer & evaluator
    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)

        # Training loop
    if not args.no_wandb:
        wandb_run = wandb.init(project='miniGrid-world-models', config=config.__dict__)
    else:
        wandb_run = None
        
    try:
        print('Starting training...')
        train_metrics = {}
        trainer.collect_seed_episodes(env)

        # reset environment
        obs_b, _ = env.reset()
        prev_rssmstate = trainer.RSSM._init_rssm_state(env.num_envs)    # ← now shape (N, …)
        prev_action    = torch.zeros(env.num_envs, trainer.action_size, device=device)  # ← (N, A)
        score = 0
        episode_actor_ent = []
        scores = []
        best_mean_score = -float('inf')
        best_save_path = os.path.join(model_dir, 'models_best.pth')

        last_time = time.time()
        for step in range(1, trainer.config.train_steps + 1):
            if step % 100 == 0:
                buf_size = trainer.buffer.idx if hasattr(trainer.buffer, "idx") else getattr(trainer.buffer, "size", "NA")
                print(f"Step {step}: {time.time() - last_time:.2f}s for last 100 steps")
                last_time = time.time()
            

            if step % 500 == 0:   # Log masks and overlays every 500 steps
                # 1) get the last raw frame (C,H,W) and convert to H×W×C
                frame = obs.transpose(1, 2, 0)  # uint8 in [0,255], shape (H, W, C)

                # 2) get & upsample masks (Bf, K, Hp*Wp) → (Bf, K, Hp, Wp) → (K, H, W)
                attn    = trainer.ObsEncoder.slot_attn.last_attn    # (Bf, K, Hp*Wp)
                Hp, Wp  = trainer.ObsEncoder.slot_attn.last_Hp_Wp
                masks_t = attn.view(-1, attn.shape[1], Hp, Wp)      # (Bf, K, Hp, Wp)
                masks_up = F.interpolate(
                    masks_t, size=(raw_h, raw_w),
                    mode='bilinear', align_corners=False
                )  # (Bf, K, H, W)
                masks_np = masks_up[0].detach().cpu().numpy()       # (K, H, W), first env in batch

                # 3) prepare raw-mask images
                raw_mask_images = [
                    wandb.Image((mask * 255).astype(np.uint8), caption=f"mask slot {k}")
                    for k, mask in enumerate(masks_np)
                ]

                # 4) prepare colored overlays
                overlay_images = []
                for k, mask in enumerate(masks_np):
                    # make a red-tint color mask
                    color = np.zeros_like(frame, dtype=np.uint8)
                    color[..., 0] = (mask * 255).astype(np.uint8)   # red channel

                    # alpha-blend: 60% frame + 40% mask
                    overlay = (0.6 * frame + 0.4 * color).astype(np.uint8)

                    overlay_images.append(
                        wandb.Image(overlay, caption=f"overlay slot {k}")
                    )

                # 5) log both galleries in one call
                if not args.no_wandb:
                    wandb.log({
                        "slot_masks": raw_mask_images,
                        "slot_overlays": overlay_images
                    }, step=step)

            # periodic model updates
            if step % trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            if step % trainer.config.slow_target_update == 0:
                trainer.update_target()
            if step % trainer.config.save_every == 0:
                #trainer.save_model(step)
                continue # skip saving during training for now

            # obs_b: (N, C, H, W); prev_actions_b: (N, A)
            prev_actions_b = prev_action  # keep shape (N, A) from last step
            with torch.no_grad():
                # 1) to tensor: (N, C, H, W) → (N, 1, C, H, W)
                t_obs = torch.as_tensor(obs_b, dtype=torch.float32, device=device).unsqueeze(1)
                # 2) encode & observe: handle batch in RSSM
                emb_b = trainer.ObsEncoder(t_obs.reshape(-1, *obs_shape))
                _, post_state_b = trainer.RSSM.rssm_observe(
                    emb_b, prev_actions_b, True, prev_rssmstate
                )
                model_state_b = trainer.RSSM.get_model_state(post_state_b)
                # 3) actor: returns (N, A)
                actions_b, action_dist = trainer.ActionModel(model_state_b)
                actions_b = trainer.ActionModel.add_exploration(actions_b, step).detach()
                episode_actor_ent.append(torch.mean(action_dist.entropy()).item())

            # Step ALL N envs at once
            obs_n_b, rew_b, term_b, trunc_b, info_b = env.step(actions_b.cpu().numpy())
            done_b = np.logical_or(term_b, trunc_b)
            score_b = rew_b  # vector of length N

            # Add *each* transition into the buffer
            for o, a, r, d in zip(obs_b, actions_b.cpu().numpy(), score_b, done_b):
                trainer.buffer.add(o, a, float(r), bool(d))

            # For logging & reset logic, we’ll track only env[0]:
            score = score_b[0]
            done = done_b[0]
            next_obs = obs_n_b[0]

            # Update for next step
            obs_b = obs_n_b
            prev_rssmstate = post_state_b
            prev_action    = actions_b

            # end of episode
            if done:
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] = np.mean(episode_actor_ent)
                if not args.no_wandb:
                    wandb.log(train_metrics, step=step)

                scores.append(score)
                print(f"Episode ended at step {step}, score: {score}, total episodes: {len(scores)}")
                
                if len(scores) > 100:
                    scores.pop(0)
                
                if len(scores) >= 10:  # Only evaluate best model after at least 10 episodes
                    avg_score = np.mean(scores)
                    print(f"Average score over last {len(scores)} episodes: {avg_score:.2f}, best so far: {best_mean_score:.2f}")
                    if avg_score > best_mean_score:
                        best_mean_score = avg_score
                        print(f"NEW BEST! Saving best model (avg score: {best_mean_score:.2f})")
                        torch.save(trainer.get_save_dict(), best_save_path)

                obs_b, _ = env.reset()
                obs = obs_b[0]
                score = 0
                prev_rssmstate = trainer.RSSM._init_rssm_state(env.num_envs)             # ← batched reset
                prev_action    = torch.zeros(env.num_envs, trainer.action_size, device=device)  # ← (N, A)
                episode_actor_ent = []
            else:
                obs = next_obs
                obs_b = obs_n_b
                prev_rssmstate = post_state_b
                prev_action = actions_b

        # final evaluation
        # Save final model as best if no best model was saved during training
        if not os.path.exists(best_save_path):
            print(f"No best model was saved during training. Saving final model as best.")
            torch.save(trainer.get_save_dict(), best_save_path)
        
        evaluator.eval_saved_agent(env, best_save_path)
        
    finally:
        # Close wandb run if it was initialized
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", help="MiniGrid env name")
    parser.add_argument("--id", type=str, default="0", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--device", choices=["cpu","cuda"], default="cuda", help="CUDA or CPU")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    parser.add_argument("--train_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument('--num_envs', type=int, default=1,
                    help='Number of parallel MiniGrid copies')
    parser.add_argument('--no_wandb', action='store_true',
                    help='Disable Weights & Biases logging')
    args = parser.parse_args()
    main(args)


# python scripts/train.py   --env MiniGrid-DoorKey-6x6-v0   --id overnight   --device cuda   --batch_size 8   --seq_len 16   --train_steps 1000 --no_wandb  --num_envs 2   