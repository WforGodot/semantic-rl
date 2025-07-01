# File: MiniGrid_Empty_5x5_v0.py
import numpy as np

class RewardShaper:
    """Reward shaping for MiniGrid-Empty-5x5-v0"""
    def __init__(self):
        # Previous observation and step counter
        self.prev_obs = None
        self.episode_step_count = 0
        # Fixed settings for 5x5 env
        self.max_steps = 100
        self.optimal_steps = 5

    def compute_exploration_bonus(self, prev_obs, next_obs):
        """Small bonus when the visual observation changes."""
        img0 = prev_obs if isinstance(prev_obs, np.ndarray) else prev_obs.get('image', None)
        img1 = next_obs if isinstance(next_obs, np.ndarray) else next_obs.get('image', None)
        if isinstance(img0, np.ndarray) and isinstance(img1, np.ndarray):
            diff = np.abs(img0.astype(float) - img1.astype(float)).sum()
            return min(diff / 25000.0, 0.002)
        return 0.0

    def compute_speed_bonus(self, steps_taken):
        """Reward bonus for reaching goal quickly."""
        if steps_taken <= self.optimal_steps:
            ratio = steps_taken / self.optimal_steps
            return 100.0 + (1.0 - ratio) * 50.0
        ratio = min(steps_taken / self.max_steps, 1.0)
        optimal_ratio = self.optimal_steps / self.max_steps
        if ratio <= optimal_ratio:
            return 100.0
        normalized = (ratio - optimal_ratio) / (1.0 - optimal_ratio)
        return 100.0 * (1.0 - normalized) ** 2

    def shape_reward(self, next_obs, reward, done, info):
        """Apply shaping to the scalar reward. Matches base wrapper signature."""
        # Base penalty to encourage speed
        shaped = reward - 0.005
        # Exploration bonus
        if self.prev_obs is not None:
            shaped += self.compute_exploration_bonus(self.prev_obs, next_obs)
        # Update step count and prev_obs
        self.episode_step_count += 1
        self.prev_obs = next_obs

        # Goal reached bonus
        if reward > 0:
            speed = self.compute_speed_bonus(self.episode_step_count)
            shaped += 100.0 + speed
            print(f"GOAL in {self.episode_step_count} steps! Speed bonus {speed:.1f}, total {shaped:.1f}")

        # Reset on episode end
        if done:
            self.episode_step_count = 0
            self.prev_obs = None

        return shaped
