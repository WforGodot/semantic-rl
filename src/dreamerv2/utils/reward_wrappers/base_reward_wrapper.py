# File: base_reward_wrapper.py
import importlib
import gym
import numpy as np

class RewardShapingWrapper(gym.Wrapper):
    """
    Wraps an env and, if available, applies per-env reward shaping.
    Dynamically loads a module named after the env (underscores instead of hyphens).
    """
    def __init__(self, env, env_name: str):
        super().__init__(env)
        module_name = env_name.replace('-', '_')
        try:
            reward_module = importlib.import_module(
                f"dreamerv2.utils.reward_wrappers.{module_name}"
            )
            self.shaper = reward_module.RewardShaper()
        except ModuleNotFoundError:
            print(f"No reward shaping module found for {env_name}. Using default rewards.")
            self.shaper = None

    def step(self, action):
        try:
            # Original env step
            result = self.env.step(action)

            # Unpack result
            if result is None:
                raise TypeError("Received None result from env.step")
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(np.logical_or(terminated, truncated).any())
            else:
                obs, reward, done, info = result
                terminated, truncated = bool(done), False

            # Reward shaping
            if self.shaper:
                info = info or {}
                info['env'] = self.env
                reward = self.shaper.shape_reward(obs, reward, done, info)

        except Exception as e:
            # On error, reset this sub-env to allow training to continue
            print(f"[Error] RewardShapingWrapper caught '{e}'. Resetting environment.")
            obs, info = self.env.reset()
            reward = 0.0
            terminated = False
            truncated = False

        # Return for Gymnasium vector envs
        return obs, reward, terminated, truncated, info