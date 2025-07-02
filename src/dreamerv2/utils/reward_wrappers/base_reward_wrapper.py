# Directory: dreamerv2/utils/reward_wrappers
# =========================================
# This folder will contain per-env reward shaping modules and a base wrapper.


# File: base_reward_wrapper.py
# ----------------------------
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
        # Convert env name to a valid module name: replace '-' with '_'
        module_name = env_name.replace('-', '_')
        try:
            reward_module = importlib.import_module(
                f"dreamerv2.utils.reward_wrappers.{module_name}"
            )
            self.shaper = reward_module.RewardShaper()
        except ModuleNotFoundError:
            # No shaping file for this env; leave rewards unchanged
            print(f"No reward shaping module found for {env_name}. Using default rewards.")
            self.shaper = None

    def step(self, action):
        """
        Accept either 4- or 5-tuple from the wrapped env and *always*
        return the Gymnasium 5-tuple upward so AsyncVectorEnv can batch.
        Also inject `env` into the info dict for RewardShaper.
        """
        result = self.env.step(action)

        # Unpack whichever length we got
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(np.logical_or(terminated, truncated).any())
        else:  # legacy 4-tuple
            obs, reward, done, info = result
            terminated, truncated = bool(done), False

        # Reward shaping
        if self.shaper:
            info = info or {}
            info["env"] = self.env          # ← needed by RewardShaper
            reward = self.shaper.shape_reward(obs, reward, done, info)

        # Return full 5-tuple for Gymnasium vector envs
        return obs, reward, terminated, truncated, info
