# Directory: dreamerv2/utils/reward_wrappers
# =========================================
# This folder will contain per-env reward shaping modules and a base wrapper.


# File: base_reward_wrapper.py
# ----------------------------
import importlib
import gym

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
        obs, reward, done, info = self.env.step(action)
        if self.shaper:
            # pass env for direct state access
            info = info or {}
            info["env"] = self.env
            reward = self.shaper.shape_reward(obs, reward, done, info)
        return obs, reward, done, info

