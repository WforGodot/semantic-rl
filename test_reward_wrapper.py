import numpy as np
import gym
from dreamerv2.utils.reward_wrappers.base_reward_wrapper import RewardShapingWrapper

# ——— 1) Dummy env that tracks resets ————————————————
class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # simple 1-D box obs & 2-action discrete space
        self.observation_space = gym.spaces.Box(0, 255, (4,), np.uint8)
        self.action_space      = gym.spaces.Discrete(2)
        self.reset_called      = False

    def reset(self):
        self.reset_called = True
        # return (obs, info) format
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        # always returns a valid transition
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, 1.0, False, False, {}

# ——— 2) Wrap it and force a shaping error ————————————
env = DummyEnv()
wrapper = RewardShapingWrapper(env, env_name="dummy_env")  # no real shaper module, but .shaper exists
# monkey-patch a shaper that always throws
class BoomShaper:
    def shape_reward(self, obs, reward, done, info):
        raise RuntimeError("💥 boom!")
wrapper.shaper = BoomShaper()

# ——— 3) Call step() and check that it catches + resets ————
obs, reward, terminated, truncated, info = wrapper.step(0)

assert env.reset_called, "❌ wrapper did NOT reset the env on exception!"
assert reward == 0.0,    "❌ expected fallback reward 0.0, got " + str(reward)
print("✅ Test passed: exception was caught and env.reset() was called.")
