# File: src/dreamerv2/envs/minigrid_wrapper.py

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
from gymnasium import spaces
from gymnasium import ObservationWrapper
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper
import inspect
import cv2


class SafeResizeObservation(ObservationWrapper):
    """
    cv2-based H×W resize that lives at module-top level ⇒ picklable.
    """
    def __init__(self, env, shape: tuple[int, int]):
        super().__init__(env)
        self.shape = shape            # (H, W)
        h, w = shape
        c = env.observation_space.shape[-1]
        self.observation_space = spaces.Box(
            low   = 0,
            high  = 255,
            shape = (h, w, c),
            dtype = env.observation_space.dtype,
        )

    def observation(self, obs):
        # cv2 expects (W, H) when resizing colour images
        obs = cv2.resize(obs, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return obs


class GymMiniGrid(ObservationWrapper):
    """
    Composite wrapper for MiniGrid environments that:
    1) Adds partial egocentric RGB image (H×W×3 uint8)
    2) Extracts image as sole observation
    3) Resizes to `resize_shape`
    4) Transposes from H×W×C → C×H×W
    5) Normalizes pixels to [0,1] float32
    6) Converts Gymnasium API to classic Gym:
       - step(action) → (obs, reward, done, info)
       - reset(... ) → obs
    """
    def __init__(
        self,
        env_name: str,
        tile_size: int = 11,
        resize_shape: tuple[int, int] = (84, 84),
    ):
        # 1) Base MiniGrid env (Gymnasium API)
        env = gym.make(env_name, render_mode='rgb_array')

        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        # 4) Resize to target H×W
        env = SafeResizeObservation(env, shape=resize_shape) 
        # Initialize ObservationWrapper
        super().__init__(env)

        # Detect if reset accepts seed
        self._reset_accepts_seed = 'seed' in inspect.signature(self.env.reset).parameters

        # 5) Normalize obs: define space as float32 [0.0, 1.0]
        h, w, c = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c, h, w),
            dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Transpose H×W×C → C×H×W and normalize to [0,1]
        obs = obs.transpose(2, 0, 1)
        return obs.astype(np.float32) / 255.0

    def step(self, action):
            # Gymnasium step → (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self.observation(obs)                         # transpose & normalize
            return obs, reward, terminated, truncated, info     # forward full signature

    def reset(self, *args, **kwargs):
        """
        Gymnasium-style reset:

        Returns (obs, info) even though Dreamer ultimately ignores `info`.
        This keeps AsyncVectorEnv happy while remaining backward-compatible.
        """
        # Unpack explicit kwargs that Gymnasium may forward
        seed    = kwargs.pop("seed", None)
        options = kwargs.pop("options", None)

        # Call the underlying env with or without seed
        try:
            if seed is not None and self._reset_accepts_seed:
                result = self.env.reset(seed=seed, options=options)
            else:
                result = self.env.reset(options=options) if options is not None else self.env.reset()
        except TypeError:
            result = self.env.reset()

        # Normalise return to (obs, info) tuple
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        return self.observation(obs), info   # ← TWO values now

    def seed(self, seed: int | None = None):
        """
        Provide the classic `env.seed()` API that Dreamer’s utils expect.

        ▸ We forward the call to `.reset(seed=…)` if the wrapped
          MiniGrid env supports it.

        ▸ Returning `[seed]` keeps parity with old Gym signatures,
          so any code that unpacks the return value won’t break.
        """
        if seed is not None and self._reset_accepts_seed:
            # Under the hood this sets RNGs for env, action_space & obs_space
            self.env.reset(seed=seed)
        return [seed]   # classic-Gym convention