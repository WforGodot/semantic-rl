# File: DoorKey_6x6_v0_final.py
from minigrid.core.world_object import Door, Key
import math

class RewardShaper:
    """
    Advanced reward shaping for MiniGrid-DoorKey-6x6-v0.
    - Adds speed bonuses for key pickup and door opening.
    - Adds a proximity bonus for moving closer to the next objective.
    - Adds a novelty bonus for visiting new grid squares.
    """

    def __init__(self):
        # --- Episode progress tracking ---
        self.step_count = 0
        self.max_steps = 150  # Default timeout in DoorKey-6x6

        # --- Optimal steps for speed bonuses ---
        self.optimal_steps_key = 10
        self.optimal_steps_door = 15 # Steps from the start
        self.optimal_steps_goal = 25 # Steps from the start

        # --- Flags for sub-goals ---
        self.key_picked = False
        self.door_opened = False
        
        # --- State for proximity bonus ---
        self.key_pos = None
        self.door_pos = None
        self.prev_dist_to_key = float('inf')
        self.prev_dist_to_door = float('inf')
        
        # --- State for exploration bonus ---
        self.visited_positions = set()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _get_speed_bonus(self, optimal_steps):
        """Quadratic bonus that peaks at or before `optimal_steps`."""
        t = self.step_count
        if t <= optimal_steps:
            return 1.0  # Max bonus for optimal performance
        ratio = min(t / self.max_steps, 1.0)
        opt_ratio = optimal_steps / self.max_steps
        norm = max(0, (ratio - opt_ratio) / (1.0 - opt_ratio))
        return (1.0 - norm) ** 2

    def _reset_episode_state(self):
        """Resets all flags and state variables at the end of an episode."""
        self.step_count = 0
        self.key_picked = False
        self.door_opened = False
        self.key_pos = None
        self.door_pos = None
        self.prev_dist_to_key = float('inf')
        self.prev_dist_to_door = float('inf')
        # Clear the set of visited positions for the new episode
        self.visited_positions = set()

    def _find_objects(self, unwrapped_env):
        """Finds and caches the positions of the key and door."""
        if self.key_pos and self.door_pos:
            return
        for y in range(unwrapped_env.height):
            for x in range(unwrapped_env.width):
                cell = unwrapped_env.grid.get(x, y)
                if isinstance(cell, Key):
                    self.key_pos = (x, y)
                elif isinstance(cell, Door):
                    self.door_pos = (x, y)
        
        if not self.key_pos or not self.door_pos:
            raise RuntimeError("Could not find Key or Door in the environment grid.")

    # ------------------------------------------------------------------
    # Main shaping function
    # ------------------------------------------------------------------

    def shape_reward(self, next_obs, base_reward, done, info):
        """Called from wrapper after env.step(...)"""
        env = info.get("env")
        if env is None:
            raise RuntimeError("RewardShaper needs env in info dict")

        unwrapped_env = env.unwrapped
        shaped = base_reward - 0.01  # Mild time-penalty each step
        self.step_count += 1

        if self.step_count == 1:
            self._find_objects(unwrapped_env)

        agent_pos = unwrapped_env.agent_pos

        # --- Novelty-based Exploration Bonus ---
        if agent_pos not in self.visited_positions:
            shaped += 0.015  # Small bonus for visiting a new square
        
        # Add current position to the set of visited locations for this episode
        self.visited_positions.add(agent_pos)

        # --- Proximity and Sub-goal Bonuses ---
        if not self.key_picked:
            dist_to_key = math.hypot(agent_pos[0] - self.key_pos[0], agent_pos[1] - self.key_pos[1])
            if dist_to_key < self.prev_dist_to_key:
                shaped += 0.02
            self.prev_dist_to_key = dist_to_key

            if unwrapped_env.carrying is not None and isinstance(unwrapped_env.carrying, Key):
                self.key_picked = True
                shaped += 0.5 + self._get_speed_bonus(self.optimal_steps_key)

        elif not self.door_opened:
            dist_to_door = math.hypot(agent_pos[0] - self.door_pos[0], agent_pos[1] - self.door_pos[1])
            if dist_to_door < self.prev_dist_to_door:
                shaped += 0.02
            self.prev_dist_to_door = dist_to_door
            
            door_cell = unwrapped_env.grid.get(*self.door_pos)
            if isinstance(door_cell, Door) and door_cell.is_open:
                self.door_opened = True
                shaped += 0.5 + self._get_speed_bonus(self.optimal_steps_door)

        # --- Final Goal Bonus ---
        if base_reward > 0:
            shaped += 1.0 + self._get_speed_bonus(self.optimal_steps_goal)

        # --- Episode Done ---
        if done:
            self._reset_episode_state()

        return shaped