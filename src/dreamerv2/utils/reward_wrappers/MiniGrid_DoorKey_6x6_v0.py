# File: src/dreamerv2/utils/reward_wrappers/MiniGrid_DoorKey_6x6_v0.py
from minigrid.core.world_object import Door, Key
import math

class RewardShaper:
    """
    Advanced reward shaping for MiniGrid-DoorKey-6x6-v0.
    - Time penalty to incentivize speed.
    - Annealed novelty bonus for exploring new squares.
    - Proximity bonuses for moving closer to subgoals.
    - Tunable, larger rewards for picking up the key, opening the door, and completing the episode.
    """

    def __init__(self):
        # Episode tracking
        self.step_count = 0
        self.max_steps = 150  # Env timeout

        # Shaping hyperparameters
        self.step_penalty      = 0.005   # cost each step
        self.novelty_bonus     = 0.015   # initial exploration bonus
        self.key_base_reward   = 2.0     # reward on key pickup
        self.door_base_reward  = 3.0     # reward on door opening
        self.goal_base_reward  = 5.0     # reward on final success

        # Speed bonus parameters
        self.optimal_steps_key  = 10
        self.optimal_steps_door = 15
        self.optimal_steps_goal = 25

        # Subgoal flags and state
        self.key_picked  = False
        self.door_opened = False
        self.key_pos   = None
        self.door_pos  = None
        self.prev_dist_to_key  = float('inf')
        self.prev_dist_to_door = float('inf')

        # Exploration tracking
        self.visited_positions = set()

    def _get_speed_bonus(self, optimal_steps):
        t = self.step_count
        if t <= optimal_steps:
            return 1.0
        ratio = min(t / self.max_steps, 1.0)
        opt_ratio = optimal_steps / self.max_steps
        norm = max(0, (ratio - opt_ratio) / (1.0 - opt_ratio))
        return (1.0 - norm) ** 2

    def _reset_episode_state(self):
        self.step_count = 0
        self.key_picked = False
        self.door_opened = False
        self.key_pos = None
        self.door_pos = None
        self.prev_dist_to_key = float('inf')
        self.prev_dist_to_door = float('inf')
        self.visited_positions = set()

    def _find_objects(self, unwrapped_env):
        if self.key_pos and self.door_pos:
            return
        found_key = False
        found_door = False
        for y in range(unwrapped_env.height):
            for x in range(unwrapped_env.width):
                cell = unwrapped_env.grid.get(x, y)
                if isinstance(cell, Key):
                    self.key_pos = (x, y)
                    found_key = True
                elif isinstance(cell, Door):
                    self.door_pos = (x, y)
                    found_door = True
        if not found_key or not found_door:
            return

    def shape_reward(self, next_obs, base_reward, done, info):
        env = info.get("env")
        if env is None:
            raise RuntimeError("RewardShaper needs env in info dict")
        unwrapped_env = env.unwrapped

        # Time penalty
        shaped = base_reward - self.step_penalty
        self.step_count += 1
        agent_pos = unwrapped_env.agent_pos

        if self.step_count == 1:
            self._find_objects(unwrapped_env)

        # Annealed novelty bonus
        if agent_pos not in self.visited_positions:
            anneal = max(0.0, 1.0 - self.step_count/self.max_steps)
            shaped += self.novelty_bonus * anneal
        self.visited_positions.add(agent_pos)

        # Proximity & subgoal rewards
        if not self.key_picked:
            dist = math.hypot(agent_pos[0]-self.key_pos[0], agent_pos[1]-self.key_pos[1])
            if dist < self.prev_dist_to_key:
                shaped += 0.02
            self.prev_dist_to_key = dist
            if unwrapped_env.carrying and isinstance(unwrapped_env.carrying, Key):
                self.key_picked = True
                shaped += self.key_base_reward + self._get_speed_bonus(self.optimal_steps_key)
        elif not self.door_opened:
            dist = math.hypot(agent_pos[0]-self.door_pos[0], agent_pos[1]-self.door_pos[1])
            if dist < self.prev_dist_to_door:
                shaped += 0.02
            self.prev_dist_to_door = dist
            door_cell = unwrapped_env.grid.get(*self.door_pos)
            if isinstance(door_cell, Door) and door_cell.is_open:
                self.door_opened = True
                shaped += self.door_base_reward + self._get_speed_bonus(self.optimal_steps_door)

        # Final success reward
        if base_reward > 0:
            shaped += self.goal_base_reward + self._get_speed_bonus(self.optimal_steps_goal)

        if done:
            self._reset_episode_state()

        return shaped
