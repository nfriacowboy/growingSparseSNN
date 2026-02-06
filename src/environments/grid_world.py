"""
Foraging Grid Environment

A simple 2D grid world where an agent must forage for food while avoiding obstacles.
Designed to test continual learning and adaptation in growing SNNs.

Features:
- Dynamically spawning food
- Optional obstacles/barriers
- Sparse reward signal
- Observations as flattened grid + extra info (agent pos, food direction, etc.)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ForagingGrid(gym.Env):
    """
    2D grid foraging environment.
    
    The agent must navigate a grid to collect food items. Food spawns randomly
    and the agent receives a reward when reaching food. The episode terminates
    after collecting all food or reaching max steps.
    
    Observation:
        - Grid representation (2 channels: agent position, food locations)
        - Flattened to a vector: size × size × 2
        
    Actions:
        0: Move up
        1: Move down
        2: Move left
        3: Move right
    
    Rewards:
        +10: Collect food
        -0.01: Each step (encourage efficiency)
        -1: Hit wall (optional)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        size: int = 15,
        n_food: int = 10,
        max_steps: int = 500,
        food_respawn: bool = True,
        wall_penalty: bool = False,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.size = size
        self.n_food = n_food
        self.max_steps = max_steps
        self.food_respawn = food_respawn
        self.wall_penalty = wall_penalty
        self.render_mode = render_mode
        
        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: flattened 2-channel grid
        obs_shape = size * size * 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_shape,), dtype=np.float32
        )
        
        # State
        self.agent_pos = np.array([size // 2, size // 2])
        self.food_positions = []
        self.collected_food = 0
        self.steps = 0
        
        # Action mapping
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1]),   # Right
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset agent to center
        self.agent_pos = np.array([self.size // 2, self.size // 2])
        
        # Spawn food randomly
        self.food_positions = []
        for _ in range(self.n_food):
            self._spawn_food()
        
        self.collected_food = 0
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.steps += 1
        
        # Move agent
        direction = self._action_to_direction[action]
        new_pos = self.agent_pos + direction
        
        # Check boundaries
        reward = -0.01  # Small negative reward for each step
        hit_wall = False
        
        if self._is_valid_pos(new_pos):
            self.agent_pos = new_pos
        else:
            hit_wall = True
            if self.wall_penalty:
                reward -= 1.0
        
        # Check if food collected
        food_collected = False
        for i, food_pos in enumerate(self.food_positions):
            if np.array_equal(self.agent_pos, food_pos):
                reward += 10.0  # Food reward
                self.collected_food += 1
                food_collected = True
                
                # Remove food
                self.food_positions.pop(i)
                
                # Respawn if enabled
                if self.food_respawn:
                    self._spawn_food()
                
                break
        
        # Termination conditions
        terminated = len(self.food_positions) == 0  # All food collected
        truncated = self.steps >= self.max_steps  # Max steps reached
        
        observation = self._get_obs()
        info = self._get_info()
        info['hit_wall'] = hit_wall
        info['food_collected'] = food_collected
        
        return observation, reward, terminated, truncated, info
    
    def _is_valid_pos(self, pos: np.ndarray) -> bool:
        """Check if position is within grid bounds."""
        return (0 <= pos[0] < self.size) and (0 <= pos[1] < self.size)
    
    def _spawn_food(self):
        """Spawn a food item at random unoccupied position."""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = np.array([
                np.random.randint(0, self.size),
                np.random.randint(0, self.size)
            ])
            
            # Check if position is free (not agent, not other food)
            if not np.array_equal(pos, self.agent_pos) and \
               not any(np.array_equal(pos, f) for f in self.food_positions):
                self.food_positions.append(pos)
                return
    
    def _get_obs(self) -> np.ndarray:
        """
        Get observation as flattened 2-channel grid.
        Channel 0: Agent position (1 at agent location, 0 elsewhere)
        Channel 1: Food positions (1 at food locations, 0 elsewhere)
        """
        # Create 2-channel grid
        grid = np.zeros((2, self.size, self.size), dtype=np.float32)
        
        # Channel 0: Agent
        grid[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Channel 1: Food
        for food_pos in self.food_positions:
            grid[1, food_pos[0], food_pos[1]] = 1.0
        
        # Flatten and return
        return grid.flatten()
    
    def _get_info(self) -> dict:
        """Get auxiliary information."""
        # Calculate distance to nearest food
        if len(self.food_positions) > 0:
            distances = [np.linalg.norm(self.agent_pos - food) 
                        for food in self.food_positions]
            nearest_food_dist = min(distances)
        else:
            nearest_food_dist = 0.0
        
        return {
            'agent_pos': self.agent_pos.copy(),
            'n_food_remaining': len(self.food_positions),
            'collected_food': self.collected_food,
            'steps': self.steps,
            'nearest_food_dist': nearest_food_dist,
        }
    
    def render(self):
        """Render the environment (simple ASCII for now)."""
        if self.render_mode is None:
            return
        
        # Create grid representation
        grid = np.full((self.size, self.size), '.', dtype=str)
        
        # Place food
        for food_pos in self.food_positions:
            grid[food_pos[0], food_pos[1]] = 'F'
        
        # Place agent
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        
        # Print
        print('\n' + '='*self.size*2)
        for row in grid:
            print(' '.join(row))
        print(f"Steps: {self.steps}, Food: {len(self.food_positions)}, Collected: {self.collected_food}")
        print('='*self.size*2 + '\n')


if __name__ == '__main__':
    # Test environment
    print("Testing ForagingGrid environment...")
    
    env = ForagingGrid(size=10, n_food=5, food_respawn=False, render_mode='human')
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    env.render()
    
    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, info={info}")
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished! Terminated: {terminated}, Truncated: {truncated}")
            break
    
    print("✓ ForagingGrid test passed!")
