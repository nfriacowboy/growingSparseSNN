"""Test environment functionality."""

import pytest
import numpy as np
from src.environments.grid_world import ForagingGrid


def test_environment_reset():
    """Test environment reset."""
    env = ForagingGrid(size=10, n_food=5)
    obs, info = env.reset(seed=42)
    
    # Check observation shape
    assert obs.shape == (10 * 10 * 2,)  # Flattened 2-channel grid
    
    # Check info
    assert 'agent_pos' in info
    assert 'n_food_remaining' in info
    assert info['n_food_remaining'] == 5


def test_environment_step():
    """Test environment step."""
    env = ForagingGrid(size=10, n_food=3, max_steps=50)
    obs, info = env.reset(seed=42)
    
    # Take action
    obs, reward, terminated, truncated, info = env.step(0)  # Move up
    
    # Check outputs
    assert obs.shape == (200,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_food_collection():
    """Test that food can be collected."""
    env = ForagingGrid(size=5, n_food=1, food_respawn=False, max_steps=100)
    env.reset(seed=42)
    
    # Manually place food next to agent for easy collection
    env.agent_pos = np.array([2, 2])
    env.food_positions = [np.array([2, 3])]
    
    initial_food = len(env.food_positions)
    
    # Move right to collect food
    obs, reward, terminated, truncated, info = env.step(3)
    
    # Check food was collected
    assert reward > 0  # Should get positive reward
    assert info['food_collected'] == True
    assert len(env.food_positions) < initial_food or env.food_respawn


def test_episode_termination():
    """Test that episode terminates correctly."""
    env = ForagingGrid(size=5, n_food=1, food_respawn=False, max_steps=10)
    env.reset(seed=42)
    
    done = False
    steps = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Ensure we don't loop forever
        assert steps <= 20, "Episode did not terminate"
    
    # Should terminate either by collecting food or hitting max steps
    assert terminated or truncated


def test_boundary_handling():
    """Test that agent can't move outside boundaries."""
    env = ForagingGrid(size=5, n_food=1)
    env.reset(seed=42)
    
    # Place agent at corner
    env.agent_pos = np.array([0, 0])
    initial_pos = env.agent_pos.copy()
    
    # Try to move up (should hit wall)
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Agent should stay in place
    assert np.array_equal(env.agent_pos, initial_pos)
    assert info['hit_wall'] == True


def test_observation_format():
    """Test observation format is correct."""
    env = ForagingGrid(size=10, n_food=3)
    obs, info = env.reset(seed=42)
    
    # Reshape to check channels
    grid = obs.reshape(2, 10, 10)
    
    # Channel 0 should have exactly 1 agent
    assert grid[0].sum() == 1.0
    
    # Channel 1 should have exactly 3 food items
    assert grid[1].sum() == 3.0
    
    # All values should be 0 or 1
    assert np.all((obs == 0) | (obs == 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
