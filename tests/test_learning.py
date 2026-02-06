"""Test basic learning and training loop."""

import pytest
import torch
from src.models.growing_snn import GrowingSparseSNN
from src.environments.grid_world import ForagingGrid
from src.training.trainer import AdaptiveTrainer


def test_basic_training_episode():
    """Test that one training episode completes successfully."""
    device = 'cpu'
    
    snn = GrowingSparseSNN(
        input_features=200,  # 10×10 × 2
        init_hidden=32,
        max_hidden=128,
        device=device
    )
    
    env = ForagingGrid(size=10, n_food=3, max_steps=50)
    
    trainer = AdaptiveTrainer(
        model=snn,
        env=env,
        lr=1e-3,
        device=device,
        metrics_port=8001  # Different port for testing
    )
    
    # Train one episode
    stats = trainer.train_episode(timesteps=5, render=False)
    
    # Check stats are returned
    assert 'episode_return' in stats
    assert 'episode_length' in stats
    assert 'loss' in stats
    assert stats['episode_length'] <= 50


def test_select_action():
    """Test action selection from SNN."""
    device = 'cpu'
    
    snn = GrowingSparseSNN(
        input_features=200,
        init_hidden=32,
        max_hidden=128,
        device=device
    )
    
    env = ForagingGrid(size=10, n_food=3)
    trainer = AdaptiveTrainer(model=snn, env=env, device=device, metrics_port=8002)
    
    # Get initial observation
    obs, _ = env.reset()
    
    # Select action
    action, log_prob, logits = trainer.select_action(obs, timesteps=5)
    
    # Check action is valid
    assert 0 <= action < 4
    assert isinstance(log_prob, torch.Tensor)
    assert logits.shape == (4,)


def test_training_updates_weights():
    """Test that training updates model weights."""
    device = 'cpu'
    
    snn = GrowingSparseSNN(
        input_features=200,
        init_hidden=32,
        max_hidden=128,
        device=device
    )
    
    env = ForagingGrid(size=10, n_food=2, max_steps=30)
    trainer = AdaptiveTrainer(model=snn, env=env, lr=1e-2, device=device, metrics_port=8003)
    
    # Capture initial weights
    initial_weights = snn.fc_in.weight.data.clone()
    
    # Train a few episodes
    for _ in range(5):
        trainer.train_episode(timesteps=5)
    
    # Check weights have changed
    final_weights = snn.fc_in.weight.data
    
    # At least some weights should have changed
    diff = (initial_weights - final_weights).abs().sum().item()
    assert diff > 1e-6, "Weights did not update during training"


def test_evaluation():
    """Test evaluation mode."""
    device = 'cpu'
    
    snn = GrowingSparseSNN(
        input_features=200,
        init_hidden=32,
        max_hidden=128,
        device=device
    )
    
    env = ForagingGrid(size=10, n_food=2, max_steps=30)
    trainer = AdaptiveTrainer(model=snn, env=env, device=device, metrics_port=8004)
    
    # Evaluate
    eval_stats = trainer.evaluate(n_episodes=3, timesteps=5)
    
    # Check eval stats
    assert 'avg_reward' in eval_stats
    assert 'avg_length' in eval_stats
    assert 'avg_food' in eval_stats


def test_checkpoint_save_load():
    """Test saving and loading checkpoints."""
    import tempfile
    import os
    
    device = 'cpu'
    
    snn = GrowingSparseSNN(
        input_features=200,
        init_hidden=32,
        max_hidden=128,
        device=device
    )
    
    env = ForagingGrid(size=10, n_food=2, max_steps=30)
    trainer = AdaptiveTrainer(model=snn, env=env, device=device, metrics_port=8005)
    
    # Train a bit
    for _ in range(3):
        trainer.train_episode(timesteps=5)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
    
    try:
        trainer.save_checkpoint(checkpoint_path)
        
        # Create new trainer and load
        snn2 = GrowingSparseSNN(
            input_features=200,
            init_hidden=32,
            max_hidden=128,
            device=device
        )
        trainer2 = AdaptiveTrainer(model=snn2, env=env, device=device, metrics_port=8006)
        trainer2.load_checkpoint(checkpoint_path)
        
        # Check weights match
        torch.testing.assert_close(
            snn.fc_in.weight.data,
            snn2.fc_in.weight.data
        )
        
    finally:
        os.unlink(checkpoint_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
