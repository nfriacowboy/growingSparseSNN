"""Test pruning mechanism of GrowingSparseSNN."""

import pytest
import torch
from src.models.growing_snn import GrowingSparseSNN


def test_pruning_removes_low_activity_neurons():
    """Test that pruning removes neurons with low firing rates."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=64,
        max_hidden=128,
        device='cpu'
    )
    
    # Simulate activity: make half of neurons have low firing
    snn.firing_history[:32] = 0.001  # Low activity
    snn.firing_history[32:] = 0.1    # High activity
    
    initial_neurons = snn.hidden
    n_removed = snn.prune(min_rate=0.005)
    
    # Should remove ~32 low-activity neurons
    assert n_removed > 0
    assert snn.hidden < initial_neurons
    assert snn.hidden >= 32  # Keep at least min_neurons


def test_pruning_respects_min_neurons():
    """Test that pruning keeps at least min_neurons."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=64,
        max_hidden=128,
        device='cpu'
    )
    
    # Make all neurons low activity
    snn.firing_history[:] = 0.001
    
    # Prune with min_neurons=32
    n_removed = snn.prune(min_rate=0.005, min_neurons=32)
    
    # Should keep at least 32 neurons
    assert snn.hidden >= 32


def test_forward_after_pruning():
    """Test that forward pass works after pruning."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=64,
        max_hidden=128,
        device='cpu'
    )
    
    x = torch.randn(4, 100)
    
    # Forward before pruning
    out1 = snn(x, timesteps=5)
    assert out1.shape == (4, 4)
    
    # Simulate some activity
    snn.firing_history[:32] = 0.001
    snn.firing_history[32:] = 0.1
    
    # Prune
    snn.prune(min_rate=0.005)
    snn.reset_state()
    
    # Forward after pruning
    out2 = snn(x, timesteps=5)
    assert out2.shape == (4, 4)


def test_growth_then_pruning():
    """Test growth followed by pruning."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=32,
        max_hidden=128,
        device='cpu'
    )
    
    # Grow
    snn.grow(n_neurons=32)
    assert snn.hidden == 64
    
    # Simulate activity
    snn.firing_history[:20] = 0.001  # Low
    snn.firing_history[20:] = 0.1    # High
    
    # Prune
    n_removed = snn.prune(min_rate=0.005)
    assert n_removed > 0
    assert snn.hidden < 64
    
    # Check stats
    stats = snn.get_stats()
    assert stats['growth_events'] == 1
    assert stats['pruning_events'] == 1


def test_no_pruning_when_all_active():
    """Test that no pruning occurs when all neurons are active."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=64,
        max_hidden=128,
        device='cpu'
    )
    
    # Make all neurons highly active
    snn.firing_history[:] = 0.1
    
    initial_neurons = snn.hidden
    n_removed = snn.prune(min_rate=0.005)
    
    # Should not prune anything
    assert n_removed == 0
    assert snn.hidden == initial_neurons


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
