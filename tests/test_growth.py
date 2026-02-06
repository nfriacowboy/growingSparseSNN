"""Test growth mechanism of GrowingSparseSNN."""

import pytest
import torch
from src.models.growing_snn import GrowingSparseSNN


def test_growth_increases_neurons():
    """Test that growth increases neuron count."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=32,
        max_hidden=128,
        device='cpu'
    )
    
    initial_neurons = snn.hidden
    n_added = snn.grow(n_neurons=16)
    
    assert n_added == 16
    assert snn.hidden == initial_neurons + 16
    assert snn.hidden <= snn.max_hidden


def test_growth_preserves_old_weights():
    """Test that growth preserves existing neuron weights."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=32,
        max_hidden=128,
        device='cpu'
    )
    
    # Capture initial weights
    old_w_in = snn.fc_in.weight.data.clone()
    old_w_out = snn.fc_out.weight.data.clone()
    
    # Grow
    snn.grow(n_neurons=16)
    
    # Check old weights are preserved
    torch.testing.assert_close(
        snn.fc_in.weight.data[:32],  # First 32 neurons
        old_w_in,
        rtol=1e-5,
        atol=1e-5
    )
    
    torch.testing.assert_close(
        snn.fc_out.weight.data[:, :32],  # First 32 neurons
        old_w_out,
        rtol=1e-5,
        atol=1e-5
    )


def test_growth_respects_max_limit():
    """Test that growth respects max_hidden limit."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=100,
        max_hidden=110,
        device='cpu'
    )
    
    # Try to grow beyond limit
    n_added = snn.grow(n_neurons=20)
    
    # Should only add 10 neurons (up to max)
    assert n_added == 10
    assert snn.hidden == 110


def test_forward_after_growth():
    """Test that forward pass works after growth."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=32,
        max_hidden=128,
        device='cpu'
    )
    
    x = torch.randn(4, 100)  # Batch of 4
    
    # Forward before growth
    out1 = snn(x, timesteps=5)
    assert out1.shape == (4, 4)  # (batch, actions)
    
    # Grow
    snn.grow(n_neurons=16)
    snn.reset_state()
    
    # Forward after growth
    out2 = snn(x, timesteps=5)
    assert out2.shape == (4, 4)


def test_multiple_growths():
    """Test multiple sequential growth operations."""
    snn = GrowingSparseSNN(
        input_features=100,
        init_hidden=32,
        max_hidden=128,
        device='cpu'
    )
    
    # Multiple growths
    snn.grow(n_neurons=16)
    assert snn.hidden == 48
    
    snn.grow(n_neurons=16)
    assert snn.hidden == 64
    
    snn.grow(n_neurons=32)
    assert snn.hidden == 96
    
    # Statistics
    stats = snn.get_stats()
    assert stats['n_neurons'] == 96
    assert stats['growth_events'] == 3
    assert stats['neurons_added'] == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
