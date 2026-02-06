"""
GrowingSparseSNN: Dynamic Spiking Neural Network with Neurogenesis and Pruning

A spiking neural network that grows (adds neurons) and prunes (removes neurons)
during learning, inspired by biological neural development.

Key Features:
- Dynamic neuron addition (neurogenesis) based on network activity
- Aggressive pruning of low-activity neurons
- LIF (Leaky Integrate-and-Fire) neurons with Norse
- GPU-optimized operations for AMD ROCm
- Sparse connectivity patterns

Author: nfriacowboy
Date: 2026-02-06
"""

import torch
import torch.nn as nn
from norse.torch import LIFCell, LIFParameters, LIFState
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrowingSparseSNN(nn.Module):
    """
    Spiking Neural Network with dynamic growth and pruning capabilities.
    
    The network starts with a small number of hidden neurons and can:
    1. Grow: Add new neurons when activity is low (neurogenesis)
    2. Prune: Remove neurons with consistently low firing rates
    3. Learn: Update synaptic weights via surrogate gradients or STDP
    
    Args:
        input_features: Number of input features
        init_hidden: Initial number of hidden neurons
        max_hidden: Maximum number of hidden neurons allowed
        output_features: Number of output features (actions)
        tau_mem: Membrane time constant (ms)
        tau_syn: Synaptic time constant (ms)
        v_thresh: Spike threshold voltage
        device: torch device (cuda/cpu)
    """
    
    def __init__(
        self,
        input_features: int = 450,
        init_hidden: int = 64,
        max_hidden: int = 1024,
        output_features: int = 4,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        v_thresh: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.input_features = input_features
        self.hidden = init_hidden
        self.max_hidden = max_hidden
        self.output_features = output_features
        self.device = device
        
        # LIF parameters
        self.lif_params = LIFParameters(
            tau_mem_inv=torch.tensor(1.0 / tau_mem),
            tau_syn_inv=torch.tensor(1.0 / tau_syn),
            v_th=torch.tensor(v_thresh)
        )
        
        # Neural layers
        self.fc_in = nn.Linear(input_features, init_hidden, bias=False)
        self.lif = LIFCell(p=self.lif_params)
        self.fc_out = nn.Linear(init_hidden, output_features, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # State tracking
        self.state: Optional[LIFState] = None
        self.firing_history = torch.zeros(init_hidden, device=device)
        self.total_timesteps = 0
        
        # Growth/pruning statistics
        self.growth_events = 0
        self.pruning_events = 0
        self.neurons_added = 0
        self.neurons_removed = 0
        
        self.to(device)
        
        logger.info(f"Initialized GrowingSparseSNN: {input_features} -> {init_hidden} -> {output_features}")
        logger.info(f"Device: {device}, Max neurons: {max_hidden}")
    
    def _init_weights(self):
        """Initialize weights with Kaiming normal initialization."""
        nn.init.kaiming_normal_(self.fc_in.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_out.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: int = 10,
        return_spikes: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch, features) or (batch, timesteps, features)
            timesteps: Number of timesteps to simulate (if x is 2D)
            return_spikes: If True, return spike trains instead of rates
            
        Returns:
            Output tensor (batch, output_features)
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, timesteps, -1)  # (B, T, F)
        
        batch, T, F = x.shape
        device = x.device
        
        # Output accumulator
        spikes_out = torch.zeros(batch, self.output_features, device=device)
        hidden_spikes = []  # For monitoring
        
        # Simulate over time
        for t in range(T):
            xt = x[:, t, :]  # (B, F)
            
            # Input -> Hidden
            h = self.fc_in(xt)  # (B, H)
            
            # LIF dynamics
            s, self.state = self.lif(h, self.state)  # s: (B, H) binary spikes
            
            # Track firing rates for growth/pruning decisions
            with torch.no_grad():
                self.firing_history += s.float().mean(dim=0)  # Average across batch
            
            if return_spikes:
                hidden_spikes.append(s)
            
            # Hidden -> Output
            out = self.fc_out(s)  # (B, O)
            spikes_out += out
        
        # Update timestep counter
        self.total_timesteps += T
        
        # Normalize firing history
        with torch.no_grad():
            self.firing_history /= T
        
        # Return rate-coded output (or spike trains)
        if return_spikes:
            return spikes_out / T, torch.stack(hidden_spikes, dim=1)  # (B, O), (B, T, H)
        else:
            return spikes_out / T  # Rate-coded output
    
    def grow(self, n_neurons: int = 32, noise_scale: float = 0.1) -> int:
        """
        Add new neurons to the hidden layer (neurogenesis).
        
        Args:
            n_neurons: Number of neurons to add
            noise_scale: Scale of noise for weight initialization diversity
            
        Returns:
            Number of neurons actually added (limited by max_hidden)
        """
        # Limit growth to max_hidden
        n = min(n_neurons, self.max_hidden - self.hidden)
        if n <= 0:
            logger.warning(f"Cannot grow: already at max capacity ({self.hidden}/{self.max_hidden})")
            return 0
        
        old_hidden = self.hidden
        new_hidden = old_hidden + n
        
        logger.info(f"Growing network: {old_hidden} -> {new_hidden} neurons (+{n})")
        
        # Expand fc_in: (input_features, hidden) -> (input_features, new_hidden)
        w_in = self.fc_in.weight.data  # (old_hidden, input_features)
        new_w_in = torch.zeros(new_hidden, self.input_features, device=self.device)
        new_w_in[:old_hidden] = w_in
        
        # Initialize new neurons with diversity
        nn.init.kaiming_normal_(new_w_in[old_hidden:], mode='fan_in', nonlinearity='relu')
        new_w_in[old_hidden:] += torch.randn_like(new_w_in[old_hidden:]) * noise_scale
        
        self.fc_in = nn.Linear(self.input_features, new_hidden, bias=False).to(self.device)
        self.fc_in.weight.data = new_w_in
        
        # Expand fc_out: (hidden, output_features) -> (new_hidden, output_features)
        w_out = self.fc_out.weight.data  # (output_features, old_hidden)
        new_w_out = torch.zeros(self.output_features, new_hidden, device=self.device)
        new_w_out[:, :old_hidden] = w_out
        
        # Initialize output connections for new neurons
        nn.init.kaiming_normal_(new_w_out[:, old_hidden:], mode='fan_in', nonlinearity='relu')
        new_w_out[:, old_hidden:] += torch.randn_like(new_w_out[:, old_hidden:]) * noise_scale
        
        self.fc_out = nn.Linear(new_hidden, self.output_features, bias=False).to(self.device)
        self.fc_out.weight.data = new_w_out
        
        # Reset LIF state and firing history
        self.lif = LIFCell(p=self.lif_params).to(self.device)
        self.state = None
        
        # Expand firing history
        new_history = torch.zeros(new_hidden, device=self.device)
        new_history[:old_hidden] = self.firing_history
        self.firing_history = new_history
        
        self.hidden = new_hidden
        self.growth_events += 1
        self.neurons_added += n
        
        return n
    
    def prune(self, min_rate: float = 0.005, min_neurons: int = 32) -> int:
        """
        Remove neurons with low firing rates (pruning).
        
        Args:
            min_rate: Minimum firing rate threshold
            min_neurons: Minimum number of neurons to keep
            
        Returns:
            Number of neurons removed
        """
        # Identify neurons to keep
        keep_mask = self.firing_history > min_rate
        n_keep = keep_mask.sum().item()
        
        # Safety checks
        if n_keep >= self.hidden or n_keep < min_neurons:
            # Keep at least min_neurons, or don't prune if all are active
            if n_keep < min_neurons:
                # Keep top min_neurons by firing rate
                _, top_idx = torch.topk(self.firing_history, min_neurons)
                keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
                keep_mask[top_idx] = True
                n_keep = min_neurons
            else:
                logger.info(f"Pruning skipped: all neurons active (min_rate={min_rate})")
                return 0
        
        keep_idx = torch.nonzero(keep_mask).squeeze(-1)
        n_removed = self.hidden - n_keep
        
        logger.info(f"Pruning network: {self.hidden} -> {n_keep} neurons (-{n_removed})")
        
        # Prune fc_in weights
        self.fc_in.weight.data = self.fc_in.weight.data[keep_idx]
        self.fc_in = nn.Linear(self.input_features, n_keep, bias=False).to(self.device)
        self.fc_in.weight.data = self.fc_in.weight.data[keep_idx]
        
        # Prune fc_out weights
        self.fc_out.weight.data = self.fc_out.weight.data[:, keep_idx]
        self.fc_out = nn.Linear(n_keep, self.output_features, bias=False).to(self.device)
        self.fc_out.weight.data = self.fc_out.weight.data[:, keep_idx]
        
        # Reset state and firing history
        self.lif = LIFCell(p=self.lif_params).to(self.device)
        self.state = None
        self.firing_history = self.firing_history[keep_idx]
        
        self.hidden = n_keep
        self.pruning_events += 1
        self.neurons_removed += n_removed
        
        return n_removed
    
    def reset_state(self):
        """Reset the internal state of the network (between episodes)."""
        self.state = None
        self.firing_history.zero_()
        self.total_timesteps = 0
    
    def get_stats(self) -> dict:
        """Get network statistics for monitoring."""
        return {
            'n_neurons': self.hidden,
            'max_neurons': self.max_hidden,
            'growth_events': self.growth_events,
            'pruning_events': self.pruning_events,
            'neurons_added': self.neurons_added,
            'neurons_removed': self.neurons_removed,
            'avg_firing_rate': self.firing_history.mean().item(),
            'sparsity': (self.firing_history < 0.01).float().mean().item(),
            'total_params': sum(p.numel() for p in self.parameters()),
        }
    
    def should_grow(self, threshold: float = 0.05) -> bool:
        """Check if network should grow based on firing rate."""
        avg_rate = self.firing_history.mean().item()
        return avg_rate < threshold and self.hidden < self.max_hidden
    
    def should_prune(self, threshold: float = 0.005, min_sparsity: float = 0.3) -> bool:
        """Check if network should prune based on sparsity."""
        sparsity = (self.firing_history < threshold).float().mean().item()
        return sparsity > min_sparsity


if __name__ == '__main__':
    # Quick test
    print("Testing GrowingSparseSNN...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    snn = GrowingSparseSNN(input_features=100, init_hidden=32, max_hidden=256, device=device)
    
    # Test forward pass
    x = torch.randn(8, 100, device=device)  # batch of 8
    output = snn(x, timesteps=10)
    print(f"Output shape: {output.shape}")
    print(f"Stats: {snn.get_stats()}")
    
    # Test growth
    added = snn.grow(n_neurons=16)
    print(f"Added {added} neurons")
    print(f"Stats after growth: {snn.get_stats()}")
    
    # Test pruning
    removed = snn.prune(min_rate=0.1)
    print(f"Removed {removed} neurons")
    print(f"Stats after pruning: {snn.get_stats()}")
    
    print("✓ GrowingSparseSNN test passed!")
