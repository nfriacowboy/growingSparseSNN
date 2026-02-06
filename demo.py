"""
Quick demo script to visualize the growing SNN in action.

Runs a training session and creates visualizations of:
- Network growth over time
- Firing rate distributions
- Learning curves
- Energy consumption
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.models.growing_snn import GrowingSparseSNN
from src.environments.grid_world import ForagingGrid
from src.training.trainer import AdaptiveTrainer


def plot_training_results(trainer, output_dir='results'):
    """Create visualization plots from training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    rewards = trainer.episode_rewards
    window = 50
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label='Smoothed (50 ep)', linewidth=2)
    ax.plot(rewards, alpha=0.3, label='Raw')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax = axes[0, 1]
    lengths = trainer.episode_lengths
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label='Smoothed (50 ep)', linewidth=2)
    ax.plot(lengths, alpha=0.3, label='Raw')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Network Size (would need to track this)
    ax = axes[1, 0]
    stats = trainer.model.get_stats()
    ax.bar(['Current', 'Max'], [stats['n_neurons'], stats['max_neurons']])
    ax.set_ylabel('Neuron Count')
    ax.set_title(f"Network Size (Growth: {stats['growth_events']}, Pruning: {stats['pruning_events']})")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Firing Rate Distribution
    ax = axes[1, 1]
    firing_rates = trainer.model.firing_history.cpu().numpy()
    ax.hist(firing_rates, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(firing_rates.mean(), color='red', linestyle='--', 
               label=f'Mean: {firing_rates.mean():.4f}')
    ax.set_xlabel('Firing Rate')
    ax.set_ylabel('Count')
    ax.set_title('Neuron Firing Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150)
    print(f"Saved plot to {output_dir / 'training_results.png'}")
    plt.close()


def main():
    print("🧠 GrowingSparseSNN Demo")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    snn = GrowingSparseSNN(
        input_features=450,  # 15×15 × 2
        init_hidden=64,
        max_hidden=512,
        device=device
    )
    print(f"\nInitial model: {snn.get_stats()}")
    
    # Create environment
    env = ForagingGrid(size=15, n_food=10, food_respawn=True, max_steps=500)
    print(f"Environment: {env.size}×{env.size} grid, {env.n_food} food items")
    
    # Create trainer
    trainer = AdaptiveTrainer(
        model=snn,
        env=env,
        lr=1e-3,
        device=device,
        metrics_port=8000
    )
    
    # Train
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)
    
    trainer.train(
        episodes=500,
        timesteps=10,
        grow_interval=50,
        prune_interval=30,
        grow_threshold=0.05,
        prune_threshold=0.005,
        log_interval=25,
        eval_interval=100,
    )
    
    # Final stats
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    final_stats = snn.get_stats()
    print(f"Final neurons: {final_stats['n_neurons']} / {final_stats['max_neurons']}")
    print(f"Growth events: {final_stats['growth_events']}")
    print(f"Pruning events: {final_stats['pruning_events']}")
    print(f"Neurons added: {final_stats['neurons_added']}")
    print(f"Neurons removed: {final_stats['neurons_removed']}")
    print(f"Average firing rate: {final_stats['avg_firing_rate']:.4f}")
    print(f"Sparsity: {final_stats['sparsity']*100:.1f}%")
    print(f"Total parameters: {final_stats['total_params']:,}")
    
    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluating...")
    print("=" * 50)
    eval_stats = trainer.evaluate(n_episodes=10, timesteps=10)
    print(f"Average reward: {eval_stats['avg_reward']:.2f}")
    print(f"Average episode length: {eval_stats['avg_length']:.1f}")
    print(f"Average food collected: {eval_stats['avg_food']:.1f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_results(trainer)
    
    # Save model
    print("\nSaving checkpoint...")
    trainer.save_checkpoint('checkpoints/demo_model.pt')
    
    print("\n✅ Demo complete!")
    print("View metrics at: http://localhost:8000/metrics")


if __name__ == '__main__':
    main()
