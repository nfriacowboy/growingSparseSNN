"""
Adaptive Trainer for Growing Sparse SNN

Trains the SNN with dynamic growth and pruning, using simple RL (REINFORCE).
Monitors performance and triggers neurogenesis/pruning based on heuristics.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Optional, List
import logging
from tqdm import tqdm

from ..models.growing_snn import GrowingSparseSNN
from ..environments.grid_world import ForagingGrid
from ..monitoring.metrics import get_metrics

logger = logging.getLogger(__name__)


class AdaptiveTrainer:
    """
    Trainer for GrowingSparseSNN with growth/pruning and REINFORCE learning.
    
    Features:
    - REINFORCE policy gradient (with baseline)
    - Automatic neurogenesis when performance plateaus or activity is low
    - Automatic pruning when sparsity is high
    - Prometheus metrics export
    """
    
    def __init__(
        self,
        model: GrowingSparseSNN,
        env: ForagingGrid,
        lr: float = 1e-3,
        gamma: float = 0.99,
        baseline_momentum: float = 0.9,
        device: str = None,
        metrics_port: int = 8000
    ):
        self.model = model
        self.env = env
        self.device = device or model.device
        self.gamma = gamma
        self.baseline_momentum = baseline_momentum
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Baseline for variance reduction
        self.baseline = 0.0
        
        # Metrics
        self.metrics = get_metrics(port=metrics_port)
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
        logger.info(f"AdaptiveTrainer initialized with lr={lr}, gamma={gamma}")
    
    def select_action(self, state: np.ndarray, timesteps: int = 10) -> tuple:
        """
        Select action using the SNN policy.
        
        Returns:
            action, log_prob, value (for policy gradient)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Forward pass through SNN
        action_logits = self.model(state_tensor, timesteps=timesteps)  # (1, n_actions)
        
        # Sample action from categorical distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, action_logits.squeeze(0)
    
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns (variance reduction)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def train_episode(self, timesteps: int = 10, render: bool = False) -> dict:
        """
        Train for one episode using REINFORCE.
        
        Returns:
            Episode statistics dictionary
        """
        state, info = self.env.reset()
        
        # Episode storage
        log_probs = []
        rewards = []
        done = False
        
        self.model.reset_state()
        
        steps = 0
        while not done:
            # Select action
            action, log_prob, _ = self.select_action(state, timesteps=timesteps)
            
            # Environment step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            steps += 1
            
            if render:
                self.env.render()
        
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Update baseline (moving average)
        episode_return = sum(rewards)
        self.baseline = self.baseline_momentum * self.baseline + \
                       (1 - self.baseline_momentum) * episode_return
        
        # Compute policy loss (REINFORCE with baseline)
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            advantage = R - self.baseline
            policy_loss.append(-log_prob * advantage)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Statistics
        stats = {
            'episode_return': episode_return,
            'episode_length': steps,
            'food_collected': info.get('collected_food', 0),
            'loss': policy_loss.item(),
            'baseline': self.baseline,
        }
        
        return stats
    
    def train(
        self,
        episodes: int = 1000,
        timesteps: int = 10,
        grow_interval: int = 100,
        prune_interval: int = 50,
        grow_threshold: float = 0.05,
        prune_threshold: float = 0.005,
        log_interval: int = 10,
        eval_interval: int = 50,
        render_eval: bool = False
    ):
        """
        Main training loop with adaptive growth and pruning.
        
        Args:
            episodes: Number of episodes to train
            timesteps: SNN simulation timesteps per step
            grow_interval: Check for growth every N episodes
            prune_interval: Check for pruning every N episodes
            grow_threshold: Firing rate threshold for growth
            prune_threshold: Firing rate threshold for pruning
            log_interval: Log statistics every N episodes
            eval_interval: Evaluate (without learning) every N episodes
            render_eval: Render during evaluation
        """
        logger.info(f"Starting training for {episodes} episodes")
        
        # Start metrics server
        try:
            self.metrics.start_server()
        except Exception as e:
            logger.warning(f"Could not start metrics server: {e}")
        
        for episode in tqdm(range(episodes), desc="Training"):
            # Train one episode
            stats = self.train_episode(timesteps=timesteps, render=False)
            
            # Store statistics
            self.episode_rewards.append(stats['episode_return'])
            self.episode_lengths.append(stats['episode_length'])
            
            # Update metrics
            self.metrics.record_episode(
                reward=stats['episode_return'],
                length=stats['episode_length'],
                food=stats['food_collected']
            )
            self.metrics.record_loss(stats['loss'])
            
            snn_stats = self.model.get_stats()
            self.metrics.update_from_snn(snn_stats)
            
            # Check for growth
            if episode > 0 and episode % grow_interval == 0:
                if self.model.should_grow(threshold=grow_threshold):
                    n_added = self.model.grow(n_neurons=32)
                    self.metrics.record_growth(n_added)
                    logger.info(f"Episode {episode}: Grew network by {n_added} neurons")
                    
                    # Reinitialize optimizer with new parameters
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'])
            
            # Check for pruning
            if episode > 0 and episode % prune_interval == 0:
                if self.model.should_prune(threshold=prune_threshold):
                    n_removed = self.model.prune(min_rate=prune_threshold)
                    self.metrics.record_pruning(n_removed)
                    logger.info(f"Episode {episode}: Pruned {n_removed} neurons")
                    
                    # Reinitialize optimizer
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'])
            
            # Logging
            if episode % log_interval == 0:
                recent_rewards = self.episode_rewards[-log_interval:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                
                logger.info(
                    f"Episode {episode}/{episodes} | "
                    f"Reward: {stats['episode_return']:.2f} (avg: {avg_reward:.2f}) | "
                    f"Length: {stats['episode_length']} | "
                    f"Neurons: {snn_stats['n_neurons']} | "
                    f"Loss: {stats['loss']:.4f}"
                )
            
            # Evaluation (no learning, just assess performance)
            if episode > 0 and episode % eval_interval == 0:
                eval_stats = self.evaluate(n_episodes=5, timesteps=timesteps, render=render_eval)
                logger.info(
                    f"Evaluation @ {episode}: "
                    f"Avg Reward: {eval_stats['avg_reward']:.2f} | "
                    f"Avg Length: {eval_stats['avg_length']:.1f} | "
                    f"Avg Food: {eval_stats['avg_food']:.1f}"
                )
        
        logger.info("Training complete!")
        self.save_checkpoint('final_model.pt')
    
    def evaluate(self, n_episodes: int = 10, timesteps: int = 10, render: bool = False) -> dict:
        """Evaluate the model without learning."""
        eval_rewards = []
        eval_lengths = []
        eval_food = []
        
        for _ in range(n_episodes):
            state, info = self.env.reset()
            self.model.reset_state()
            
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Select action (deterministic: take argmax)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_logits = self.model(state_tensor, timesteps=timesteps)
                action = action_logits.argmax().item()
                
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                if render:
                    self.env.render()
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)
            eval_food.append(info['collected_food'])
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'avg_food': np.mean(eval_food),
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'baseline': self.baseline,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'model_stats': self.model.get_stats(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.baseline = checkpoint['baseline']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        logger.info(f"Checkpoint loaded from {path}")


if __name__ == '__main__':
    # Quick test
    print("Testing AdaptiveTrainer...")
    
    from ..models.growing_snn import GrowingSparseSNN
    from ..environments.grid_world import ForagingGrid
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    snn = GrowingSparseSNN(input_features=200, init_hidden=32, max_hidden=256, device=device)
    env = ForagingGrid(size=10, n_food=5, food_respawn=True)
    
    trainer = AdaptiveTrainer(snn, env, lr=1e-3, metrics_port=8001)
    
    # Train for a few episodes
    trainer.train(episodes=20, grow_interval=10, prune_interval=15, log_interval=5)
    
    print("✓ AdaptiveTrainer test passed!")
