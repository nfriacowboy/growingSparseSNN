"""
Main training script for GrowingSparseSNN

Usage:
    python src/training/train.py --config configs/training_config.yaml
    
Or with default settings:
    python src/training/train.py
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path

from src.models.growing_snn import GrowingSparseSNN
from src.environments.grid_world import ForagingGrid
from src.training.trainer import AdaptiveTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train GrowingSparseSNN')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to config file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to load checkpoint from')
    parser.add_argument('--output', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'model': {
                'input_features': 450,  # 15x15 grid x 2 channels
                'init_hidden': 64,
                'max_hidden': 512,
                'output_features': 4,  # 4 actions
                'tau_mem': 20.0,
                'tau_syn': 10.0,
            },
            'environment': {
                'size': 15,
                'n_food': 10,
                'max_steps': 500,
                'food_respawn': True,
            },
            'training': {
                'episodes': args.episodes,
                'lr': args.lr,
                'gamma': 0.99,
                'timesteps': 10,
                'grow_interval': 100,
                'prune_interval': 50,
                'grow_threshold': 0.05,
                'prune_threshold': 0.005,
                'log_interval': 10,
                'eval_interval': 50,
            }
        }
    
    # Device selection
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Check for ROCm on AMD GPU
    if device == 'cuda' and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch version: {torch.__version__}")
        if 'rocm' in torch.__version__:
            logger.info("✓ ROCm detected!")
    
    # Create model
    model_config = config['model']
    model = GrowingSparseSNN(
        input_features=model_config['input_features'],
        init_hidden=model_config['init_hidden'],
        max_hidden=model_config['max_hidden'],
        output_features=model_config['output_features'],
        tau_mem=model_config.get('tau_mem', 20.0),
        tau_syn=model_config.get('tau_syn', 10.0),
        device=device
    )
    
    logger.info(f"Model created: {model.get_stats()}")
    
    # Create environment
    env_config = config['environment']
    env = ForagingGrid(
        size=env_config['size'],
        n_food=env_config['n_food'],
        max_steps=env_config['max_steps'],
        food_respawn=env_config.get('food_respawn', True)
    )
    
    logger.info(f"Environment created: {env_config['size']}x{env_config['size']} grid")
    
    # Create trainer
    train_config = config['training']
    trainer = AdaptiveTrainer(
        model=model,
        env=env,
        lr=train_config['lr'],
        gamma=train_config.get('gamma', 0.99),
        device=device,
        metrics_port=8000
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Train
    logger.info("Starting training...")
    trainer.train(
        episodes=train_config['episodes'],
        timesteps=train_config.get('timesteps', 10),
        grow_interval=train_config.get('grow_interval', 100),
        prune_interval=train_config.get('prune_interval', 50),
        grow_threshold=train_config.get('grow_threshold', 0.05),
        prune_threshold=train_config.get('prune_threshold', 0.005),
        log_interval=train_config.get('log_interval', 10),
        eval_interval=train_config.get('eval_interval', 50),
    )
    
    # Save final model
    final_path = output_dir / 'final_model.pt'
    trainer.save_checkpoint(str(final_path))
    
    logger.info(f"Training complete! Model saved to {final_path}")
    logger.info(f"Final stats: {model.get_stats()}")


if __name__ == '__main__':
    main()
