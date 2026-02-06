"""
Prometheus/OpenMetrics integration for monitoring SNN training.

Exports metrics about:
- Network size (neuron count)
- Firing rates and sparsity
- Growth and pruning events
- Learning performance (rewards, loss)
- Energy estimates
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server
import logging

logger = logging.getLogger(__name__)


class SNNMetrics:
    """Prometheus metrics for SNN monitoring."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        
        # Network structure metrics
        self.neuron_count = Gauge('snn_neuron_count', 'Current number of neurons in hidden layer')
        self.max_neurons = Gauge('snn_max_neurons', 'Maximum allowed neurons')
        
        # Activity metrics
        self.avg_firing_rate = Gauge('snn_avg_firing_rate', 'Average firing rate across neurons')
        self.sparsity = Gauge('snn_sparsity', 'Proportion of neurons with very low activity')
        
        # Plasticity events
        self.growth_events = Counter('snn_growth_events_total', 'Total neurogenesis events')
        self.pruning_events = Counter('snn_pruning_events_total', 'Total pruning events')
        self.neurons_added = Counter('snn_neurons_added_total', 'Total neurons added')
        self.neurons_removed = Counter('snn_neurons_removed_total', 'Total neurons removed')
        
        # Learning metrics
        self.episode_reward = Gauge('snn_episode_reward', 'Reward in current episode')
        self.episode_length = Gauge('snn_episode_length', 'Steps in current episode')
        self.total_episodes = Counter('snn_episodes_total', 'Total episodes completed')
        
        # Performance metrics
        self.training_loss = Gauge('snn_training_loss', 'Training loss')
        self.food_collected = Counter('snn_food_collected_total', 'Total food items collected')
        
        # Energy estimate (spike count × neuron count)
        self.energy_estimate = Gauge('snn_energy_estimate', 'Estimated energy consumption')
        
        # Model parameters
        self.total_params = Gauge('snn_total_parameters', 'Total trainable parameters')
        
        logger.info(f"Metrics initialized, will serve on port {port}")
    
    def start_server(self):
        """Start Prometheus HTTP server."""
        start_http_server(self.port)
        logger.info(f"Prometheus metrics server started on port {self.port}")
    
    def update_from_snn(self, snn_stats: dict):
        """Update metrics from SNN statistics dictionary."""
        self.neuron_count.set(snn_stats.get('n_neurons', 0))
        self.max_neurons.set(snn_stats.get('max_neurons', 0))
        self.avg_firing_rate.set(snn_stats.get('avg_firing_rate', 0))
        self.sparsity.set(snn_stats.get('sparsity', 0))
        self.total_params.set(snn_stats.get('total_params', 0))
        
        # Update counters (only increment by difference)
        # Note: Counters should only increase, handle elsewhere
    
    def record_episode(self, reward: float, length: int, food: int = 0):
        """Record episode statistics."""
        self.episode_reward.set(reward)
        self.episode_length.set(length)
        self.total_episodes.inc()
        if food > 0:
            self.food_collected.inc(food)
    
    def record_growth(self, n_added: int):
        """Record growth event."""
        if n_added > 0:
            self.growth_events.inc()
            self.neurons_added.inc(n_added)
    
    def record_pruning(self, n_removed: int):
        """Record pruning event."""
        if n_removed > 0:
            self.pruning_events.inc()
            self.neurons_removed.inc(n_removed)
    
    def record_loss(self, loss: float):
        """Record training loss."""
        self.training_loss.set(loss)
    
    def estimate_energy(self, n_spikes: float, n_neurons: int):
        """Estimate energy as spike_count × neuron_count."""
        energy = n_spikes * n_neurons
        self.energy_estimate.set(energy)


# Global metrics instance
_metrics_instance = None


def get_metrics(port: int = 8000) -> SNNMetrics:
    """Get or create global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = SNNMetrics(port=port)
    return _metrics_instance


if __name__ == '__main__':
    # Test metrics
    print("Testing SNNMetrics...")
    
    metrics = get_metrics(port=8001)
    
    # Simulate some metrics
    metrics.neuron_count.set(128)
    metrics.avg_firing_rate.set(0.05)
    metrics.record_growth(32)
    metrics.record_episode(reward=100.5, length=250, food=5)
    
    print("Metrics updated successfully")
    print("Visit http://localhost:8001 to see metrics")
    
    # Start server (would block in real usage)
    # metrics.start_server()
    # import time
    # time.sleep(100)
