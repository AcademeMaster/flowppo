"""ACFQL agent configuration file."""

from ml_collections import ConfigDict

def get_config():
    """Get ACFQL agent configuration."""
    config = ConfigDict()
    
    # Agent settings
    config.agent_name = 'acfql'
    config.batch_size = 256
    config.critic_lr = 3e-4
    config.actor_lr = 3e-4
    config.tau = 0.005
    config.discount = 0.99
    config.target_update_period = 1
    config.alpha = 100.0  # Default alpha value, can be overridden
    
    # Action chunking settings
    config.action_chunking = True
    config.horizon_length = 5  # Default, can be overridden
    
    # Actor settings
    config.actor_type = 'best-of-n'
    config.actor_num_samples = 32
    
    # Network architecture
    config.hidden_dims = (256, 256)
    config.use_layer_norm = False
    config.dropout_rate = None
    
    # Flow-based settings for ACFQL
    config.flow_steps = 32
    config.use_fourier_features = True
    config.fourier_features = 256
    config.fourier_scale = 1.0
    
    # Q-function settings
    config.q_agg = 'min'  # 'min' or 'mean'
    config.num_critics = 2
    
    return config
