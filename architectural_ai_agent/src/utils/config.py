"""
Configuration Management Module

Handles configuration settings for the architectural AI agent.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the architectural AI agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_data = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            self._load_config_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            'model': {
                'segmentation': {
                    'architecture': 'unet',
                    'encoder': 'resnet50',
                    'num_classes': 15,
                    'input_size': [512, 512]
                },
                'depth_estimation': {
                    'architecture': 'custom',
                    'input_size': [256, 256]
                },
                'object_detection': {
                    'architecture': 'custom',
                    'num_classes': 13,
                    'confidence_threshold': 0.5
                }
            },
            'preprocessing': {
                'target_size': [1024, 1024],
                'noise_reduction': True,
                'perspective_correction': True,
                'line_enhancement': True
            },
            'reconstruction': {
                'output_format': 'obj',
                'standard_heights': {
                    'wall_height': 2.4,
                    'door_height': 2.1,
                    'window_height': 1.2,
                    'window_sill': 0.9,
                    'counter_height': 0.9,
                    'cabinet_height': 2.1,
                    'fixture_height': 0.8
                },
                'wall_thickness': 0.15,
                'door_thickness': 0.05,
                'window_thickness': 0.02
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 0.001,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            },
            'paths': {
                'models_dir': 'models',
                'data_dir': 'data',
                'output_dir': 'output',
                'logs_dir': 'logs'
            },
            'device': 'auto',  # 'cpu', 'cuda', or 'auto'
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def _load_config_file(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    file_config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {config_path}")
                    return
            
            # Merge with default config
            self._deep_update(self.config_data, file_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.segmentation.architecture')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self.config_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        current = self.config_data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save(self, config_path: str):
        """Save configuration to file."""
        try:
            with open(config_path, 'w') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                elif config_path.endswith('.json'):
                    json.dump(self.config_data, f, indent=2)
                else:
                    logger.error(f"Unsupported config file format: {config_path}")
                    return
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config_data.copy()
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary."""
        self._deep_update(self.config_data, updates)