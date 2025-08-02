#!/usr/bin/env python3
"""
Configuration helper for dog breed identifier project
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigHelper:
    """Helper class to load and access configuration settings"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration helper
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.config['data']['breed_dataset_path'],
            self.config['data']['my_dog_dataset_path'],
            self.config['paths']['output_dir'],
            self.config['paths']['checkpoints_dir'],
            self.config['paths']['logs_dir'],
            self.config['paths']['plots_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'data.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_model_config(self, model_type: str = 'breed_classifier') -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {}).get(model_type, {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Get augmentation configuration"""
        return self.config.get('augmentation', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.config.get('paths', {})
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary with configuration updates
        """
        for key, value in updates.items():
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
    
    def save_config(self, output_path: str = None):
        """
        Save current configuration to file
        
        Args:
            output_path: Output path (defaults to original config path)
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def print_config(self):
        """Print current configuration"""
        print("📋 Current Configuration:")
        print(json.dumps(self.config, indent=2))


def load_config(config_path: str = "config.json") -> ConfigHelper:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigHelper instance
    """
    return ConfigHelper(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    config.print_config()
    
    # Test getting specific values
    print(f"\n📊 Batch size: {config.get('data.batch_size')}")
    print(f"🎯 Learning rate: {config.get('training.learning_rate')}")
    print(f"🏗️ Model classes: {config.get('model.breed_classifier.num_classes')}") 