"""Configuration management for the spam classifier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Config:
    """Configuration class for the spam classifier."""
    # Data
    data_path: str = "mail_data.csv"
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature extraction
    min_df: int = 1
    max_df: float = 0.95
    stop_words: str = "english"
    max_features: Optional[int] = None
    
    # Model
    model_type: str = "logistic_regression"
    max_iter: int = 100
    
    # Paths
    model_dir: str = "models"
    logs_dir: str = "logs"
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Config instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML config file
        """
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
