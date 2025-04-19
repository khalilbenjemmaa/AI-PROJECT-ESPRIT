import os
import numpy as np
from pathlib import Path
import yaml


def save_config(config, save_path):
    """Save configuration to a YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory based on epoch number."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
        
    # Extract epoch numbers
    epochs = [int(f.split('_')[1].split('.')[0]) for f in checkpoints]
    latest_idx = np.argmax(epochs)
    
    return os.path.join(checkpoint_dir, checkpoints[latest_idx])


def get_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint in the directory."""
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_path):
        return best_path
    
    # If best model doesn't exist, get latest
    return get_latest_checkpoint(checkpoint_dir)


def prepare_experiment_dir(base_dir, experiment_name):
    """Create a directory structure for an experiment."""
    experiment_dir = Path(base_dir) / experiment_name
    
    # Create directories
    dirs = {
        'base': experiment_dir,
        'checkpoints': experiment_dir / 'checkpoints',
        'visualizations': experiment_dir / 'visualizations',
        'logs': experiment_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs


def save_model_summary(model, save_path):
    """Save model summary to a text file."""
    try:
        from torchsummary import summary
        
        # Redirect stdout to file
        import sys
        original_stdout = sys.stdout
        
        with open(save_path, 'w') as f:
            sys.stdout = f
            # Print model architecture (assuming model is on CPU)
            # and input size is (1, 256, 256)
            summary(model, (1, 256, 256))
            
        # Reset stdout
        sys.stdout = original_stdout
        
        print(f"Model summary saved to {save_path}")
    except ImportError:
        print("torchsummary package not found. Install with: pip install torchsummary")
        
        # Fallback - just save model's string representation
        with open(save_path, 'w') as f:
            f.write(str(model))
            
        print(f"Basic model structure saved to {save_path}")    