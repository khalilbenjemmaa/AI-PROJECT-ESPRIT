## Project Structure

```
marine-debris-segmentation/
├── data.py               # Dataset loading and processing
├── model.py              # Model training and evaluation functions
├── utils.py              # Utility functions for configuration and checkpoints
├── main.py               # Main entry point for training/evaluation
├── run.py                # User-friendly command-line interface
├── output/               # Generated outputs
│   ├── checkpoints/      # Model checkpoints
│   └── visualizations/   # Training plots and prediction visualizations
└── marine-debris-fls-datasets/ # Dataset directory (cloned from GitHub)
```

## Installation

1. Clone the dataset repository:
```bash
git clone https://github.com/mvaldenegro/marine-debris-fls-datasets
```

2. Install the required dependencies:
```bash
pip install -r requiremements.txt
```

## Usage

### Training a Model

To train a new model with custom parameters:

```bash
python main.py --data_path path/to/dataset \
               --output_dir output/my_experiment \
               --epochs 30 \
               --batch_size 16 \
               --learning_rate 1e-4 \
               --image_size 256 \
               --num_classes 12 \
               --save_interval 1
```


## Model Checkpoints

Checkpoints are saved at regular intervals and when a new best model is found (based on validation IoU). Each checkpoint contains:

- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training configuration
- Performance metrics (validation IoU, loss values)

## Visualizations

The code automatically generates:

- Sample images and masks from the dataset
- Batch visualizations during training
- Prediction visualizations during/after training
- Training metrics plots (loss and IoU curves)
