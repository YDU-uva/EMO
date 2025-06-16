# EMO: Episodic Memory Optimization for Few-Shot Meta-Learning

## Project Overview

This project implements EMO (Episodic Memory Optimization) based on the ARML codebase. EMO is a novel meta-learning approach that leverages episodic memory mechanisms to enhance few-shot learning performance through memory-based optimization and uncertainty quantification.

## Method Features

### 1. Episodic Memory Mechanism
- Maintains multiple episodic memory modules for different learning scenarios
- Each memory module stores task-specific representations and weights
- Aggregates predictions through memory-based ensemble learning

### 2. Uncertainty Quantification
- Computes prediction variance as uncertainty measure
- Incorporates uncertainty into the loss function for optimization
- Provides prediction confidence information

### 3. Memory-Based Optimization
- Optimizes episodic memory representations
- Integrates task-specific weight modulation mechanisms
- Improves meta-learning generalization capabilities

## Core Files

- `emo.py`: Core implementation of EMO model
- `main_emo.py`: Main program for EMO training and testing
- `run_emo.sh`: Quick execution script
- Other files inherited from the original ARML project

## Dependencies

```bash
# Python 3.x
# TensorFlow 1.10+
# Numpy 1.15+
pip install tensorflow==1.15
pip install numpy
```

## Usage

### Quick Start

```bash
# Add execution permission to the script
chmod +x run_emo.sh

# Run training and testing
./run_emo.sh
```

### Manual Execution

#### Training

```bash
python main_emo.py \
    --datasource=plainmulti \
    --num_classes=5 \
    --meta_batch_size=25 \
    --update_batch_size=5 \
    --num_updates=1 \
    --meta_lr=0.001 \
    --update_lr=0.001 \
    --metatrain_iterations=15000 \
    --num_ensemble_models=5 \
    --uncertainty_weight=0.1 \
    --expected_output_weight=1.0 \
    --train=True
```

#### Testing

```bash
python main_emo.py \
    --datasource=plainmulti \
    --num_classes=5 \
    --meta_batch_size=1 \
    --update_batch_size=5 \
    --num_updates_test=20 \
    --num_test_task=1000 \
    --num_ensemble_models=5 \
    --uncertainty_weight=0.1 \
    --expected_output_weight=1.0 \
    --train=False \
    --test_set=True
```

## Parameters

### EMO-Specific Parameters

- `--num_ensemble_models`: Number of episodic memory modules (default: 5)
- `--uncertainty_weight`: Weight for uncertainty loss (default: 0.1)
- `--expected_output_weight`: Weight for memory-based output loss (default: 1.0)

### Dataset Options

- `--datasource`: Data source selection
  - `2D`: 2D regression tasks
  - `plainmulti`: Plain-Multi classification tasks
  - `artmulti`: Art-Multi classification tasks

### Training Parameters

- `--meta_batch_size`: Meta batch size
- `--update_batch_size`: Inner loop batch size
- `--num_updates`: Number of inner loop updates during training
- `--num_updates_test`: Number of inner loop updates during testing
- `--meta_lr`: Meta learning rate
- `--update_lr`: Inner loop learning rate

## Model Architecture

### EMO Class Structure

```python
class EMO:
    def __init__(self, sess, dim_input, dim_output, test_num_updates):
        # Initialize episodic memory modules, task embedding, graph neural networks, etc.
        
    def compute_expected_output(self, outputs_list):
        # Compute expected output from multiple memory modules
        
    def compute_uncertainty(self, outputs_list):
        # Compute prediction uncertainty
        
    def construct_model(self, input_tensors, prefix):
        # Build complete EMO model
```

### Key Components

1. **Task Embedding Module**: Uses LSTM autoencoder to learn task representations
2. **Graph Neural Network**: Leverages MetaDAG for knowledge propagation
3. **Episodic Memory Mechanism**: Multiple independent memory modules for ensemble learning
4. **Uncertainty Estimation**: Variance-based uncertainty quantification

## Experimental Results

EMO model provides improvements in the following aspects:

- **Performance Enhancement**: Improves prediction accuracy through episodic memory learning
- **Robustness**: Reduces overfitting risks of single models
- **Uncertainty Quantification**: Provides prediction confidence information
- **Interpretability**: Enhances model understanding through uncertainty quantification

## Directory Structure

```
├── emo.py                 # Core EMO model implementation
├── main_emo.py           # EMO main program
├── run_emo.sh            # Execution script
├── README_EMO.md         # This documentation
├── logs_emo/             # EMO model log directory
├── data_generator.py     # Data generator (inherited)
├── image_embedding.py    # Image embedding (inherited)
├── task_embedding.py     # Task embedding (inherited)
├── metadag.py           # Graph neural network (inherited)
└── utils.py             # Utility functions (inherited)
```

## Notes

1. Ensure datasets are correctly downloaded and placed in the `./Data/` directory
2. Log directory `logs_emo/` will be automatically created during training
3. Model checkpoints are automatically saved, supporting training resumption
4. Recommended to run in GPU environment for training acceleration

## Citation

If you use this code, please cite the original ARML paper:

```bibtex
@inproceedings{du2023emo,
  title={EMO: episodic memory optimization for few-shot meta-learning},
  author={Du, Yingjun and Shen, Jiayi and Zhen, Xiantong and Snoek, Cees GM},
  booktitle={Conference on Lifelong Learning Agents},
  pages={1--20},
  year={2023},
  organization={PMLR}
}
```

## Extensibility

EMO implementation provides good extensibility:

- Adjustable number of episodic memory modules
- Support for different uncertainty quantification methods
- Integration with other meta-learning algorithms
- Support for custom loss function weights

## Contact

For questions, please submit feedback through project issues. 