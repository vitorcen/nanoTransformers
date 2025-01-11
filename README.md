# Small Language Models Training Project

This project focuses on training and fine-tuning small-scale language models (Phi3 and Mixtral) using the Hugging Face Transformers library. The implementation includes efficient training strategies and custom configurations for both models.

## Project Structure

```
.
├── README.md
├── train-phi3/
│   ├── train_phi3.py
│   ├── phi3_config.json
│   └── train_log.md
└── train-mixtral/
    ├── train_mixtral.py
    ├── mixtral_config.json
    └── train_log.md
```

## Prerequisites

- Python 3.8+
- CUDA 12.1+ (for GPU support)
- Git

## Installation

1. Clone the Hugging Face Transformers repository:
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
```

2. Create and activate a conda environment:
```bash
conda create -n llm_training python=3.12
conda activate llm_training
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate huggingface_hub
```

## Training Models

### Training Phi3

1. Navigate to the phi3 training directory:
```bash
cd train-phi3
```

2. Run the training script:
```bash
python train_phi3.py
```

The training progress and results will be logged in `train_log.md`.

### Training Mixtral

1. Navigate to the mixtral training directory:
```bash
cd train-mixtral
```

2. Run the training script:
```bash
python train_mixtral.py
```

The training progress and results will be logged in `train_log.md`.

## Model Configurations

### Phi3 Configuration
- Hidden Size: 512
- Intermediate Size: 2048
- Attention Heads: 8
- Hidden Layers: 6
- Max Position Embeddings: 256
- Vocabulary Size: 51200

### Mixtral Configuration
- Hidden Size: 256
- Intermediate Size: 512
- Attention Heads: 8
- Hidden Layers: 4
- Number of Experts: 8
- Experts per Token: 2
- Max Position Embeddings: 4096
- Vocabulary Size: 32000

## Training Logs

Training progress, model performance, and generation examples are documented in the respective `train_log.md` files in each model's directory.

## Notes

- The models are configured for research and experimentation purposes
- Training parameters can be adjusted in the respective training scripts
- GPU memory usage is optimized using gradient checkpointing and mixed precision training
