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

## Training Configuration

### Mixtral Model
- Model: Mixtral-8x7B-v0.1
- Dataset: Iess/chinese_modern_poetry
- Training Parameters:
  - Batch size: 16 per device
  - Gradient accumulation steps: 4
  - Learning rate: 5e-5
  - Weight decay: 0.01
  - Warmup steps: 100
  - Max sequence length: 512
  - Training epochs: 20
  - Evaluation strategy: Steps (every 500 steps)
  - Logging steps: 100
  - FP16 training enabled

### Phi3 Model
- Hidden Size: 512
- Intermediate Size: 2048
- Attention Heads: 8
- Hidden Layers: 6
- Max Position Embeddings: 256
- Vocabulary Size: 51200

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

## Training Chinese Poetry Model

We have successfully trained a Chinese poetry generation model using the Phi-3 architecture. The model was trained on a dataset of Chinese modern poetry.

### Training Process

1. Setup the environment:
```bash
pip install transformers datasets torch jieba sentencepiece
```

2. Train the model:
```bash
cd train-phi3
python train_phi3.py
```

The training script will:
- Use the Chinese RoBERTa tokenizer
- Load the Chinese modern poetry dataset
- Train for 3 epochs with appropriate hyperparameters
- Save checkpoints during training

### Generating Poetry

After training, you can generate Chinese poetry using the trained model:

```bash
cd train-phi3
python generate.py
```

Example prompts:
- "春天来了，"
- "夜晚的城市，"
- "远方的山，"
- "雨后的天空，"

The generation script includes various parameters to control the output:
- Temperature: Controls randomness (higher = more random)
- Top-k and Top-p: Control sampling strategy
- Repetition penalty: Avoid repeating phrases
- Min/Max length: Control the length of generated text

### Model Details

- Base architecture: Phi-3
- Tokenizer: `hfl/chinese-roberta-wwm-ext`
- Training data: Chinese modern poetry dataset
- Parameters: ~45M
- Training time: Varies by hardware
- GPU Memory usage: ~200MB

### Example Outputs

```
春天来了，
花开满枝头，
微风轻抚过，
带来远方的呼唤。

夜晚的城市，
霓虹闪烁，
街道上行人匆匆，
像流动的星辰。
```

## Mixtral Chinese Poetry Generation

This project implements a Chinese poetry generation model based on the Mixtral architecture. The model is trained on a dataset of Chinese poems and can generate poetic text in response to prompts.

### Model Architecture

The model uses a modified Mixtral architecture with the following specifications:
- Hidden size: 768
- Attention heads: 24
- Hidden layers: 12
- Intermediate size: 1536
- Key-value heads: 12
- Local experts: 8
- Experts per token: 2
- Total parameters: 361.24M

### Training Details

The model is trained with the following configuration:
- Training epochs: 50
- Batch size: 12
- Gradient accumulation steps: 6
- Learning rate: 5e-5
- Weight decay: 0.01
- Mixed precision training (fp16)
- Custom Chinese tokenizer

### Usage

To generate poetry, use the `generate.py` script:

```bash
python generate.py
```

Example prompts:
```python
prompts = [
    "春天的花园，",
    "月光下的湖泊，",
    "山间的小路，",
    "夏日的午后，",
    "秋风吹过，"
]
```

### Generation Parameters

The generation uses the following parameters for balanced creativity and coherence:
- Max length: 100
- Number of beams: 5
- Temperature: 0.8
- Top-k: 30
- Top-p: 0.9
- Repetition penalty: 1.5
- No repeat ngram size: 3
- Length penalty: 1.2

### Model Checkpoints

Checkpoints are saved every 1000 steps in the `mixtral-chinese-poetry` directory. Use the latest checkpoint for best results.

### Future Improvements

Planned improvements include:
1. Expanding the training dataset with more diverse poetry
2. Fine-tuning generation parameters for better coherence
3. Implementing more sophisticated post-processing for better formatting
4. Adding support for different poetry styles and formats

## Notes

- The models are configured for research and experimentation purposes
- Training parameters can be adjusted in the respective training scripts
- GPU memory usage is optimized using gradient checkpointing and mixed precision training
