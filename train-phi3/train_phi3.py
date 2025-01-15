import logging
import sys
import os
import torch
import json
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def print_gpu_memory():
    if torch.cuda.is_available():
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def create_chinese_poetry_dataset():
    # 获取数据集路径
    dataset_dir = get_dataset_path()
    if dataset_dir is None:
        raise ValueError("Failed to download or locate the dataset")

    # 初始化训练和验证数据
    train_data = []
    val_data = []

    # 遍历目录中的所有 JSON 文件
    logger.info("Processing JSON files:")
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(dataset_dir, filename)
            logger.info(f"Reading file: {filename}")
            
            entries_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        entry = json.loads(line)
                        entries_count += 1
                        
                        # 在每条数据末尾添加 END 标记
                        formatted_text = f"{entry['prompt']}\n{entry['response']}\nEND\n"
                        
                        # 随机分配到训练集或验证集
                        if np.random.random() < 0.9:  # 90% 概率进入训练集
                            train_data.append(formatted_text)
                        else:
                            val_data.append(formatted_text)
            
            logger.info(f"  - Processed {entries_count} entries from {filename}")

    logger.info(f"\nTotal entries: Train = {len(train_data)}, Val = {len(val_data)}")

    train_dataset = Dataset.from_dict({"text": train_data})
    eval_dataset = Dataset.from_dict({"text": val_data})
    
    return DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })

def get_dataset_path(dataset_name="Iess/chinese_modern_poetry"):
    """获取数据集路径，如果本地不存在则下载"""
    try:
        from huggingface_hub import snapshot_download
        # 尝试使用 snapshot_download 下载整个数据集
        dataset_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=None,  # 使用默认的缓存目录
            ignore_patterns=[".*"],
        )
        logger.info(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def test_generation(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    logger.info("Starting training script")
    print_gpu_memory()
    
    # Set memory efficient settings
    torch.cuda.empty_cache()
    
    # Load configuration
    logger.info("Loading configuration")
    config = AutoConfig.from_pretrained("./phi3_config.json")
    logger.info(f"Model config: {config}")
    
    # Create tokenizer
    logger.info("Creating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print_gpu_memory()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    print_gpu_memory()
    
    # Load dataset
    logger.info("Loading dataset")
    dataset = create_chinese_poetry_dataset()
    
    # Preprocessing function
    def preprocess_function(examples):
        logger.info(f"Processing batch of {len(examples['text'])} examples")
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_overflowing_tokens=False,  
            return_length=True,
        )
        logger.info(f"Tokenization complete. Input length: {tokenized['length'][0]}")
        
        # Create labels for language modeling (input_ids shifted right)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Preprocess the dataset
    logger.info("Preprocessing dataset")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    print_gpu_memory()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        report_to=["tensorboard"],
    )
    
    logger.info(f"Training arguments: {training_args}")
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training")
    logger.info(f"Number of training examples: {len(tokenized_datasets['train'])}")
    logger.info(f"Number of validation examples: {len(tokenized_datasets['validation'])}")
    logger.info(f"Number of epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    
    # Start training
    print_gpu_memory()
    trainer.train()
    
    # Test generation
    logger.info("\nTesting generation:")
    test_prompts = [
        "The Earth is",
        "Deep neural networks",
        "The Renaissance period",
        "In the ancient forest",
        "The human brain",
        "The golden sunset",
        "Scientists discovered that",
        "The future of technology",
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        generated = test_generation(model, tokenizer, prompt, max_length=100)  
        logger.info(f"Generated: {generated}")

if __name__ == "__main__":
    main()
