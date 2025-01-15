import logging
import sys
import os
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    MixtralConfig,
    MixtralForCausalLM,
    TrainerCallback,
)
import time
import json
import numpy as np

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

def create_poetry_dataset():
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

def test_generation(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_length=150,  # 增加最大长度
        num_return_sequences=1,
        num_beams=8,  # 增加beam search的beam数
        temperature=0.7,  # 适度调整temperature
        do_sample=True,
        top_k=20,  # 减小top_k值
        top_p=0.85,  # 减小top_p值
        repetition_penalty=1.8,  # 增加重复惩罚
        no_repeat_ngram_size=4,
        length_penalty=1.5,  # 添加长度惩罚
        bad_words_ids=[[tokenizer.encode(c)[0]] for c in "►◤⌒かりレむーンgsｼ］►"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def create_model_and_tokenizer():
    config = MixtralConfig(
        vocab_size=32000,
        hidden_size=512,  # 从768减少到512
        num_attention_heads=16,  # 从24减少到16
        num_hidden_layers=8,  # 从12减少到8
        intermediate_size=1024,  # 从1536减少到1024
        max_position_embeddings=4096,
        num_key_value_heads=8,  # 从12减少到8
        num_local_experts=8,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
        use_cache=False,
    )
    
    # 使用 Mixtral 官方的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        trust_remote_code=True,
        use_fast=True
    )
    
    # 确保有特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建模型
    model = MixtralForCausalLM(config)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer

def main():
    logger.info("Starting training script")
    print_gpu_memory()
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer")
    model, tokenizer = create_model_and_tokenizer()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    print_gpu_memory()
    
    # Create dataset
    logger.info("Creating dataset")
    raw_datasets = create_poetry_dataset()
    
    # Preprocessing function
    def preprocess_function(examples):
        # 只在第一次调用时打印信息
        if not hasattr(preprocess_function, 'called'):
            preprocess_function.called = True
            logger.info(f"Starting preprocessing with batch size: {len(examples['text'])}")
        
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_overflowing_tokens=False,
            return_length=True,
        )
        
        # Create labels for language modeling (input_ids shifted right)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Preprocess the dataset
    logger.info("Preprocessing dataset")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=1,
    )
    print_gpu_memory()

    # 计算数据集大小（以token为单位）
    train_dataset_size = sum(len(x) for x in tokenized_datasets["train"]["input_ids"])
    logger.info(f"Train dataset size: {train_dataset_size:,} tokens")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./mixtral-chinese-poetry",
        num_train_epochs=20,
        per_device_train_batch_size=8*2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        save_steps=500,
        fp16=True,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",  # 禁用wandb等报告
    )
    
    # 计算 MFU 所需的参数
    model_config = model.config
    n_params = sum(p.numel() for p in model.parameters())
    # 每个token的前向+后向传播需要的计算量
    forward_backward_flops_per_token = 2 * n_params * 2  # 2 for fwd+bwd, 2 for multiply+add
    
    # 自定义训练回调
    class CustomCallback(TrainerCallback):
        def __init__(self):
            self.train_start_time = time.time()
            self.best_val_loss = float('inf')
            self.total_flops = 0
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % args.logging_steps == 0:
                # 计算处理速度
                elapsed_time = time.time() - self.train_start_time
                tokens_per_sec = (state.global_step * args.per_device_train_batch_size * 
                                args.gradient_accumulation_steps) / elapsed_time
                
                # 计算 TFLOPS
                flops_per_sec = tokens_per_sec * forward_backward_flops_per_token
                tflops = flops_per_sec / 1e12
                
                # 计算 epoch 进度
                total_tokens = state.global_step * args.per_device_train_batch_size * \
                            args.gradient_accumulation_steps
                epochs = total_tokens / train_dataset_size
                
                logger.info(f"\nStep {state.global_step}:")
                logger.info(f"Training speed: {tflops:.2f} TFLOPS")
                logger.info(f"Epochs: {epochs:.2f}")
                logger.info(f"Tokens/sec: {tokens_per_sec:.0f}")
    
    # Initialize Trainer with callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=default_data_collator,
        callbacks=[CustomCallback()],
    )
    
    # Start training
    logger.info("Starting training")
    print_gpu_memory()
    trainer.train()
    
    # Test generation
    logger.info("\nTesting generation:")
    test_prompts = [
        "春天来了，",
        "夜晚的城市，",
        "远方的山，",
        "我想起了，",
        "雨后的天空，",
        "秋天的落叶，",
        "清晨的阳光，",
        "月光下的湖面，",
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        generated = test_generation(model, tokenizer, prompt)
        logger.info(f"Generated: {generated}")

if __name__ == "__main__":
    main()
