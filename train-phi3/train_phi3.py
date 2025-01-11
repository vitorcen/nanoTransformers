import logging
import sys
import os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPT2TokenizerFast,
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

def create_sample_dataset():
    # Create a more diverse dataset for testing
    train_texts = [
        # Technical content
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "Deep neural networks consist of multiple layers of interconnected nodes, processing information hierarchically.",
        "Natural language processing combines linguistics and machine learning to understand human language.",
        
        # General knowledge
        "The Earth orbits around the Sun in an elliptical path, completing one revolution in approximately 365.25 days.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy to produce glucose.",
        "The human brain contains approximately 86 billion neurons, forming complex neural networks.",
        
        # Creative writing
        "The golden sunset painted the sky in brilliant hues of orange and purple, as birds flew homeward.",
        "In the ancient forest, towering trees whispered secrets that had been kept for centuries.",
        "The bustling city never sleeps, its lights twinkling like earthbound stars in the night.",
        
        # Historical facts
        "The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing processes.",
        "Ancient Egyptians built the pyramids as tombs for their pharaohs, using sophisticated engineering techniques.",
        "The Renaissance period marked a revival of art, science, and classical learning in Europe.",
    ] * 50  # Repeat to create more samples
    
    eval_texts = [
        "Quantum computing leverages the principles of quantum mechanics to process information.",
        "Climate change affects global weather patterns and ecosystems in complex ways.",
        "The development of writing systems revolutionized human civilization and knowledge transfer.",
        "Space exploration has led to numerous technological advances benefiting everyday life.",
    ] * 25  # Repeat to create more samples
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})
    
    return DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset
    })

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
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', local_files_only=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    logger.info("Creating model")
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print_gpu_memory()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    print_gpu_memory()
    
    # Create dataset
    logger.info("Creating dataset")
    raw_datasets = create_sample_dataset()
    
    # Preprocessing function
    def preprocess_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./train-output",  
        overwrite_output_dir=True,
        num_train_epochs=3,           
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        evaluation_strategy="steps",
        eval_steps=50,
        fp16=True,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        report_to="none",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # Use full dataset
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    
    # Start training
    logger.info("Starting training")
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
