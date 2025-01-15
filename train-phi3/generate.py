import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and format the generated text."""
    # 移除空格
    text = text.replace(" ", "")
    # 移除特殊标记
    text = text.replace("end", "")
    # 移除提示词
    text = text.replace("使用下列意象写一首现代诗：", "")
    text = text.replace("使用下列意象写一首现代", "")
    # 移除标题相关
    text = text.split("标题:")[0]
    # 移除标点相关
    text = text.replace("标点，", "")
    # 移除冗余标点
    text = text.replace("，，", "，")
    text = text.replace("。。", "。")
    # 移除一些常见的不合适的词语
    text = text.replace("标", "")
    text = text.replace("酒店", "")
    text = text.replace("开会", "")
    text = text.replace("文化", "")
    text = text.replace("写一首", "")
    text = text.replace("使用", "")
    text = text.replace("下列", "")
    text = text.replace("意象", "")
    text = text.replace("现代诗", "")
    text = text.replace("：", "")
    # 移除一些口语化的词语
    text = text.replace("就是", "")
    text = text.replace("这样", "")
    text = text.replace("可以", "")
    text = text.replace("不是", "")
    text = text.replace("没有", "")
    text = text.replace("一个", "")
    text = text.replace("什么", "")
    text = text.replace("还是", "")
    text = text.replace("知道", "")
    text = text.replace("不会", "")
    text = text.replace("要", "")
    text = text.replace("会", "")
    
    # 移除连续的标点
    while "，，" in text:
        text = text.replace("，，", "，")
    while "。。" in text:
        text = text.replace("。。", "。")
        
    # 移除开头的标点
    while text and text[0] in "，。":
        text = text[1:]
        
    # 移除结尾的标点
    while text and text[-1] in "，":
        text = text[:-1]
        
    # 如果最后没有句号，添加句号
    if text and text[-1] != "。":
        text += "。"
        
    return text

def generate_text(checkpoint_path, prompt, max_length=100, num_return_sequences=5, temperature=1.2):
    # Load tokenizer and model
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    logger.info("Loading model from checkpoint")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    logger.info(f"Generating text with prompt: {prompt}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.8,
            no_repeat_ngram_size=4,
            top_k=40,
            top_p=0.85,
            length_penalty=2.0,
            min_length=30,
        )
    
    # Decode and print generations
    generated_texts = []
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        text = clean_text(text)
        
        # 如果生成的文本太短，跳过
        if len(text) < 20:
            continue
            
        logger.info(f"\nGeneration {i+1}:\n{text}")
        generated_texts.append(text)
    
    return generated_texts

if __name__ == "__main__":
    checkpoint_path = "./results/checkpoint-3000"
    prompts = [
        "春天来了，",
        "夜晚的城市，",
        "远方的山，",
        "我想起了，",
        "雨后的天空，",
        "秋天的落叶，",
        "清晨的阳光，",
        "月光下的湖面，",
        "黄昏时分，",
        "雪花飘落，",
        "海边的风，",
    ]
    
    for prompt in prompts:
        logger.info(f"\n=== Generating with prompt: {prompt} ===")
        generate_text(checkpoint_path, prompt, max_length=100, temperature=1.2)
