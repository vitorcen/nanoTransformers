from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig
import torch

def clean_generated_text(text):
    # 移除提示语部分
    if "使用下列意象写一首现代诗：" in text:
        text = text.split("使用下列意象写一首现代诗：")[-1]
    
    # 移除END标记
    text = text.replace("END", "").replace("end", "")
    
    # 提取标题和正文
    if "标题:" in text:
        parts = text.split("标题:")
        if len(parts) > 1:
            title = parts[1].split("\n")[0].strip()
            content = "\n".join(parts[1].split("\n")[1:]).strip()
            text = f"标题：{title}\n\n{content}"
    
    # 清理特殊字符
    text = text.replace("，，", "，")
    text = text.replace("。。", "。")
    text = text.replace("��", "")
    text = text.replace("�", "")
    
    # 清理多余的标点
    while text and text[0] in '，。':
        text = text[1:]
    
    # 确保结尾有适当的标点
    if text and text[-1] not in '。！？':
        text += "。"
    
    return text.strip()

def generate_poetry(model_path, prompt):
    # 加载模型和tokenizer
    config = MixtralConfig(
        vocab_size=32000,
        hidden_size=512,
        num_attention_heads=16,
        num_hidden_layers=8,
        intermediate_size=1024,
        max_position_embeddings=4096,
        num_key_value_heads=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
        use_cache=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = MixtralForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 构建完整的提示
    full_prompt = f"使用下列意象写一首现代诗：{prompt}\n标题："
    
    # 设置生成参数
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成文本
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        num_beams=5,
        temperature=0.9,  # 略微提高温度增加创造性
        do_sample=True,
        top_k=50,  # 增加可选词数
        top_p=0.95,  # 增加采样范围
        repetition_penalty=1.2,  # 降低重复惩罚
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_generated_text(generated_text)

def main():
    model_path = "./mixtral-chinese-poetry/checkpoint-4000"
    
    test_prompts = [
        "月光，湖面",
        "落叶，秋风",
        "雨，思念",
        "春天，花朵",
        "夜晚，星空",
    ]
    
    print("\n=== 现代诗生成 ===")
    for prompt in test_prompts:
        print(f"\n意象：{prompt}")
        try:
            poem = generate_poetry(model_path, prompt)
            print(f"{poem}\n{'='*30}")
        except Exception as e:
            print(f"生成失败: {str(e)}")

if __name__ == "__main__":
    main()
