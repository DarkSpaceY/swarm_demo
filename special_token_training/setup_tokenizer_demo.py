from transformers import AutoTokenizer
import torch

def setup_tokenizer(model_name="unsloth/Qwen2.5-0.5B-Instruct"):
    """
    演示如何将 special tokens 加入 tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 定义 special tokens
    special_tokens_dict = {
        'additional_special_tokens': [
            '<|send|>', 
            '<|to|>', 
            '<|to_all|>', 
            '<|msg|>', 
            '<|fin|>'
        ]
    }
    
    # 增加 tokens
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")
    
    # 验证 token 是否被正确编码
    test_str = "<|send|><|to|>1,2<|msg|>Hello everyone!<|fin|>42"
    tokens = tokenizer.tokenize(test_str)
    print(f"Tokenized: {tokens}")
    
    encoded = tokenizer.encode(test_str)
    print(f"Encoded IDs: {encoded}")
    
    # 提醒：如果是微调，需要调用 model.resize_token_embeddings(len(tokenizer))
    return tokenizer

if __name__ == "__main__":
    setup_tokenizer()
