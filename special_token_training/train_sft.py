import os
import torch
import logging
import json
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# 默认训练配置
DEFAULT_TRAINING_CONFIG = {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora": {"r": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "lora_alpha": 16, "lora_dropout": 0},
    "output_dir": "outputs/special_token_sft",
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "logging_steps": 1,
    "optimizer": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42,
}

# 尝试导入主程序的配置加载器，如果失败则使用默认配置
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import get_config
    CONFIG = get_config()
    # 如果主配置中没有 training 部分，或者 training 部分不完整，进行合并
    if "training" not in CONFIG:
        CONFIG["training"] = DEFAULT_TRAINING_CONFIG
    else:
        # 深度合并确保所有字段都存在
        for k, v in DEFAULT_TRAINING_CONFIG.items():
            if k not in CONFIG["training"]:
                CONFIG["training"][k] = v
except Exception:
    CONFIG = {
        "training": DEFAULT_TRAINING_CONFIG,
        "agent": {
            "role_system": "system",
            "role_user": "user",
            "role_assistant": "assistant"
        }
    }

logging.basicConfig(level=logging.INFO)

def train_special_tokens():
    train_cfg = CONFIG["training"]
    model_name = train_cfg["model_name"]
    max_seq_length = train_cfg["max_seq_length"]
    load_in_4bit = train_cfg["load_in_4bit"]
    
    # 1. 加载模型和 Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None, # 自动检测
        load_in_4bit = load_in_4bit,
    )

    # 2. 注入 Special Tokens
    # 注意：这些 tokens 必须与 generate_sft_data.py 中使用的 TOKENS 一致
    special_tokens = ["<|send|>", "<|to|>", "<|to_all|>", "<|msg|>", "<|fin|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. 设置 Chat Template
    # 我们使用 unsloth 提供的模板，并确保它能正确处理我们注入的 tokens
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5", # 或者根据你的基座模型选择
        mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
    )

    # 4. 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = train_cfg["lora"]["r"],
        target_modules = train_cfg["lora"]["target_modules"],
        lora_alpha = train_cfg["lora"]["lora_alpha"],
        lora_dropout = train_cfg["lora"]["lora_dropout"],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = train_cfg["seed"],
    )

    # 5. 加载数据集
    dataset_path = os.path.join(os.path.dirname(__file__), "sft_dataset.jsonl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件未找到: {dataset_path}。请先运行 generate_sft_data.py")

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 格式化函数：将 messages 列表转换为对话字符串
    def formatting_prompts_func(examples):
        instructions = examples["messages"]
        texts = []
        for messages in instructions:
            # 使用 tokenizer.apply_chat_template 进行格式化
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 6. 训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size = train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"],
        warmup_steps = train_cfg["warmup_steps"],
        num_train_epochs = train_cfg["num_train_epochs"],
        learning_rate = train_cfg["learning_rate"],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = train_cfg["logging_steps"],
        optim = train_cfg["optimizer"],
        weight_decay = train_cfg["weight_decay"],
        lr_scheduler_type = train_cfg["lr_scheduler_type"],
        seed = train_cfg["seed"],
        output_dir = train_cfg["output_dir"],
        report_to = "none", # 默认不开启 wandb 等
    )

    # 7. 开始训练
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # 序列通常不长，不需要 packing
        args = training_args,
    )

    trainer_stats = trainer.train()
    
    # 8. 保存模型
    final_output_dir = os.path.join(train_cfg["output_dir"], "final_lora")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    logging.info(f"微调完成。模型已保存至: {final_output_dir}")
    return trainer_stats

if __name__ == "__main__":
    train_special_tokens()
