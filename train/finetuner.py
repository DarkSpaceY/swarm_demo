import logging
import os
import torch
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO)

def resolve_local_files_only(model_name, cache_dir, local_files_only):
    if local_files_only:
        return True
    if os.path.isdir(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(
            model_name,
            "config.json",
            cache_dir=cache_dir,
        )
        return cached is not None
    except Exception:
        return False

class SwarmTrainer:
    def __init__(self, model_name="unsloth/Qwen2.5-1.5B-Instruct", output_dir="./train/checkpoints", max_seq_length=2048, load_in_4bit=True, cache_dir=None, local_files_only=False):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        
        logging.info(f"Initializing SwarmTrainer with model: {model_name}")

        resolved_local_only = resolve_local_files_only(model_name, cache_dir, local_files_only)
        
        # 加载 Unsloth 模型和 Tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir,
            local_files_only=resolved_local_only,
        )
        
        # 配置 LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def train_agent(
        self,
        agent_id,
        data_path,
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        max_steps=None,
        resume_from_checkpoint=None,
    ):
        """
        为指定 Agent 训练 LoRA。
        
        Args:
            agent_id: str, Agent ID (例如 "Agent_0")
            data_path: str, 该 Agent 的训练数据 JSONL 路径
            epochs: int, 训练轮数
        """
        logging.info(f"Starting training for {agent_id} using data from {data_path}")
        
        try:
            dataset = load_dataset("json", data_files=data_path, split="train")
        except Exception as e:
            logging.warning(f"No data found for {agent_id}, skipping training. Error: {e}")
            return
        if dataset is None or len(dataset) == 0:
            logging.warning(f"No samples for {agent_id}, skipping training.")
            return

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=max_steps if max_steps is not None else -1,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            output_dir=f"{self.output_dir}/{agent_id}",
            optim="adamw_8bit",
            seed=3407,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="instruction", # 需要根据 converter 的输出格式调整
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logging.info(f"Training completed for {agent_id}")
        
        # 保存 LoRA 适配器
        self.model.save_pretrained(f"{self.output_dir}/{agent_id}/lora")
        self.tokenizer.save_pretrained(f"{self.output_dir}/{agent_id}/lora")
        
        return trainer_stats
