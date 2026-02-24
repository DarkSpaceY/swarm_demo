import logging
import os
import torch
import builtins
from accelerate.utils.dataclasses import FP8BackendType
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTTrainer, DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from config_loader import get_config

CONFIG = get_config()

logging.basicConfig(level=getattr(logging, str(CONFIG["logging"]["level"]).upper(), logging.DEBUG))

def ensure_qwen2_temp_qa(model):
    for module in model.modules():
        if module.__class__.__name__ == "Qwen2Attention":
            if not hasattr(module, "temp_QA"):
                module.temp_QA = None
            if not hasattr(module, "paged_attention_K"):
                module.paged_attention_K = None
            if not hasattr(module, "paged_attention_V"):
                module.paged_attention_V = None

def resolve_local_files_only(model_name, cache_dir, local_files_only):
    if local_files_only:
        return True
    if os.path.isdir(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(
            model_name,
            CONFIG["files"]["hf_config"],
            cache_dir=cache_dir,
        )
        return cached is not None
    except Exception:
        return False

class SwarmTrainer:
    def __init__(self, model_name=None, output_dir=None, max_seq_length=None, load_in_4bit=None, cache_dir=None, local_files_only=None):
        run_defaults = CONFIG["run_defaults"]
        training_defaults = CONFIG["training"]
        self.model_name = model_name if model_name is not None else run_defaults["model_name"]
        self.output_dir = output_dir if output_dir is not None else training_defaults["output_dir"]
        self.max_seq_length = max_seq_length if max_seq_length is not None else run_defaults["max_seq_length"]
        load_in_4bit = load_in_4bit if load_in_4bit is not None else run_defaults["load_in_4bit"]
        local_files_only = local_files_only if local_files_only is not None else run_defaults["local_files_only"]
        
        logging.info(f"Initializing SwarmTrainer with model: {self.model_name}")

        resolved_local_only = resolve_local_files_only(self.model_name, cache_dir, local_files_only)
        
        # 加载 Unsloth 模型和 Tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir,
            local_files_only=resolved_local_only,
        )
        ensure_qwen2_temp_qa(self.model)
        
        # 配置 LoRA
        lora_cfg = training_defaults["lora"]
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_cfg["r"],
            target_modules=lora_cfg["target_modules"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
            random_state=lora_cfg["random_state"],
            use_rslora=lora_cfg["use_rslora"],
            loftq_config=lora_cfg["loftq_config"],
        )

    def train_agent(
        self,
        agent_id,
        data_path,
        epochs=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        learning_rate=None,
        logging_steps=None,
        save_strategy=None,
        save_total_limit=None,
        max_steps=None,
        resume_from_checkpoint=None,
        judge_oversample=None,
    ):
        """
        为指定 Agent 训练 LoRA。
        
        Args:
            agent_id: str, Agent ID (例如 "Agent_0")
            data_path: str, 该 Agent 的训练数据 JSONL 路径
            epochs: int, 训练轮数
        """
        training_defaults = CONFIG["training"]
        logging.info(f"Starting training for {agent_id} using data from {data_path}")
        builtins.FP8BackendType = FP8BackendType
        try:
            dataset = load_dataset(training_defaults["dataset_format"], data_files=data_path, split=training_defaults["dataset_split"])
            dataset = load_dataset(training_defaults["dataset_format"], data_files=data_path, split=training_defaults["dataset_split"])
        except Exception as e:
            logging.warning(f"No data found for {agent_id}, skipping training. Error: {e}")
            return
        if dataset is None or len(dataset) == 0:
            logging.warning(f"No samples for {agent_id}, skipping training.")
            return
        agents_cfg = CONFIG["agents"]
        if agent_id == agents_cfg["judge_id"] and judge_oversample and judge_oversample > 1:
            dataset = concatenate_datasets([dataset] * int(judge_oversample))

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size if batch_size is not None else training_defaults["batch_size"],
            gradient_accumulation_steps=gradient_accumulation_steps if gradient_accumulation_steps is not None else training_defaults["grad_accum"],
            warmup_steps=training_defaults["warmup_steps"],
            max_steps=max_steps if max_steps is not None else -1,
            num_train_epochs=epochs if epochs is not None else training_defaults["epochs"],
            learning_rate=learning_rate if learning_rate is not None else training_defaults["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps if logging_steps is not None else training_defaults["logging_steps"],
            output_dir=f"{self.output_dir}/{agent_id}",
            optim=training_defaults["optim"],
            seed=training_defaults["seed"],
            save_strategy=save_strategy if save_strategy is not None else training_defaults["save_strategy"],
            save_total_limit=save_total_limit if save_total_limit is not None else training_defaults["save_total_limit"],
        )

        formatted_records = []
        has_template = hasattr(self.tokenizer, "apply_chat_template")
        data_keys = CONFIG["data_keys"]
        for sample in dataset:
            messages = sample.get(data_keys["messages"])
            if not isinstance(messages, list):
                text = training_defaults["empty_text"]
            elif has_template:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                chunks = []
                role_format = CONFIG["generation"]["fallback_role_format"]
                joiner = training_defaults["prompt_joiner"]
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role and content is not None:
                        chunks.append(role_format.format(role=role, content=content))
                text = joiner.join(chunks)
            formatted_records.append({data_keys["text"]: text})
        dataset = Dataset.from_list(formatted_records)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field=data_keys["text"],
            max_seq_length=self.max_seq_length,
            packing=False,
            args=training_args,
        )

        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logging.info(f"Training completed for {agent_id}")
        
        # 保存 LoRA 适配器
        lora_dir = training_defaults["lora_dir"]
        self.model.save_pretrained(f"{self.output_dir}/{agent_id}/{lora_dir}")
        self.tokenizer.save_pretrained(f"{self.output_dir}/{agent_id}/{lora_dir}")
        
        return trainer_stats

    def train_preference_agent(
        self,
        agent_id,
        data_path,
        algorithm=None,
        beta=None,
        sft_weight=None,
        epochs=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        learning_rate=None,
        logging_steps=None,
        save_strategy=None,
        save_total_limit=None,
        max_steps=None,
        resume_from_checkpoint=None,
    ):
        training_defaults = CONFIG["training"]
        logging.info(f"Starting preference training for {agent_id} using data from {data_path}")
        builtins.FP8BackendType = FP8BackendType
        try:
            dataset = load_dataset(training_defaults["dataset_format"], data_files=data_path, split=training_defaults["dataset_split"])
        except Exception as e:
            logging.warning(f"No preference data found for {agent_id}, skipping training. Error: {e}")
            return
        if dataset is None or len(dataset) == 0:
            logging.warning(f"No preference samples for {agent_id}, skipping training.")
            return

        formatted_records = []
        has_template = hasattr(self.tokenizer, "apply_chat_template")
        data_keys = CONFIG["data_keys"]
        for sample in dataset:
            messages = sample.get(data_keys["messages"])
            prompt = training_defaults["empty_text"]
            if isinstance(messages, list):
                if has_template:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    chunks = []
                    role_format = CONFIG["generation"]["fallback_role_format"]
                    joiner = training_defaults["prompt_joiner"]
                    suffix = CONFIG["generation"]["fallback_suffix"]
                    for msg in messages:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role and content is not None:
                            chunks.append(role_format.format(role=role, content=content))
                    prompt = joiner.join(chunks) + suffix
            chosen = sample.get(data_keys["chosen"], training_defaults["empty_text"])
            rejected = sample.get(data_keys["rejected"], training_defaults["empty_text"])
            if prompt and chosen and rejected:
                formatted_records.append({
                    data_keys["prompt"]: prompt,
                    data_keys["chosen"]: chosen,
                    data_keys["rejected"]: rejected,
                })
        if not formatted_records:
            logging.warning(f"No valid preference samples for {agent_id}, skipping training.")
            return
        dataset = Dataset.from_list(formatted_records)

        loss_type = [training_defaults["loss_type_dpo"]]
        loss_weights = [training_defaults["loss_weight_dpo"]]
        if algorithm and algorithm.lower() == training_defaults["algorithm_rpo"]:
            loss_type = [training_defaults["loss_type_dpo"], training_defaults["loss_type_sft"]]
            loss_weights = [training_defaults["loss_weight_dpo"], sft_weight if sft_weight is not None else training_defaults["sft_weight"]]

        training_args = DPOConfig(
            per_device_train_batch_size=batch_size if batch_size is not None else training_defaults["batch_size"],
            gradient_accumulation_steps=gradient_accumulation_steps if gradient_accumulation_steps is not None else training_defaults["grad_accum"],
            warmup_steps=training_defaults["warmup_steps"],
            max_steps=max_steps if max_steps is not None else -1,
            num_train_epochs=epochs if epochs is not None else training_defaults["epochs"],
            learning_rate=learning_rate if learning_rate is not None else training_defaults["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps if logging_steps is not None else training_defaults["logging_steps"],
            output_dir=f"{self.output_dir}/{agent_id}",
            optim=training_defaults["optim"],
            seed=training_defaults["seed"],
            save_strategy=save_strategy if save_strategy is not None else training_defaults["save_strategy"],
            save_total_limit=save_total_limit if save_total_limit is not None else training_defaults["save_total_limit"],
            beta=beta if beta is not None else training_defaults["beta"],
            loss_type=loss_type,
            loss_weights=loss_weights,
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
        )
        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logging.info(f"Preference training completed for {agent_id}")
        lora_dir = training_defaults["lora_dir"]
        self.model.save_pretrained(f"{self.output_dir}/{agent_id}/{lora_dir}")
        self.tokenizer.save_pretrained(f"{self.output_dir}/{agent_id}/{lora_dir}")
        return trainer_stats
