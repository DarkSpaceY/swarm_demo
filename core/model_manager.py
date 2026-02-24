import os
import torch
import logging
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

class InferenceEngine(ABC):
    """
    统一的推理引擎接口。
    """
    def __init__(self, model_name, cache_dir, local_files_only):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompt, **gen_config):
        pass

class UnslothEngine(InferenceEngine):
    """
    基于 Unsloth 的推理引擎。
    """
    def load(self, max_seq_length=2048, load_in_4bit=True, **kwargs):
        logging.info(f"Loading model {self.model_name} with Unsloth...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.model = FastLanguageModel.for_inference(self.model)
        return self.model, self.tokenizer

    def generate(self, prompt, **gen_config):
        # 实现具体的生成逻辑
        pass

class TransformersEngine(InferenceEngine):
    """
    基于原生 Transformers 的推理引擎（用于 Mac 或非 Unsloth 环境）。
    """
    def load(self, device="cpu", **kwargs):
        logging.info(f"Loading model {self.model_name} with Transformers on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=False,
        )
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.model.to(device)
        return self.model, self.tokenizer

    def generate(self, prompt, **gen_config):
        # 实现具体的生成逻辑
        pass

class VLLMEngine(InferenceEngine):
    """
    基于 vLLM 的推理引擎。
    """
    def load(self, **kwargs):
        logging.info(f"Initializing vLLM engine for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=False,
        )
        # vLLM 的实际初始化通常在单独的进程或通过其 Engine 类完成
        return None, self.tokenizer

    def generate(self, prompt, **gen_config):
        # vLLM 生成逻辑
        pass

def get_engine(engine_type, model_name, cache_dir, local_files_only):
    if engine_type == "vllm":
        return VLLMEngine(model_name, cache_dir, local_files_only)
    elif engine_type == "transformers":
        return TransformersEngine(model_name, cache_dir, local_files_only)
    else:
        return UnslothEngine(model_name, cache_dir, local_files_only)
