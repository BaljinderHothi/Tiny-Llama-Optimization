import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List
import numpy as np
from baseline import BaselineInference, InferenceMetrics
import time


class QuantizedInference(BaselineInference):
    """post-training int8 quantization using pytorch dynamic quantization"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 device: str = "auto", quant_dtype: torch.dtype = torch.qint8):
        
        self.quant_dtype = quant_dtype
        
        if device == "auto":
            self.device = "cpu"
        else:
            self.device = device
            
        print(f"loading model for quantization on {self.device}...")
        
        self.model_name = model_name
        self.use_cache = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        print("quantizing model to int8...")
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=quant_dtype
        )
        
        self.model.to(self.device)
        self.model_size_mb = self._get_model_size()
        print(f"quantized model size: {self.model_size_mb:.2f} mb")


def main():
    print("="*50)
    print("post-training quantization (int8)")
    print("="*50)
    
    quantized = QuantizedInference()
    
    test_prompts = [
        "explain quantum computing in simple terms:",
        "write a short poem about artificial intelligence:",
        "what are the benefits of machine learning?"
    ]
    
    results = quantized.benchmark(test_prompts, max_new_tokens=100)
    
    print("\n" + "="*50)
    print("quantization results")
    print("="*50)
    print(f"model size: {results['model_size_mb']:.2f} mb")
    print(f"tokens/sec: {results['metrics']['avg_tokens_per_second']:.2f}")
    print(f"memory peak: {results['metrics']['avg_memory_peak_mb']:.2f} mb")
    
    quantized.save_results(results, "quantized_results.json")


if __name__ == "__main__":
    main()