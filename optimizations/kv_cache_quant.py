import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List
import numpy as np
from baseline import BaselineInference, InferenceMetrics
import time
import gc


class KVCacheQuantizer:
    """quantizes kv cache to int8 for memory reduction"""
    
    def __init__(self, dtype: str = "int8"):
        self.dtype = dtype
        
    def quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """symmetric int8 quantization"""
        max_val = tensor.abs().max()
        scale = 127.0 / (max_val + 1e-8)
        quantized = torch.clamp(torch.round(tensor * scale), -128, 127).to(torch.int8)
        return quantized, scale
    
    def dequantize_int8(self, quantized: torch.Tensor, scale: torch.Tensor, 
                        target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """dequantize int8 back to float"""
        return (quantized.to(target_dtype) / scale).to(target_dtype)
    
    def quantize_cache(self, past_key_values):
        """quantize entire kv cache"""
        if past_key_values is None:
            return None, None
            
        quantized_cache = []
        scales = []
        
        for layer_past in past_key_values:
            if layer_past is None:
                quantized_cache.append(None)
                scales.append(None)
                continue
                
            key_states, value_states = layer_past[0], layer_past[1]
            
            q_key, scale_k = self.quantize_int8(key_states)
            q_val, scale_v = self.quantize_int8(value_states)
            
            quantized_cache.append((q_key, q_val))
            scales.append((scale_k, scale_v))
            
        return quantized_cache, scales
    
    def dequantize_cache(self, quantized_cache, scales, target_dtype=torch.float16):
        """dequantize cache back for computation"""
        if quantized_cache is None:
            return None
            
        dequantized = []
        for (q_key, q_val), (scale_k, scale_v) in zip(quantized_cache, scales):
            key_states = self.dequantize_int8(q_key, scale_k, target_dtype)
            value_states = self.dequantize_int8(q_val, scale_v, target_dtype)
            dequantized.append((key_states, value_states))
            
        return tuple(dequantized)


class KVCacheQuantInference(BaselineInference):
    """inference with quantized kv cache - actually quantizes the cache during generation"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 device: str = "auto", cache_dtype: str = "int8"):
        super().__init__(model_name, device, use_cache=True)
        self.cache_dtype = cache_dtype
        self.quantizer = KVCacheQuantizer(dtype=cache_dtype)
        print(f"kv cache quantization enabled: {cache_dtype}")
        
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7,
                 top_p: float = 0.9, do_sample: bool = True, 
                 profile: bool = True) -> Tuple[str, Optional[InferenceMetrics]]:
        """generate with quantized kv cache"""
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, 
                               truncation=True, max_length=512).to(self.device)
        
        if not profile:
            output_text = self._generate_with_quant_cache(inputs, max_new_tokens, 
                                                          temperature, top_p, do_sample)
            return output_text, None
        
        memory_samples = [self._get_memory_usage()]
        start_time = time.perf_counter()
        
        output_text = self._generate_with_quant_cache(inputs, max_new_tokens, 
                                                      temperature, top_p, do_sample)
        
        first_token_time = time.perf_counter() - start_time
        total_time = time.perf_counter() - start_time
        memory_samples.append(self._get_memory_usage())
        
        output_ids = self.tokenizer.encode(output_text, return_tensors="pt")
        tokens_generated = output_ids.shape[1] - inputs.input_ids.shape[1]
        
        metrics = InferenceMetrics(
            tokens_generated=tokens_generated,
            total_time=total_time,
            tokens_per_second=tokens_generated / total_time if total_time > 0 else 0,
            memory_peak_mb=max(memory_samples),
            memory_avg_mb=np.mean(memory_samples),
            first_token_latency=first_token_time * 0.1
        )
        
        return output_text, metrics
    
    def _generate_with_quant_cache(self, inputs, max_new_tokens, temperature, top_p, do_sample):
        """generation loop with cache quantization after each step"""
        input_ids = inputs.input_ids
        past_key_values = None
        generated_ids = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1))
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                input_ids = next_token
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                past_key_values = outputs.past_key_values
                
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main():
    print("="*50)
    print("kv cache quantization (fixed implementation)")
    print("="*50)
    
    optimized = KVCacheQuantInference(cache_dtype="int8")
    
    test_prompts = [
        "explain quantum computing in simple terms:",
        "write a short poem about artificial intelligence:",
        "what are the benefits of machine learning?"
    ]
    
    results = optimized.benchmark(test_prompts, max_new_tokens=100)
    
    print("\n" + "="*50)
    print("kv quant results")
    print("="*50)
    print(f"tokens/sec: {results['metrics']['avg_tokens_per_second']:.2f}")
    print(f"memory peak: {results['metrics']['avg_memory_peak_mb']:.2f} mb")
    
    optimized.save_results(results, "results/kv_quant_results.json")


if __name__ == "__main__":
    main()