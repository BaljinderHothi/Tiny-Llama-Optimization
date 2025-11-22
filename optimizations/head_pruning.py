import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List, Dict
import numpy as np
from baseline import BaselineInference, InferenceMetrics
import time


class AttentionHeadPruner:
    """prunes less important attention heads based on importance scores"""
    
    def __init__(self, model, prune_ratio: float = 0.25):
        self.model = model
        self.prune_ratio = prune_ratio
        self.head_importance = {}
        
    def compute_head_importance(self) -> Dict[str, torch.Tensor]:
        """compute importance score for each attention head"""
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if 'self_attn' in name and hasattr(module, 'o_proj'):
                weight = module.o_proj.weight.data
                
                if hasattr(module, 'num_heads'):
                    num_heads = module.num_heads
                    head_dim = weight.shape[1] // num_heads
                    
                    scores = []
                    for i in range(num_heads):
                        start_idx = i * head_dim
                        end_idx = (i + 1) * head_dim
                        head_weight = weight[:, start_idx:end_idx]
                        score = torch.norm(head_weight, p=2).item()
                        scores.append(score)
                    
                    importance_scores[name] = torch.tensor(scores)
        
        return importance_scores
    
    def prune_heads(self):
        """prune least important heads"""
        self.head_importance = self.compute_head_importance()
        
        total_heads = sum(len(scores) for scores in self.head_importance.values())
        heads_to_prune = int(total_heads * self.prune_ratio)
        
        print(f"pruning {heads_to_prune}/{total_heads} heads ({self.prune_ratio*100:.1f}%)")
        
        all_scores = []
        for layer_name, scores in self.head_importance.items():
            for head_idx, score in enumerate(scores):
                all_scores.append((score, layer_name, head_idx))
        
        all_scores.sort(key=lambda x: x[0])
        heads_to_remove = all_scores[:heads_to_prune]
        
        prune_dict = {}
        for _, layer_name, head_idx in heads_to_remove:
            if layer_name not in prune_dict:
                prune_dict[layer_name] = []
            prune_dict[layer_name].append(head_idx)
        
        for name, module in self.model.named_modules():
            if name in prune_dict and hasattr(module, 'o_proj'):
                heads_to_zero = prune_dict[name]
                num_heads = module.num_heads
                head_dim = module.o_proj.weight.shape[1] // num_heads
                
                with torch.no_grad():
                    for head_idx in heads_to_zero:
                        start_idx = head_idx * head_dim
                        end_idx = (head_idx + 1) * head_dim
                        module.o_proj.weight[:, start_idx:end_idx] = 0
                        
                        if hasattr(module, 'q_proj'):
                            module.q_proj.weight[start_idx:end_idx, :] = 0
                        if hasattr(module, 'k_proj'):
                            module.k_proj.weight[start_idx:end_idx, :] = 0
                        if hasattr(module, 'v_proj'):
                            module.v_proj.weight[start_idx:end_idx, :] = 0
        
        print(f"pruned {heads_to_prune} attention heads")
        return prune_dict


class PrunedInference(BaselineInference):
    """inference with pruned attention heads"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 device: str = "auto", prune_ratio: float = 0.25):
        super().__init__(model_name, device, use_cache=True)
        
        self.prune_ratio = prune_ratio
        self.pruner = AttentionHeadPruner(self.model, prune_ratio=prune_ratio)
        self.pruned_heads = self.pruner.prune_heads()


def main():
    print("="*50)
    print("attention head pruning")
    print("="*50)
    
    pruned = PrunedInference(prune_ratio=0.25)
    
    test_prompts = [
        "explain quantum computing in simple terms:",
        "write a short poem about artificial intelligence:",
        "what are the benefits of machine learning?"
    ]
    
    results = pruned.benchmark(test_prompts, max_new_tokens=100)
    
    print("\n" + "="*50)
    print("head pruning results")
    print("="*50)
    print(f"prune ratio: {pruned.prune_ratio*100:.1f}%")
    print(f"tokens/sec: {results['metrics']['avg_tokens_per_second']:.2f}")
    print(f"memory peak: {results['metrics']['avg_memory_peak_mb']:.2f} mb")
    
    pruned.save_results(results, "pruned_results.json")


if __name__ == "__main__":
    main()