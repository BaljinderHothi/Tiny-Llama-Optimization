import torch
import numpy as np
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer
import time


class BenchmarkSuite:
    """comprehensive benchmark suite for model quality evaluation"""
    
    def __init__(self, model_inference):
        self.model = model_inference
        self.tokenizer = model_inference.tokenizer
        
    def perplexity_benchmark(self, texts: List[str], max_length: int = 512) -> float:
        """compute perplexity on test texts - lower is better"""
        total_loss = 0
        total_tokens = 0
        
        self.model.model.eval()
        
        with torch.no_grad():
            for text in texts:
                encodings = self.tokenizer(text, return_tensors='pt', 
                                          max_length=max_length, truncation=True)
                input_ids = encodings.input_ids.to(self.model.device)
                
                outputs = self.model.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return perplexity
    
    def generation_quality_benchmark(self, prompts: List[str], 
                                    max_new_tokens: int = 50) -> Dict:
        """evaluate generation quality with multiple metrics"""
        results = {
            'outputs': [],
            'avg_length': 0,
            'unique_tokens_ratio': []
        }
        
        for prompt in prompts:
            output, _ = self.model.generate(prompt, max_new_tokens=max_new_tokens, 
                                           profile=False)
            results['outputs'].append(output)
            
            # calculate unique token ratio (diversity metric)
            tokens = self.tokenizer.encode(output)
            unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
            results['unique_tokens_ratio'].append(unique_ratio)
        
        results['avg_length'] = np.mean([len(self.tokenizer.encode(o)) 
                                        for o in results['outputs']])
        results['avg_unique_ratio'] = np.mean(results['unique_tokens_ratio'])
        
        return results
    
    def get_test_prompts(self) -> List[str]:
        """get diverse test prompts for benchmarking"""
        return [
            "explain how neural networks work:",
            "write a story about a robot learning to paint:",
            "what is the difference between supervised and unsupervised learning?",
            "describe the process of training a language model:",
            "explain gradient descent in simple terms:",
            "what are transformers in machine learning?",
            "how does attention mechanism work?",
            "write a haiku about artificial intelligence:",
            "what is the purpose of backpropagation?",
            "explain overfitting and how to prevent it:"
        ]
    
    def get_perplexity_texts(self) -> List[str]:
        """get texts for perplexity evaluation"""
        texts = [
            "machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "neural networks are composed of layers of interconnected nodes that process information.",
            "deep learning uses multiple layers to progressively extract higher-level features from raw input.",
            "transformers revolutionized natural language processing by introducing self-attention mechanisms.",
            "training large language models requires substantial computational resources and carefully curated datasets."
        ]
        return texts
    
    def run_full_benchmark(self) -> Dict:
        """run complete benchmark suite"""
        print("running full benchmark suite...")
        
        # perplexity test
        print("computing perplexity...")
        ppl_texts = self.get_perplexity_texts()
        perplexity = self.perplexity_benchmark(ppl_texts)
        
        # generation quality
        print("evaluating generation quality...")
        test_prompts = self.get_test_prompts()
        quality = self.generation_quality_benchmark(test_prompts[:5])
        
        # performance benchmark
        print("measuring performance...")
        perf_results = self.model.benchmark(test_prompts[:5], max_new_tokens=100)
        
        results = {
            'perplexity': perplexity,
            'avg_unique_token_ratio': quality['avg_unique_ratio'],
            'avg_output_length': quality['avg_length'],
            'performance': perf_results['metrics']
        }
        
        return results


def main():
    from baseline import BaselineInference
    
    print("="*50)
    print("comprehensive benchmark suite")
    print("="*50)
    
    model = BaselineInference()
    bench = BenchmarkSuite(model)
    
    results = bench.run_full_benchmark()
    
    print("\n" + "="*50)
    print("benchmark results")
    print("="*50)
    print(f"perplexity: {results['perplexity']:.2f}")
    print(f"unique token ratio: {results['avg_unique_token_ratio']:.3f}")
    print(f"avg output length: {results['avg_output_length']:.1f} tokens")
    print(f"tokens/sec: {results['performance']['avg_tokens_per_second']:.2f}")


if __name__ == "__main__":
    main()