import torch
import time
import psutil
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class InferenceMetrics:
    tokens_generated: int
    total_time: float
    tokens_per_second: float
    memory_peak_mb: float
    memory_avg_mb: float
    first_token_latency: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BaselineInference:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        use_cache: bool = True
    ):
        self.model_name = model_name
        self.use_cache = use_cache
        
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"loading model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.float32 if self.device in ["cpu", "mps"] else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model.eval()
        self.model_size_mb = self._get_model_size()
        print(f"model loaded: {self.model_size_mb:.2f} mb")
        
    def _get_model_size(self) -> float:
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def _get_memory_usage(self) -> float:
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 2)
        elif self.device == "mps":
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        profile: bool = True
    ) -> Tuple[str, Optional[InferenceMetrics]]:
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        if not profile:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text, None
        
        memory_samples = []
        initial_memory = self._get_memory_usage()
        memory_samples.append(initial_memory)
        
        start_time = time.perf_counter()
        first_token_time = None
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True
            )
            
            if first_token_time is None:
                first_token_time = time.perf_counter() - start_time
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        peak_memory = self._get_memory_usage()
        memory_samples.append(peak_memory)
        
        generated_text = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        tokens_generated = len(outputs.sequences[0]) - len(inputs.input_ids[0])
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        metrics = InferenceMetrics(
            tokens_generated=tokens_generated,
            total_time=total_time,
            tokens_per_second=tokens_per_second,
            memory_peak_mb=peak_memory,
            memory_avg_mb=np.mean(memory_samples),
            first_token_latency=first_token_time
        )
        
        return generated_text, metrics
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        num_warmup: int = 2
    ) -> Dict:
        print(f"benchmarking {len(prompts)} prompts...")
        
        if num_warmup > 0:
            print(f"warmup: {num_warmup} iterations")
            for i in range(num_warmup):
                self.generate(prompts[0], max_new_tokens=50, profile=False)
        
        all_metrics = []
        
        for i, prompt in enumerate(prompts):
            print(f"prompt {i+1}/{len(prompts)}", end="\r")
            _, metrics = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                profile=True
            )
            all_metrics.append(metrics)
        
        print("\nbenchmark complete")
        
        results = {
            "model_name": self.model_name,
            "device": self.device,
            "model_size_mb": self.model_size_mb,
            "num_prompts": len(prompts),
            "metrics": {
                "avg_tokens_per_second": np.mean([m.tokens_per_second for m in all_metrics]),
                "std_tokens_per_second": np.std([m.tokens_per_second for m in all_metrics]),
                "avg_memory_peak_mb": np.mean([m.memory_peak_mb for m in all_metrics]),
                "avg_first_token_latency": np.mean([m.first_token_latency for m in all_metrics]),
                "total_tokens_generated": sum(m.tokens_generated for m in all_metrics),
                "total_time": sum(m.total_time for m in all_metrics)
            },
            "individual_runs": [m.to_dict() for m in all_metrics]
        }
        
        return results
    
    def save_results(self, results: Dict, filename: str = "baseline_results.json"):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"saved to {filename}")


def main():
    baseline = BaselineInference()
    
    test_prompts = [
        "explain quantum computing in simple terms:",
        "write a short poem about artificial intelligence:",
        "what are the benefits of machine learning?",
        "describe the process of photosynthesis:",
        "how does a neural network learn?"
    ]
    
    print("\n" + "="*50)
    print("single generation test")
    print("="*50)
    text, metrics = baseline.generate(test_prompts[0], max_new_tokens=100)
    print(f"\ngenerated:\n{text}")
    print(f"\nmetrics:")
    print(f"  tokens/sec: {metrics.tokens_per_second:.2f}")
    print(f"  memory peak: {metrics.memory_peak_mb:.2f} mb")
    print(f"  first token: {metrics.first_token_latency:.4f}s")
    
    print("\n" + "="*50)
    print("full benchmark")
    print("="*50)
    results = baseline.benchmark(test_prompts, max_new_tokens=100)
    
    print("\n" + "="*50)
    print("baseline results")
    print("="*50)
    print(f"model: {results['model_name']}")
    print(f"device: {results['device']}")
    print(f"model size: {results['model_size_mb']:.2f} mb")
    print(f"\nperformance:")
    print(f"  tokens/sec: {results['metrics']['avg_tokens_per_second']:.2f} ± {results['metrics']['std_tokens_per_second']:.2f}")
    print(f"  memory peak: {results['metrics']['avg_memory_peak_mb']:.2f} mb")
    print(f"  first token: {results['metrics']['avg_first_token_latency']:.4f}s")
    
    baseline.save_results(results)


if __name__ == "__main__":
    main()