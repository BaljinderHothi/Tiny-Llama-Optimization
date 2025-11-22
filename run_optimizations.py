import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

from baseline import BaselineInference
from optimizations.quantization import QuantizedInference
from optimizations.head_pruning import PrunedInference
from optimizations.mixed_precision import MixedPrecisionInference
from benchmark import BenchmarkSuite


def run_all_optimizations(test_prompts: list, max_new_tokens: int = 100):
    """run baseline and all optimizations"""
    results = {}
    
    print("\n" + "="*60)
    print("BASELINE")
    print("="*60)
    baseline = BaselineInference()
    baseline_results = baseline.benchmark(test_prompts, max_new_tokens=max_new_tokens)
    baseline_bench = BenchmarkSuite(baseline)
    baseline_quality = baseline_bench.run_full_benchmark()
    results['baseline'] = {
        'performance': baseline_results['metrics'],
        'quality': baseline_quality,
        'model_size_mb': baseline_results['model_size_mb']
    }
    baseline.save_results(baseline_results, 'results/baseline_results.json')
    
    print("\n" + "="*60)
    print("MIXED PRECISION")
    print("="*60)
    mixed = MixedPrecisionInference()
    mixed_results = mixed.benchmark(test_prompts, max_new_tokens=max_new_tokens)
    mixed_bench = BenchmarkSuite(mixed)
    mixed_quality = mixed_bench.run_full_benchmark()
    results['mixed_precision'] = {
        'performance': mixed_results['metrics'],
        'quality': mixed_quality,
        'model_size_mb': mixed_results['model_size_mb']
    }
    mixed.save_results(mixed_results, 'results/mixed_precision_results.json')
    
    print("\n" + "="*60)
    print("INT8 QUANTIZATION")
    print("="*60)
    quantized = QuantizedInference()
    quant_results = quantized.benchmark(test_prompts, max_new_tokens=max_new_tokens)
    quant_bench = BenchmarkSuite(quantized)
    quant_quality = quant_bench.run_full_benchmark()
    results['quantized'] = {
        'performance': quant_results['metrics'],
        'quality': quant_quality,
        'model_size_mb': quant_results['model_size_mb']
    }
    quantized.save_results(quant_results, 'results/quantized_results.json')
    
    print("\n" + "="*60)
    print("ATTENTION HEAD PRUNING")
    print("="*60)
    pruned = PrunedInference(prune_ratio=0.25)
    pruned_results = pruned.benchmark(test_prompts, max_new_tokens=max_new_tokens)
    pruned_bench = BenchmarkSuite(pruned)
    pruned_quality = pruned_bench.run_full_benchmark()
    results['pruned'] = {
        'performance': pruned_results['metrics'],
        'quality': pruned_quality,
        'model_size_mb': pruned_results['model_size_mb']
    }
    pruned.save_results(pruned_results, 'results/pruned_results.json')
    
    return results


def calculate_improvements(results):
    """calculate improvement percentages vs baseline"""
    baseline = results['baseline']
    improvements = {}
    
    for name, data in results.items():
        if name == 'baseline':
            continue
            
        improvements[name] = {
            'speed_improvement': (
                (data['performance']['avg_tokens_per_second'] - 
                 baseline['performance']['avg_tokens_per_second']) / 
                baseline['performance']['avg_tokens_per_second'] * 100
            ),
            'memory_reduction': (
                (baseline['performance']['avg_memory_peak_mb'] - 
                 data['performance']['avg_memory_peak_mb']) / 
                baseline['performance']['avg_memory_peak_mb'] * 100
            ),
            'model_size_reduction': (
                (baseline['model_size_mb'] - data['model_size_mb']) / 
                baseline['model_size_mb'] * 100
            ),
            'perplexity_change': (
                (data['quality']['perplexity'] - baseline['quality']['perplexity']) / 
                baseline['quality']['perplexity'] * 100
            )
        }
    
    return improvements


def visualize_results(results, improvements):
    """create visualization comparing all methods"""
    Path('results/visualizations').mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    methods = list(results.keys())
    
    tokens_per_sec = [results[m]['performance']['avg_tokens_per_second'] for m in methods]
    axes[0, 0].bar(methods, tokens_per_sec, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    axes[0, 0].set_title('tokens per second (higher is better)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('tokens/sec')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    memory = [results[m]['performance']['avg_memory_peak_mb'] for m in methods]
    axes[0, 1].bar(methods, memory, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    axes[0, 1].set_title('peak memory usage (lower is better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('memory (mb)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    model_sizes = [results[m]['model_size_mb'] for m in methods]
    axes[1, 0].bar(methods, model_sizes, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    axes[1, 0].set_title('model size (lower is better)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('size (mb)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    perplexities = [results[m]['quality']['perplexity'] for m in methods]
    axes[1, 1].bar(methods, perplexities, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    axes[1, 1].set_title('perplexity (lower is better)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('perplexity')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/comparison.png', dpi=300, bbox_inches='tight')
    print("saved visualization to results/visualizations/comparison.png")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(improvements))
    width = 0.2
    
    metrics = ['speed_improvement', 'memory_reduction', 'model_size_reduction']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    labels = ['speed up (%)', 'memory reduction (%)', 'size reduction (%)']
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        values = [improvements[m][metric] for m in improvements.keys()]
        ax.bar(x + i*width, values, width, label=label, color=color)
    
    ax.set_xlabel('optimization method')
    ax.set_ylabel('improvement (%)')
    ax.set_title('performance improvements vs baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(improvements.keys()), rotation=45)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/improvements.png', dpi=300, bbox_inches='tight')
    print("saved improvements chart to results/visualizations/improvements.png")


def print_summary(results, improvements):
    """print comprehensive summary table"""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    baseline = results['baseline']
    print("\nbaseline performance:")
    print(f"  tokens/sec: {baseline['performance']['avg_tokens_per_second']:.2f}")
    print(f"  memory peak: {baseline['performance']['avg_memory_peak_mb']:.2f} mb")
    print(f"  model size: {baseline['model_size_mb']:.2f} mb")
    print(f"  perplexity: {baseline['quality']['perplexity']:.2f}")
    
    print("\n" + "-"*80)
    print(f"{'method':<20} {'speed':<12} {'memory':<12} {'size':<12} {'quality':<12}")
    print("-"*80)
    
    for method, imp in improvements.items():
        speed = f"+{imp['speed_improvement']:.1f}%" if imp['speed_improvement'] > 0 else f"{imp['speed_improvement']:.1f}%"
        memory = f"-{imp['memory_reduction']:.1f}%" if imp['memory_reduction'] > 0 else f"+{abs(imp['memory_reduction']):.1f}%"
        size = f"-{imp['model_size_reduction']:.1f}%" if imp['model_size_reduction'] > 0 else f"+{abs(imp['model_size_reduction']):.1f}%"
        quality = f"{imp['perplexity_change']:+.1f}%"
        
        print(f"{method:<20} {speed:<12} {memory:<12} {size:<12} {quality:<12}")
    
    print("-"*80)
    
    print("\nbest optimizations:")
    best_speed = max(improvements.items(), key=lambda x: x[1]['speed_improvement'])
    best_memory = max(improvements.items(), key=lambda x: x[1]['memory_reduction'])
    best_size = max(improvements.items(), key=lambda x: x[1]['model_size_reduction'])
    
    print(f"  fastest: {best_speed[0]} (+{best_speed[1]['speed_improvement']:.1f}%)")
    print(f"  lowest memory: {best_memory[0]} (-{best_memory[1]['memory_reduction']:.1f}%)")
    print(f"  smallest size: {best_size[0]} (-{best_size[1]['model_size_reduction']:.1f}%)")


def main():
    Path('results').mkdir(exist_ok=True)
    
    test_prompts = [
        "explain quantum computing in simple terms:",
        "write a short poem about artificial intelligence:",
        "what are the benefits of machine learning?",
        "describe the process of photosynthesis:",
        "how does a neural network learn?"
    ]
    
    print("="*60)
    print("TINYLLAMA OPTIMIZATION PIPELINE")
    print("="*60)
    print(f"running {len(test_prompts)} test prompts")
    print("this will take several minutes...")
    
    results = run_all_optimizations(test_prompts, max_new_tokens=100)
    
    improvements = calculate_improvements(results)
    
    consolidated = {
        'results': results,
        'improvements': improvements
    }
    with open('results/consolidated_results.json', 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    visualize_results(results, improvements)
    
    print_summary(results, improvements)
    
    print("\n" + "="*60)
    print("optimization pipeline complete!")
    print("check results/ directory for detailed outputs")
    print("="*60)


if __name__ == "__main__":
    main()