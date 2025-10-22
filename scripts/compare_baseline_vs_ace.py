#!/usr/bin/env python3
"""
Compare baseline performance vs ACE adaptation results.

This script runs both baseline (no adaptation) and ACE adaptation
on the same dataset samples to measure improvement.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ace import Generator, Reflector, Curator, OfflineAdapter, Playbook
from ace.llm_providers import LiteLLMClient
from benchmarks import BenchmarkTaskManager, BenchmarkSample

# Suppress LiteLLM debug messages
import litellm
litellm.suppress_debug_info = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="Benchmark to evaluate")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--epochs", type=int, default=2, help="ACE adaptation epochs")
    parser.add_argument("--output", default="comparison_results", help="Output directory")
    return parser.parse_args()


def run_baseline(samples: List[BenchmarkSample], model: str, benchmark_env) -> Dict:
    """Run baseline evaluation without ACE adaptation."""
    print("🔬 Running BASELINE evaluation (no adaptation)...")

    client = LiteLLMClient(model=model, temperature=0.0, max_tokens=2048)
    generator = Generator(client)
    playbook = Playbook()  # Empty playbook

    results = []
    for i, sample in enumerate(samples):
        print(f"  Processing sample {i+1}/{len(samples)}")

        output = generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=playbook
        )

        env_result = benchmark_env.evaluate(sample, output)
        results.append({
            "sample_id": sample.sample_id,
            "metrics": env_result.metrics,
            "prediction": output.final_answer,
            "ground_truth": sample.ground_truth
        })

    return {
        "type": "baseline",
        "model": model,
        "samples": len(samples),
        "results": results,
        "summary": compute_summary(results)
    }


def run_ace_adaptation(samples: List[BenchmarkSample], model: str, benchmark_env, epochs: int) -> Dict:
    """Run ACE adaptation evaluation."""
    print(f"🧠 Running ACE ADAPTATION evaluation ({epochs} epochs)...")

    client = LiteLLMClient(model=model, temperature=0.1, max_tokens=2048)
    generator = Generator(client)
    reflector = Reflector(client)
    curator = Curator(client)

    adapter = OfflineAdapter(
        playbook=Playbook(),
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=2
    )

    adaptation_results = adapter.run(samples, benchmark_env, epochs=epochs)

    results = []
    for step in adaptation_results:
        results.append({
            "sample_id": step.sample.sample_id,
            "metrics": step.environment_result.metrics,
            "prediction": step.generator_output.final_answer,
            "ground_truth": step.sample.ground_truth,
            "used_bullets": getattr(step.generator_output, 'bullet_ids', [])
        })

    return {
        "type": "ace_adaptation",
        "model": model,
        "epochs": epochs,
        "samples": len(samples),
        "results": results,
        "summary": compute_summary(results),
        "final_playbook": adapter.playbook.as_prompt(),
        "playbook_bullets": len(adapter.playbook.bullets())
    }


def compute_summary(results: List[Dict]) -> Dict:
    """Compute summary statistics from results."""
    if not results:
        return {}

    metrics_summary = {}
    all_metrics = results[0]["metrics"].keys()

    for metric in all_metrics:
        values = [r["metrics"][metric] for r in results]
        metrics_summary[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "values": values
        }

    return metrics_summary


def print_comparison(baseline_results: Dict, ace_results: Dict):
    """Print detailed comparison of results."""
    print("\n" + "="*80)
    print("📊 BASELINE vs ACE ADAPTATION COMPARISON")
    print("="*80)

    print(f"\nModel: {baseline_results['model']}")
    print(f"Samples: {baseline_results['samples']}")
    if 'epochs' in ace_results:
        print(f"ACE Epochs: {ace_results['epochs']}")

    print("\n" + "-"*50)
    print("PERFORMANCE METRICS")
    print("-"*50)

    baseline_summary = baseline_results['summary']
    ace_summary = ace_results['summary']

    for metric in baseline_summary.keys():
        baseline_mean = baseline_summary[metric]['mean']
        ace_mean = ace_summary[metric]['mean']
        improvement = ace_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean * 100) if baseline_mean > 0 else 0

        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Baseline:     {baseline_mean:.3f}")
        print(f"  ACE:          {ace_mean:.3f}")
        print(f"  Improvement:  {improvement:+.3f} ({improvement_pct:+.1f}%)")

    # Show playbook info
    if 'playbook_bullets' in ace_results:
        print(f"\n📚 LEARNED STRATEGIES:")
        print(f"  Playbook bullets: {ace_results['playbook_bullets']}")
        if ace_results.get('final_playbook'):
            print(f"  Sample strategy: {ace_results['final_playbook'][:200]}...")


def save_results(baseline_results: Dict, ace_results: Dict, output_dir: str):
    """Save detailed results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    benchmark = baseline_results.get('benchmark', 'unknown')
    model = baseline_results['model']

    # Save comparison results
    comparison = {
        "timestamp": timestamp,
        "benchmark": benchmark,
        "model": model,
        "baseline": baseline_results,
        "ace_adaptation": ace_results,
        "improvements": {}
    }

    # Calculate improvements
    baseline_summary = baseline_results['summary']
    ace_summary = ace_results['summary']

    for metric in baseline_summary.keys():
        baseline_mean = baseline_summary[metric]['mean']
        ace_mean = ace_summary[metric]['mean']
        improvement = ace_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean * 100) if baseline_mean > 0 else 0

        comparison["improvements"][metric] = {
            "absolute": improvement,
            "percentage": improvement_pct
        }

    # Save to file
    output_file = output_path / f"comparison_{benchmark}_{model}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")
    return output_file


def main():
    args = parse_args()

    # Load benchmark
    manager = BenchmarkTaskManager()
    if args.benchmark not in manager.list_benchmarks():
        print(f"Error: Unknown benchmark '{args.benchmark}'")
        sys.exit(1)

    # Load samples
    print(f"Loading {args.samples} samples from {args.benchmark}...")
    raw_data = list(manager.load_benchmark_data(args.benchmark))
    raw_data = raw_data[:args.samples]  # Limit samples

    # Convert to BenchmarkSample format (simplified)
    samples = []
    for i, data in enumerate(raw_data):
        if args.benchmark == "finer_ord":
            sample = BenchmarkSample(
                sample_id=data.get('sample_id', f"{args.benchmark}_{i}"),
                benchmark_name=args.benchmark,
                question=data['question'],
                ground_truth=data['ground_truth'],
                metadata=data.get('metadata', {})
            )
        else:
            # Generic handling
            sample = BenchmarkSample(
                sample_id=f"{args.benchmark}_{i}",
                benchmark_name=args.benchmark,
                question=data.get('question', ''),
                ground_truth=data.get('answer', data.get('ground_truth', '')),
                metadata=data
            )
        samples.append(sample)

    print(f"Loaded {len(samples)} samples")

    # Get benchmark environment
    benchmark_env = manager.get_benchmark(args.benchmark)

    # Run both evaluations
    baseline_results = run_baseline(samples, args.model, benchmark_env)
    ace_results = run_ace_adaptation(samples, args.model, benchmark_env, args.epochs)

    # Add benchmark name
    baseline_results['benchmark'] = args.benchmark
    ace_results['benchmark'] = args.benchmark

    # Print and save comparison
    print_comparison(baseline_results, ace_results)
    save_results(baseline_results, ace_results, args.output)


if __name__ == "__main__":
    main()