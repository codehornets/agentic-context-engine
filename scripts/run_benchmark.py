#!/usr/bin/env python3
"""
Run ACE benchmarks with comprehensive evaluation and reporting.

This script provides a command-line interface for running benchmarks
with the ACE framework, supporting multiple benchmark types and
configuration options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ace import (
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    Playbook,
)
from ace.llm_providers import LiteLLMClient
from benchmarks import BenchmarkTaskManager, BenchmarkSample

# Suppress LiteLLM debug messages
import litellm
litellm.suppress_debug_info = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Benchmark selection
    parser.add_argument(
        "benchmark",
        help="Benchmark name to run (finer_ord, xbrl_math, appworld, or 'list' to show available)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use for evaluation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )

    # Data configuration
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples to evaluate (default: all)"
    )

    # ACE configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of offline adaptation epochs (default: 1)"
    )
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=3,
        help="Maximum refinement rounds per sample (default: 3)"
    )
    parser.add_argument(
        "--skip-adaptation",
        action="store_true",
        help="Skip ACE adaptation and run direct evaluation"
    )

    # Output configuration
    parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save detailed per-sample results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    # Cache configuration
    parser.add_argument(
        "--cache-dir",
        help="Override cache directory for benchmark data"
    )

    return parser.parse_args()


def list_available_benchmarks() -> None:
    """List all available benchmarks."""
    manager = BenchmarkTaskManager()
    benchmarks = manager.list_benchmarks()

    print("Available benchmarks:")
    for name in benchmarks:
        try:
            config = manager.get_config(name)
            print(f"  {name} - {config.metadata.get('description', 'No description')}")
        except Exception as e:
            print(f"  {name} - (Error loading config: {e})")


def create_llm_client(args: argparse.Namespace) -> LiteLLMClient:
    """Create LLM client with specified configuration."""
    return LiteLLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=120,
    )


def load_benchmark_data(args: argparse.Namespace, manager: BenchmarkTaskManager) -> List[BenchmarkSample]:
    """Load and convert benchmark data to BenchmarkSample format."""
    if not args.quiet:
        print(f"Loading {args.benchmark} data (split: {args.split})...")

    # Load raw data
    try:
        raw_data = list(manager.load_benchmark_data(args.benchmark))
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        sys.exit(1)

    # Apply limit if specified
    if args.limit:
        raw_data = raw_data[:args.limit]

    if not args.quiet:
        print(f"Loaded {len(raw_data)} samples")

    # Convert to BenchmarkSample format
    samples = []

    for i, data in enumerate(raw_data):
        if args.benchmark == "appworld":
            # AppWorld has special handling
            sample = BenchmarkSample(
                sample_id=f"{args.benchmark}_{i:04d}",
                benchmark_name=args.benchmark,
                question=data["instruction"],
                context=f"Available APIs: {data['api_docs']}",
                ground_truth="Task completion successful",
                metadata=data
            )
        elif args.benchmark == "finer_ord":
            # FiNER now comes pre-processed from the loader
            sample = BenchmarkSample(
                sample_id=data.get('sample_id', f"{args.benchmark}_{i:04d}"),
                benchmark_name=args.benchmark,
                question=data['question'],
                ground_truth=data['ground_truth'],
                metadata=data.get('metadata', {})
            )
        elif args.benchmark == "xbrl_math":
            # XBRL-Math handling
            sample = BenchmarkSample(
                sample_id=f"{args.benchmark}_{i:04d}",
                benchmark_name=args.benchmark,
                question=data.get('question', ''),
                context=data.get('context', ''),
                ground_truth=str(data.get('answer', '')),
                metadata=data
            )
        else:
            # Generic handling - check if already processed
            if 'sample_id' in data and 'question' in data:
                # Already processed by a processor
                sample = BenchmarkSample(
                    sample_id=data['sample_id'],
                    benchmark_name=args.benchmark,
                    question=data['question'],
                    ground_truth=data.get('ground_truth', ''),
                    context=data.get('context', ''),
                    metadata=data.get('metadata', {})
                )
            else:
                # Raw data - use generic handling
                sample = BenchmarkSample(
                    sample_id=f"{args.benchmark}_{i:04d}",
                    benchmark_name=args.benchmark,
                    question=str(data.get('question', data.get('input', ''))),
                    ground_truth=str(data.get('answer', data.get('output', ''))),
                    metadata=data
                )

        samples.append(sample)

    return samples


def run_evaluation(
    args: argparse.Namespace,
    samples: List[BenchmarkSample],
    manager: BenchmarkTaskManager
) -> Dict[str, Any]:
    """Run benchmark evaluation with ACE."""
    if not args.quiet:
        print(f"Starting evaluation with {args.model}...")

    # Create LLM client and ACE components
    client = create_llm_client(args)
    generator = Generator(client)
    environment = manager.get_benchmark(args.benchmark)

    results = []

    if args.skip_adaptation:
        # Direct evaluation without ACE adaptation
        playbook = Playbook()

        for i, sample in enumerate(samples):
            if not args.quiet and i % 10 == 0:
                print(f"Progress: {i}/{len(samples)} samples processed")

            # Generate response
            output = generator.generate(
                question=sample.question,
                context=sample.context,
                playbook=playbook
            )

            # Evaluate
            env_result = environment.evaluate(sample, output)

            results.append({
                "sample_id": sample.sample_id,
                "question": sample.question,
                "prediction": output.final_answer,
                "ground_truth": sample.ground_truth,
                "metrics": env_result.metrics,
                "feedback": env_result.feedback
            })

    else:
        # Full ACE adaptation
        reflector = Reflector(client)
        curator = Curator(client)
        adapter = OfflineAdapter(
            playbook=Playbook(),
            generator=generator,
            reflector=reflector,
            curator=curator,
            max_refinement_rounds=args.max_refinement_rounds,
            enable_explainability=True  # Enable explainability tracking
        )

        # Run adaptation
        adaptation_results = adapter.run(samples, environment, epochs=args.epochs)

        # Convert to results format
        for step in adaptation_results:
            results.append({
                "sample_id": step.sample.sample_id,
                "question": step.sample.question,
                "prediction": step.generator_output.final_answer,
                "ground_truth": step.sample.ground_truth,
                "metrics": step.environment_result.metrics,
                "feedback": step.environment_result.feedback,
                "reflection": step.reflection.raw if hasattr(step.reflection, 'raw') else None,
                "curator_output": step.curator_output.raw if hasattr(step.curator_output, 'raw') else None
            })

        # Export explainability data if available
        explainability_data = None
        if hasattr(adapter, 'evolution_tracker') and adapter.evolution_tracker:
            explainability_data = {
                "evolution": adapter.evolution_tracker.get_timeline_data(),
                "attribution": adapter.attribution_analyzer.generate_attribution_report() if hasattr(adapter, 'attribution_analyzer') and adapter.attribution_analyzer else None,
                "interactions": adapter.interaction_tracer.generate_interaction_report() if hasattr(adapter, 'interaction_tracer') and adapter.interaction_tracer else None
            }

    result_dict = {
        "benchmark": args.benchmark,
        "model": args.model,
        "samples_evaluated": len(results),
        "results": results,
        "summary": compute_summary_metrics(results)
    }

    # Add explainability data if available
    if 'explainability_data' in locals() and explainability_data:
        result_dict["explainability"] = explainability_data

    return result_dict


def compute_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute summary metrics across all results."""
    if not results:
        return {}

    # Collect all metric values
    all_metrics = {}
    for result in results:
        for metric_name, value in result["metrics"].items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Compute averages
    summary = {}
    for metric_name, values in all_metrics.items():
        summary[f"{metric_name}_mean"] = mean(values)
        summary[f"{metric_name}_min"] = min(values)
        summary[f"{metric_name}_max"] = max(values)

    return summary


def save_results(args: argparse.Namespace, evaluation_results: Dict[str, Any]) -> None:
    """Save evaluation results to files."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.benchmark}_{args.model}_{timestamp}"

    # Save summary results
    summary_file = output_dir / f"{base_name}_summary.json"
    summary_data = {
        "benchmark": evaluation_results["benchmark"],
        "model": evaluation_results["model"],
        "timestamp": timestamp,
        "samples_evaluated": evaluation_results["samples_evaluated"],
        "summary_metrics": evaluation_results["summary"],
        "configuration": {
            "split": args.split,
            "epochs": args.epochs,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "skip_adaptation": args.skip_adaptation
        }
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(f"Summary saved to: {summary_file}")

    # Save detailed results if requested
    if args.save_detailed:
        detailed_file = output_dir / f"{base_name}_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        if not args.quiet:
            print(f"Detailed results saved to: {detailed_file}")

    # Print summary to console
    print("\n" + "="*60)
    print(f"Benchmark: {evaluation_results['benchmark']}")
    print(f"Model: {evaluation_results['model']}")
    print(f"Samples: {evaluation_results['samples_evaluated']}")
    print("="*60)

    for metric, value in evaluation_results["summary"].items():
        if metric.endswith("_mean"):
            base_metric = metric[:-5]
            print(f"{base_metric.replace('_', ' ').title()}: {value:.2%}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle special commands
    if args.benchmark == "list":
        list_available_benchmarks()
        return

    # Set cache directory if specified
    if args.cache_dir:
        os.environ["BENCHMARK_CACHE_DIR"] = args.cache_dir

    # Initialize benchmark manager
    try:
        manager = BenchmarkTaskManager()
    except Exception as e:
        print(f"Error initializing benchmark manager: {e}")
        sys.exit(1)

    # Validate benchmark exists
    if args.benchmark not in manager.list_benchmarks():
        print(f"Error: Unknown benchmark '{args.benchmark}'")
        print("Use 'list' to see available benchmarks")
        sys.exit(1)

    # Validate benchmark configuration
    validation_errors = manager.validate_config(args.benchmark)
    if validation_errors:
        print(f"Benchmark validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)

    # Load benchmark data
    samples = load_benchmark_data(args, manager)

    # Run evaluation
    evaluation_results = run_evaluation(args, samples, manager)

    # Save and display results
    save_results(args, evaluation_results)

    if not args.quiet:
        print(f"\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()