#!/usr/bin/env python3
"""
Test Runner for Enhanced GCG Implementation
============================================

This script runs GCG attack evaluations using the enhanced custom_gcg function
from algorithms/gcg_enhanced.py instead of the custom implementation in
evaluate_gcg_extended.py.

Key differences from evaluate_gcg_extended.py:
- Uses the production GCG implementation with extended logging
- Tracks both exact match and substring match for generations
- Compatible with the existing experiment infrastructure
- Can be used as a drop-in replacement for testing

Usage:
------
Basic usage:
    uv run python run_gcg_enhanced.py --model gpt2 --max-samples 5

Run specific difficulty level:
    uv run python run_gcg_enhanced.py --model gpt2 --level 1 --max-samples 10

Run with custom parameters:
    uv run python run_gcg_enhanced.py --model gpt2 --adv-length 20 --topk 256 --max-steps 100

Run specific sample group:
    uv run python run_gcg_enhanced.py --model gpt2 --sample-group open_ended_short

Test specific samples by ID:
    uv run python run_gcg_enhanced.py --model gpt2 --sample-ids l1_s1 l2_s3

Disable early stopping (run all steps):
    uv run python run_gcg_enhanced.py --model gpt2 --no-early-stop

Output:
-------
Results are saved to 'results_enhanced/' directory with:
- Per-step metrics in JSONL format
- Configuration summaries in JSON
- Sample-specific results
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token.*")

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, total=None, desc=None, leave=True, position=None):
        return iterable


# Import project modules
import algorithms.gcg_enhanced as gcg_enhanced
import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
from test_samples_extended import (
    ALL_SAMPLES,
    get_samples_by_level,
    get_sample_by_id,
    get_new_samples,
    get_samples_by_group,
    get_open_ended_samples,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GCG attacks using enhanced implementation with extended logging"
    )

    # Model configuration
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model name or path (default: gpt2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )

    # Sample selection
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        help="Test only samples from a specific difficulty level",
    )
    parser.add_argument("--sample-ids", nargs="+", help="Specific sample IDs to test")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples to evaluate (default: 5)",
    )
    parser.add_argument(
        "--sample-group",
        type=str,
        choices=[
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "very_short",
            "very_long",
            "open_ended_short",
            "open_ended_long",
            "extra_long",
            "extra_long_only",
            "new",
            "original",
            "all",
        ],
        help="Select specific sample group to test",
    )
    parser.add_argument(
        "--new-only", action="store_true", help="Run only newly added samples"
    )

    # GCG attack parameters
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum optimization steps (default: 100)",
    )
    parser.add_argument(
        "--topk", type=int, default=256, help="Top-k tokens to consider (default: 256)"
    )
    parser.add_argument(
        "--forward-eval-candidates",
        type=int,
        default=128,
        help="Number of candidates to evaluate per step (default: 128)",
    )
    parser.add_argument(
        "--adv-length",
        type=int,
        default=10,
        help="Length of adversarial string for both prefix and suffix (default: 10)",
    )

    # Experiment configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_enhanced",
        help="Directory to save results (default: results_enhanced)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=True,
        help="Enable early stopping on exact match (default: enabled)",
    )
    parser.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable early stopping - run all steps",
    )
    parser.add_argument(
        "--check-metrics-every",
        type=int,
        default=10,
        help="Check extended metrics every N steps (default: 10)",
    )
    parser.add_argument(
        "--save-adv-every",
        type=int,
        default=25,
        help="Save adversarial string every N steps (default: 25)",
    )

    return parser.parse_args()


def initialize_model_and_tokenizer(
    model_name: str, device: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize the model and tokenizer."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model-specific configurations
    if "gpt2" in model_name.lower():
        tokenizer.padding_side = "left"

    # Dtype handling
    if "gemma" in model_name.lower():
        torch_dtype = torch.float32
        print("Using float32 for Gemma model")
    else:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    return model, tokenizer


def prepare_sample_for_attack(
    sample: Dict, tokenizer: AutoTokenizer, adv_length: int
) -> Dict:
    """Prepare a sample for GCG attack."""

    # Initialize adversarial strings
    init_config = {
        "strategy_type": "random",
        "prefix_length": adv_length,
        "suffix_length": adv_length,
        "seed": int(time.time()),
    }

    adv_prefix_init, adv_suffix_init = attack_utility.initialize_adversarial_strings(
        tokenizer, init_config
    )

    # Replace {optim_str} with placeholders
    prompt_template = sample["prompt"].replace(
        "{optim_str}",
        f"{attack_utility.ADV_PREFIX_INDICATOR} . {attack_utility.ADV_SUFFIX_INDICATOR}",
    )

    # Create tokenized data with retry logic for tokenization issues
    input_tokenized_data = attack_utility.string_masks_with_retry(
        tokenizer, prompt_template, adv_prefix_init, adv_suffix_init, sample["target"]
    )

    input_tokenized_data["sample_id"] = sample["id"]
    input_tokenized_data["target_text"] = sample["target"]

    return input_tokenized_data


def run_enhanced_gcg_attack(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample: Dict,
    args,
    exp_dir: Path,
    logger: experiment_logger.ExperimentLogger,
) -> Dict:
    """
    Run GCG attack using the enhanced custom_gcg implementation.
    """

    # Prepare sample
    input_tokenized_data = prepare_sample_for_attack(sample, tokenizer, args.adv_length)

    # Configure GCG hyperparameters
    gcg_hyperparams = {
        "signal_function": gcg_enhanced.og_gcg_signal,
        "true_loss_function": attack_utility.target_logprobs,
        "max_steps": args.max_steps,
        "topk": args.topk,
        "forward_eval_candidates": args.forward_eval_candidates,
    }

    # Set up metrics file path
    metrics_file = exp_dir / f"step_metrics_{sample['id']}.jsonl"

    # Run enhanced GCG attack
    start_time = time.time()

    result = gcg_enhanced.custom_gcg(
        model=model,
        tokenizer=tokenizer,
        input_tokenized_data=input_tokenized_data,
        custom_gcg_hyperparams=gcg_hyperparams,
        logger=logger,
        # Standard parameters
        eval_every_step=True,
        early_stop=args.early_stop,
        eval_initial=True,
        identical_outputs_before_stop=1,
        generation_config=attack_utility.DEFAULT_TEXT_GENERATION_CONFIG,
        to_cache_logits=False,
        to_cache_attentions=False,
        # Extended logging parameters
        save_metrics_path=str(metrics_file),
        check_extended_metrics_every_n_steps=args.check_metrics_every,
        save_adv_string_every_n_steps=args.save_adv_every,
    )

    attack_time = time.time() - start_time

    # Process results based on return type
    if isinstance(result, dict):
        # Enhanced logging was enabled
        return {
            "final_success": result["final_success"],
            "attack_time": attack_time,
            "total_steps": result["total_steps"],
            "per_step_metrics": result["per_step_metrics"],
            "final_adversarial_string": tokenizer.decode(
                result["best_output_sequences"][-1][
                    input_tokenized_data["masks"]["optim_mask"]
                ]
            )
            if result["best_output_sequences"]
            else None,
        }
    else:
        # Standard return format (backward compatibility)
        logprobs_sequences, best_output_sequences = result
        return {
            "final_success": False,  # Cannot determine without generation
            "attack_time": attack_time,
            "total_steps": args.max_steps,
            "per_step_metrics": None,
            "final_adversarial_string": tokenizer.decode(
                best_output_sequences[-1][input_tokenized_data["masks"]["optim_mask"]]
            )
            if best_output_sequences
            else None,
        }


def main():
    """Main execution function."""
    args = parse_arguments()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Select samples
    if args.sample_ids:
        samples = [get_sample_by_id(sid) for sid in args.sample_ids]
        samples = [s for s in samples if s is not None]
    elif args.new_only:
        samples = get_new_samples()
    elif args.sample_group:
        samples = get_samples_by_group(args.sample_group)
    elif args.level:
        samples = get_samples_by_level(args.level)
    else:
        samples = ALL_SAMPLES

    # Limit number of samples
    if args.max_samples:
        samples = samples[: args.max_samples]

    if not samples:
        print("No samples to evaluate!")
        return

    print(f"\nRunning Enhanced GCG Evaluation")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Samples: {len(samples)}")
    print(f"Max steps: {args.max_steps}")
    print(f"Top-k: {args.topk}")
    print(f"Adversarial length: {args.adv_length}")
    print(f"Early stopping: {args.early_stop}")
    print(f"Output directory: {args.output_dir}")
    print(f"=" * 60)

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model, args.device)

    # Create output directory
    model_dir_name = args.model.replace("/", "_")
    exp_name = f"enhanced_adv{args.adv_length}_topk{args.topk}_steps{args.max_steps}"
    exp_dir = Path(args.output_dir) / model_dir_name / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = experiment_logger.ExperimentLogger(str(exp_dir / "logs"))

    # Results collection
    all_results = {
        "configuration": {
            "model": args.model,
            "adv_length": args.adv_length,
            "topk": args.topk,
            "max_steps": args.max_steps,
            "early_stop": args.early_stop,
            "check_metrics_every": args.check_metrics_every,
            "save_adv_every": args.save_adv_every,
        },
        "samples": [],
    }

    # Process each sample
    pbar = tqdm(samples, desc="Processing samples")
    for sample in pbar:
        pbar.set_description(f"Sample {sample['id']} (L{sample['level']})")

        try:
            # Run attack
            result = run_enhanced_gcg_attack(
                model, tokenizer, sample, args, exp_dir, logger
            )

            # Prepare sample result
            sample_result = {
                "sample_id": sample["id"],
                "level": sample["level"],
                "prompt_tokens": sample.get("prompt_tokens", 0),
                "target_tokens": sample.get("target_tokens", 0),
                "plausibility": sample.get("plausibility", 0),
                "final_success": result["final_success"],
                "attack_time": result["attack_time"],
                "total_steps": result["total_steps"],
                "final_adversarial_string": result["final_adversarial_string"],
                "metrics_file": f"step_metrics_{sample['id']}.jsonl",
            }

            # Check for final metrics if we have them
            if result["per_step_metrics"]:
                last_metric = result["per_step_metrics"][-1]
                sample_result["final_starts_with_target"] = last_metric.get(
                    "generation_starts_with_target", False
                )
                sample_result["final_argmax_match"] = last_metric.get(
                    "argmax_matches_target", False
                )

            all_results["samples"].append(sample_result)

            # Update progress bar
            success_indicator = "✓" if result["final_success"] else "✗"
            pbar.set_postfix(
                {"success": success_indicator, "time": f"{result['attack_time']:.1f}s"}
            )

            # Save individual result
            with open(exp_dir / f"sample_{sample['id']}_result.json", "w") as f:
                json.dump(sample_result, f, indent=2)

        except Exception as e:
            print(f"\nError processing sample {sample['id']}: {e}")
            error_result = {
                "sample_id": sample["id"],
                "level": sample["level"],
                "error": str(e),
                "final_success": False,
            }
            all_results["samples"].append(error_result)

    # Save summary
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary statistics
    successes = sum(1 for s in all_results["samples"] if s.get("final_success", False))
    starts_with_target = sum(
        1 for s in all_results["samples"] if s.get("final_starts_with_target", False)
    )
    argmax_matches = sum(
        1 for s in all_results["samples"] if s.get("final_argmax_match", False)
    )
    total = len(all_results["samples"])

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total samples: {total}")
    print(
        f"Successful attacks (starts with target): {successes} ({100 * successes / total:.1f}%)"
    )
    print(
        f"Starts with target: {starts_with_target} ({100 * starts_with_target / total:.1f}%)"
    )
    print(f"Argmax matches: {argmax_matches} ({100 * argmax_matches / total:.1f}%)")
    print(f"\nResults saved to: {exp_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

