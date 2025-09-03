# Enhanced GCG Test Runner Documentation

## Overview

The `run_gcg_enhanced.py` script provides a test runner that uses the enhanced GCG implementation from `algorithms/gcg_enhanced.py`. This implementation includes extended logging capabilities while maintaining the core GCG algorithm functionality.

## Key Features

### Enhanced Logging

- **Per-step metrics**: Loss, argmax matches, generation matches, time elapsed
- **Dual matching criteria**:
  - Exact match (original behavior for backward compatibility)
  - Substring match (more lenient, case-insensitive)
- **Incremental saving**: Metrics saved to JSONL files to prevent data loss
- **Adversarial string tracking**: Periodic saving of current adversarial strings

### Comparison with `evaluate_gcg_extended.py`

| Feature            | evaluate_gcg_extended.py | run_gcg_enhanced.py      |
| ------------------ | ------------------------ | ------------------------ |
| GCG Implementation | Custom simplified        | Production gcg_enhanced  |
| Loss Computation   | Same (cross-entropy)     | Same (cross-entropy)     |
| Gradient Caching   | No                       | Optional                 |
| Exact Match        | No                       | Yes (for early stopping) |
| Substring Match    | Yes (only)               | Yes (as metric)          |
| Argmax Checking    | Yes                      | Yes                      |
| Early Stopping     | On substring match       | On exact match           |
| Code Complexity    | Simplified               | Full-featured            |

## Installation

No additional installation required beyond the existing project dependencies.

## Usage Examples

### Basic Usage

```bash
# Run with default settings (5 samples, GPT-2)
uv run python run_gcg_enhanced.py

# Run with more samples
uv run python run_gcg_enhanced.py --max-samples 20
```

### Model Selection

```bash
# Use GPT-2 (default)
uv run python run_gcg_enhanced.py --model gpt2

# Use Gemma-2B (if available)
uv run python run_gcg_enhanced.py --model google/gemma-2b

# Use Llama-2 (if available)
uv run python run_gcg_enhanced.py --model meta-llama/Llama-2-7b-hf
```

### Sample Selection

```bash
# Run specific difficulty level
uv run python run_gcg_enhanced.py --level 1 --max-samples 10

# Run specific sample group
uv run python run_gcg_enhanced.py --sample-group open_ended_short

# Run only new samples
uv run python run_gcg_enhanced.py --new-only

# Run specific samples by ID
uv run python run_gcg_enhanced.py --sample-ids l1_s1 l2_s3 l3_s5
```

### GCG Parameters

```bash
# Adjust adversarial string length
uv run python run_gcg_enhanced.py --adv-length 20

# Change top-k tokens
uv run python run_gcg_enhanced.py --topk 512

# Increase optimization steps
uv run python run_gcg_enhanced.py --max-steps 200

# Change number of candidates per step
uv run python run_gcg_enhanced.py --forward-eval-candidates 256
```

### Early Stopping Control

```bash
# Disable early stopping (run all steps)
uv run python run_gcg_enhanced.py --no-early-stop

# Enable early stopping (default)
uv run python run_gcg_enhanced.py --early-stop
```

### Logging Configuration

```bash
# Check generation every 5 steps (instead of 10)
uv run python run_gcg_enhanced.py --check-metrics-every 5

# Save adversarial string every 10 steps (instead of 25)
uv run python run_gcg_enhanced.py --save-adv-every 10
```

### Output Directory

```bash
# Custom output directory
uv run python run_gcg_enhanced.py --output-dir my_results

# Default is "results_enhanced/"
```

## Output Structure

```
results_enhanced/
├── gpt2/
│   └── enhanced_adv10_topk256_steps100/
│       ├── logs/                           # Experiment logs
│       ├── step_metrics_l1_s1.jsonl       # Per-step metrics for each sample
│       ├── step_metrics_l1_s2.jsonl
│       ├── sample_l1_s1_result.json       # Individual sample results
│       ├── sample_l1_s2_result.json
│       └── summary.json                   # Overall experiment summary
```

### Metrics File Format (JSONL)

Each line in the step metrics file contains:

```json
{
  "step": 0,
  "loss": 15.234,
  "argmax_matches_target": false,
  "generation_exact_match": false,
  "generation_substring_match": false,
  "generated_text": "First 100 chars of generation...",
  "current_adv_string": "Current adversarial string",
  "time_elapsed": 0.523
}
```

### Summary File Format

```json
{
  "configuration": {
    "model": "gpt2",
    "adv_length": 10,
    "topk": 256,
    "max_steps": 100,
    "early_stop": true,
    "check_metrics_every": 10,
    "save_adv_every": 25
  },
  "samples": [
    {
      "sample_id": "l1_s1",
      "level": 1,
      "prompt_tokens": 8,
      "target_tokens": 1,
      "plausibility": 3,
      "final_success": true,
      "final_exact_match": true,
      "final_substring_match": true,
      "final_argmax_match": false,
      "attack_time": 23.5,
      "total_steps": 45,
      "final_adversarial_string": "...",
      "metrics_file": "step_metrics_l1_s1.jsonl"
    }
  ]
}
```

## Batch Running

For running multiple configurations, you can create a simple bash script:

```bash
#!/bin/bash
# run_experiments.sh

# Different adversarial lengths
for adv_len in 10 20 30; do
    uv run python run_gcg_enhanced.py --adv-length $adv_len --max-samples 10
done

# Different models
for model in "gpt2" "gpt2-medium"; do
    uv run python run_gcg_enhanced.py --model $model --max-samples 10
done

# Different sample groups
for group in "level_1" "level_2" "open_ended_short"; do
    uv run python run_gcg_enhanced.py --sample-group $group
done
```

## Key Differences from Original Implementations

### 1. Early Stopping Behavior

- **run_gcg_enhanced.py**: Stops on **exact match** (maintains backward compatibility)
- **evaluate_gcg_extended.py**: Stops on **substring match** (more lenient)

### 2. Metrics Tracking

- Both track similar metrics but:
  - Enhanced version tracks **both** exact and substring matches
  - Enhanced version can leverage caching if needed

### 3. Code Organization

- **run_gcg_enhanced.py**: Uses production code from `gcg_enhanced.py`
- **evaluate_gcg_extended.py**: Has its own simplified implementation

### 4. Return Values

- Enhanced version returns a dictionary with comprehensive metrics when logging is enabled
- Falls back to tuple return for backward compatibility when logging is disabled

## Troubleshooting

### Out of Memory

- Reduce `--forward-eval-candidates`
- Reduce `--topk`
- Use CPU instead of GPU: `--device cpu`

### Slow Execution

- Enable early stopping (default)
- Reduce `--max-steps`
- Reduce `--max-samples`

### NaN/Inf Errors

- The enhanced implementation includes NaN checking
- For Gemma models, float32 is automatically used
- Check console output for NaN warnings

## Performance Tips

1. **GPU Usage**: Always use GPU when available for significant speedup
2. **Batch Size**: The `--forward-eval-candidates` parameter affects memory usage
3. **Early Stopping**: Keep enabled unless you need complete optimization curves
4. **Metric Frequency**: Balance between `--check-metrics-every` for detailed tracking vs speed

## Extending the Runner

To add custom functionality:

1. **Custom Signal Functions**: Modify the `signal_function` in `gcg_hyperparams`
2. **Custom Loss Functions**: Change `true_loss_function`
3. **Additional Metrics**: Extend the `check_extended_metrics_every_n_steps` section
4. **Different Initialization**: Modify `prepare_sample_for_attack`

## Contact

For issues or questions about this implementation, refer to the main project documentation or create an issue in the project repository.
