# =============================================================================
# Logging Infrastructure
# =============================================================================
# The GCG attack uses a dual logging system:
#
# 1. DEBUG LOGGER (Python logging):
#    - Logs warnings, errors, info messages
#    - File: debug_{run_id}.log
#    - Usage: debug_logger.info("message"), debug_logger.warning("message")
#
# 2. STEP METRICS (JSONL):
#    - Logs step-wise metrics (losses, tokens, etc.)
#    - File: step_metrics_{run_id}.jsonl
#    - Format: {"step": 10, "loss": 0.523, ...} (step=-1 for initial, step>=0 for iterations)
#    - Usage: log_step_metric(step_metrics_path, {"step": 10, ...})
#
# Example usage in custom_gcg():
#   - debug_logger.info(f"Starting optimization with {num_steps} steps")
#   - log_step_metric(step_metrics_path, {"step": i, "loss": loss})
# =============================================================================
import typing
from betteroptsattack.utils import attack_utility as attack_utility
import json
import logging
from pathlib import Path


def setup_logging(
    run_id: str,
    debug_log_dir: typing.Optional[str] = None,
    metrics_dir: typing.Optional[str] = None,
):
    """
    Set up logging infrastructure for a GCG run.

    Creates:
    - Debug logger: Writes to debug_{run_id}.log
    - Step metrics file: step_metrics_{run_id}.jsonl (append mode)

    Args:
        run_id: Unique identifier for this run
        debug_log_dir: Directory for debug logs (or None to disable)
        metrics_dir: Directory for metrics

    Returns:
        Tuple of (debug_logger, step_metrics_path)
    """
    # Setup debug logger
    debug_logger = logging.getLogger(f"gcg.debug.{run_id}")
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.handlers.clear()  # Clear any existing handlers
    debug_logger.propagate = False  # Don't propagate to root logger

    if debug_log_dir:
        debug_path = Path(debug_log_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(debug_path / f"debug_{run_id}.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        debug_logger.addHandler(handler)
    else:
        debug_logger.addHandler(logging.NullHandler())

    # Setup metrics file paths
    step_metrics_path = None

    if metrics_dir:
        metrics_path = Path(metrics_dir)
        metrics_path.mkdir(parents=True, exist_ok=True)
        step_metrics_path = metrics_path / f"step_metrics_{run_id}.jsonl"

    return debug_logger, step_metrics_path


def log_step_metric(step_metrics_path: typing.Optional[Path], metric_dict: dict):
    """
    Append a step metric to the JSONL file.

    Args:
        step_metrics_path: Path to step metrics file (or None to skip)
        metric_dict: Dictionary containing metric data (must include "type" field)
    """
    if step_metrics_path is None:
        return

    with open(step_metrics_path, "a") as f:
        json.dump(metric_dict, f)
        f.write("\n")


def log_final_statistics(
    filter_tokenized_sequences: bool,
    total_candidates_checked: int,
    total_candidates_invalid: int,
    debug_logger: logging.Logger,
) -> None:
    """Log final decode-reencode validation statistics.

    Args:
        filter_tokenized_sequences: Whether filtering was enabled
        total_candidates_checked: Total number of candidates checked
        total_candidates_invalid: Total number of candidates invalidated
        logger: Logger instance
    """
    if not filter_tokenized_sequences or total_candidates_checked == 0:
        return

    invalid_rate = (total_candidates_invalid / total_candidates_checked) * 100
    debug_logger.info(
        f"DECODE-REENCODE VALIDATION: checked={total_candidates_checked}, invalid={total_candidates_invalid}, rate={invalid_rate:.2f}%"
    )
