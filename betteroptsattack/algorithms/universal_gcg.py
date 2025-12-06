import torch
import transformers
import typing
import numpy as np
from betteroptsattack.utils import attack_utility as attack_utility
from betteroptsattack.utils import logging as gcg_logging
from betteroptsattack.algorithms import signals
import random
import gc
import time
import logging
import inspect
from tqdm import tqdm


GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS


def DEFAULT_GCG_RANDOMNESS_STRATEGY(
    tokenizer,
    best_tokens_indices,
    input_tokenized_data_list,
    substitution_validity_function,
    max_candidate_size,
    debug_logger: typing.Optional[logging.Logger] = None,
):
    """
    Default randomness strategy for universal GCG.

    Note: Decode-reencode filtering is done AFTER loss computation via
    _apply_universal_decode_reencode_filter for consistency with single-sample GCG.

    Args:
        tokenizer: Tokenizer
        best_tokens_indices: Top-k token indices from signal function
        input_tokenized_data_list: List of input tokenized data
        substitution_validity_function: Optional custom validation function
        max_candidate_size: Number of candidates to generate
        debug_logger: Optional logger for logging statistics

    Returns:
        List of candidate tensors (one per input sample)
    """
    indices_to_sample = set()
    indices_to_exclude = set()

    while len(indices_to_sample) < max_candidate_size:
        first_coordinate = (
            torch.randint(0, best_tokens_indices.shape[0], (1,)).to(torch.int32).item()
        )
        second_coordinate = (
            torch.randint(0, best_tokens_indices.shape[1], (1,)).to(torch.int32).item()
        )
        if (first_coordinate, second_coordinate) in indices_to_sample:
            continue
        if (first_coordinate, second_coordinate) in indices_to_exclude:
            continue

        all_substitutions_valid = True
        for input_tokenized_data in input_tokenized_data_list:
            masks_data = input_tokenized_data["masks"]
            optim_mask = masks_data["optim_mask"]
            random_substitution_make = input_tokenized_data["tokens"].clone()
            random_substitution_make[optim_mask[first_coordinate]] = (
                best_tokens_indices[(first_coordinate, second_coordinate)]
            )

            # Check custom substitution validity function (e.g., SecAlign filter)
            if substitution_validity_function is not None:
                if not substitution_validity_function(
                    random_substitution_make, tokenizer=tokenizer, masks_data=masks_data
                ):
                    indices_to_exclude.add((first_coordinate, second_coordinate))
                    all_substitutions_valid = False
                    break

        if not all_substitutions_valid:
            continue
        else:
            indices_to_sample.add((first_coordinate, second_coordinate))

    candidates_list = []
    for input_tokenized_data in input_tokenized_data_list:
        input_new_candidates = []
        for index_to_sample in indices_to_sample:
            masks_data = input_tokenized_data["masks"]
            optim_mask = masks_data["optim_mask"]
            random_substitution_make = input_tokenized_data["tokens"].clone()
            random_substitution_make[optim_mask[index_to_sample[0]]] = (
                best_tokens_indices[(index_to_sample[0], index_to_sample[1])]
            )
            input_new_candidates.append(random_substitution_make)
        candidates_list.append(torch.stack(input_new_candidates))

    return candidates_list


def DEFAULT_ON_STEP(*args, **kwargs):
    pass


# =============================================================================
# Universal GCG Helper Functions
# =============================================================================


def _validate_universal_decode_reencode(
    candidate_tokens: torch.Tensor,
    tokenizer: transformers.AutoTokenizer,
) -> bool:
    """
    Validate that a candidate token sequence survives decode-reencode cycle.

    Args:
        candidate_tokens: Token sequence to validate
        tokenizer: Tokenizer

    Returns:
        True if tokens are stable through decode-reencode, False otherwise
    """
    try:
        # Decode full sequence
        decoded_text = tokenizer.decode(
            candidate_tokens.cpu(), skip_special_tokens=False
        )

        # Re-encode
        reencoded_tokens = tokenizer.encode(
            decoded_text, return_tensors="pt", add_special_tokens=False
        )[0]

        # Check if tokenization is preserved
        return torch.equal(candidate_tokens.cpu(), reencoded_tokens.cpu())
    except Exception:
        # If any error occurs during validation, reject the candidate
        return False


def _apply_universal_decode_reencode_filter(
    candidate_tensors: typing.List[torch.Tensor],
    tokenizer: transformers.AutoTokenizer,
    true_losses: torch.Tensor,
    debug_logger: logging.Logger,
    step_num: int,
    rejection_threshold: float = 0.5,
) -> typing.Tuple[int, int]:
    """Apply decode-reencode validation filter to universal candidates.

    For each candidate, checks all samples. If >rejection_threshold fraction of samples
    fail decode-reencode validation, sets loss to inf for that candidate.

    Args:
        candidate_tensors: List of candidate tensors (one per input sample)
        tokenizer: Tokenizer
        true_losses: Loss values for candidates (modified in place)
        debug_logger: Logger instance
        step_num: Current step number
        rejection_threshold: Fraction of samples that must fail to reject candidate (default 0.5)

    Returns:
        Tuple of (total_candidates_checked, total_candidates_invalid)
    """
    num_candidates = candidate_tensors[0].shape[0]
    num_samples = len(candidate_tensors)
    total_invalid = 0

    for candidate_idx in range(num_candidates):
        # Count how many samples fail for this candidate
        samples_failing = 0
        for sample_idx in range(num_samples):
            candidate_tokens = candidate_tensors[sample_idx][candidate_idx]
            if not _validate_universal_decode_reencode(candidate_tokens, tokenizer):
                samples_failing += 1

        # If more than threshold fraction fail, reject this candidate
        failure_fraction = samples_failing / num_samples
        if failure_fraction > rejection_threshold:
            true_losses[candidate_idx] = float("inf")
            total_invalid += 1

    if total_invalid > 0:
        debug_logger.info(
            f"Step {step_num}: Filtered {total_invalid}/{num_candidates} candidates "
            f"(>{rejection_threshold * 100:.0f}% samples failed decode-reencode)"
        )

    if total_invalid == num_candidates:
        debug_logger.warning(
            f"Step {step_num}: ALL candidates failed decode-reencode validation! "
            "Using best of invalid candidates."
        )

    return num_candidates, total_invalid


def _compute_universal_argmax_match(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    current_input_tokenized_data_list: typing.List[typing.Dict],
) -> float:
    """Compute fraction of samples where argmax predictions match target.

    Args:
        models: List of models
        tokenizer: Tokenizer
        current_input_tokenized_data_list: Current input tokenized data list

    Returns:
        Fraction in [0, 1] of samples where argmax matches target
    """
    num_elements_per_model = len(current_input_tokenized_data_list) // len(models)
    total_samples = len(current_input_tokenized_data_list)
    samples_matching = 0

    for model_idx, model in enumerate(models):
        start_idx = model_idx * num_elements_per_model
        end_idx = start_idx + num_elements_per_model

        for input_data in current_input_tokenized_data_list[start_idx:end_idx]:
            tokens = input_data["tokens"]
            masks_data = input_data["masks"]
            target_mask = masks_data["target_mask"]
            target_tokens = tokens[target_mask]

            # Check argmax match
            with torch.no_grad():
                logits = model(tokens.unsqueeze(0).to(model.device)).logits[0]
                # Get predictions at target positions (shift by 1 for causal LM)
                pred_logits = logits[target_mask - 1]
                predictions = torch.argmax(pred_logits, dim=-1)

                if torch.all(predictions.cpu() == target_tokens.cpu()):
                    samples_matching += 1

    return samples_matching / total_samples if total_samples > 0 else 0.0


def _setup_universal_caching(
    to_cache_logits: bool,
    to_cache_attentions: bool,
) -> typing.Tuple[typing.Any, typing.Optional[typing.Any]]:
    """Setup caching for universal GCG with average logprobs.

    Args:
        to_cache_logits: Whether to cache logprobs
        to_cache_attentions: Whether to cache attentions

    Returns:
        Tuple of (average_target_logprobs_function, att_cacher)
    """
    if to_cache_logits:
        average_target_logprobs = attack_utility.CachedAverageLogprobs()
    else:
        raise ValueError(
            "Universal GCG requires caching enabled. Set to_cache_logits=True"
        )

    # Attention caching is not currently implemented for universal GCG
    # The parameter is accepted for API consistency but att_cacher is always None
    att_cacher = None

    return average_target_logprobs, att_cacher


def _evaluate_universal_initial_state(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    masks_data_list: typing.List[typing.Dict[str, torch.Tensor]],
    true_loss_function: typing.Callable,
    average_target_logprobs_function: typing.Callable,
    true_loss_kwargs: typing.Dict,
    debug_logger: logging.Logger,
) -> typing.Tuple[typing.List[typing.Dict], typing.List[float], typing.Dict]:
    """Evaluate initial state before universal optimization.

    Args:
        models: List of models
        tokenizer: Tokenizer
        input_tokenized_data_list: List of input tokenized data
        masks_data_list: List of masks
        true_loss_function: Loss function
        average_target_logprobs_function: Average logprobs function
        true_loss_kwargs: Kwargs for loss function
        debug_logger: Logger instance

    Returns:
        Tuple of (best_tokens_dicts_list, average_logprobs_list, initial_metric)
    """
    step_start_time = time.time()

    # Compute initial loss
    initial_true_loss = true_loss_function(
        models,
        tokenizer,
        [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list],
        masks_data_list,
        debug_logger,
        **true_loss_kwargs,
    )

    # Compute initial average logprobs
    initial_average_logprobs = average_target_logprobs_function(
        models,
        tokenizer,
        [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list],
        masks_data_list,
        debug_logger,
    )
    initial_average_logprobs = initial_average_logprobs.item()

    best_tokens_dicts_list = [
        attack_utility.form_best_tokens_dict(input_tokenized_data_list)
    ]
    average_logprobs_list = [initial_average_logprobs]

    initial_metric = {
        "step": -1,
        "loss": initial_average_logprobs,
        "time_elapsed": time.time() - step_start_time,
    }

    return best_tokens_dicts_list, average_logprobs_list, initial_metric


def _generate_universal_candidates(
    best_tokens_indices: torch.Tensor,
    current_input_tokenized_data_list: typing.List[typing.Dict],
    randomness_strategy: typing.Callable,
    substitution_validity_function: typing.Optional[typing.Callable],
    num_forward_evals: int,
    tokenizer: transformers.AutoTokenizer,
    filter_tokenized_sequences: bool = True,
    debug_logger: typing.Optional[logging.Logger] = None,
) -> typing.List[torch.Tensor]:
    """Generate universal candidates using the randomness strategy.

    Args:
        best_tokens_indices: Top-k token indices from signal function
        current_input_tokenized_data_list: Current input tokenized data list
        randomness_strategy: Function to generate candidates
        substitution_validity_function: Optional validation function
        num_forward_evals: Number of candidates to generate
        tokenizer: Tokenizer
        filter_tokenized_sequences: Whether to filter candidates that fail decode-reencode
        debug_logger: Optional logger for logging statistics

    Returns:
        List of candidate tensors (one per input sample)
    """
    # Check if the randomness strategy accepts debug_logger
    # This maintains backward compatibility with custom strategies
    sig = inspect.signature(randomness_strategy)
    params = sig.parameters

    kwargs = {}
    if "debug_logger" in params:
        kwargs["debug_logger"] = debug_logger

    candidates = randomness_strategy(
        tokenizer,
        best_tokens_indices,
        current_input_tokenized_data_list,
        substitution_validity_function,
        num_forward_evals,
        **kwargs,
    )

    return candidates


def _compute_universal_step_metrics(
    best_loss: float,
    average_logprobs: float,
    argmax_match: float,
    step_num: int,
    step_start_time: float,
    models: list[transformers.AutoModelForCausalLM],
) -> typing.Dict:
    """Compute metrics for the current universal optimization step.

    Args:
        best_loss: Best loss value for this step
        average_logprobs: Average logprobs across samples
        argmax_match: Fraction of samples where argmax matches target [0, 1]
        step_num: Current step number
        step_start_time: Time when step started
        models: List of models

    Returns:
        Dict containing step metrics
    """
    step_metric = {
        "step": step_num,
        "best_loss": best_loss,
        "avg_loss": average_logprobs,
        "argmax_match": argmax_match,
        "time_elapsed": time.time() - step_start_time,
        "max_memory_reserved": max(
            torch.cuda.max_memory_reserved(device=model.device) / 1024**3
            for model in models
        ),
    }

    return step_metric


def _check_universal_early_stopping(
    argmax_match: float,
    argmax_match_threshold: float,
    successive_correct_outputs: int,
    identical_outputs_before_stop: int,
    debug_logger: logging.Logger,
    step_num: int,
) -> typing.Tuple[bool, int]:
    """Check if early stopping criteria are met for universal GCG.

    Uses argmax match fraction instead of generation-based checks for speed.

    Args:
        argmax_match: Fraction of samples where argmax matches target [0, 1]
        argmax_match_threshold: Threshold for considering a step successful
        successive_correct_outputs: Count of successive correct outputs
        identical_outputs_before_stop: Number of consecutive successes needed
        debug_logger: Logger instance
        step_num: Current step number

    Returns:
        Tuple of (should_stop, updated_successive_correct_outputs)
    """
    if argmax_match >= argmax_match_threshold:
        successive_correct_outputs += 1
        debug_logger.info(
            f"Step {step_num}: Argmax match {argmax_match:.2%} >= {argmax_match_threshold:.0%} "
            f"({successive_correct_outputs}/{identical_outputs_before_stop})"
        )
        if successive_correct_outputs >= identical_outputs_before_stop:
            return True, successive_correct_outputs
    else:
        successive_correct_outputs = 0

    return False, successive_correct_outputs


def _evaluate_universal_final_best(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    best_tokens_dicts_list: typing.List[typing.Dict],
    average_logprobs_list: typing.List[float],
    input_tokenized_data_list: typing.List[typing.Dict],
    generation_config: typing.Dict,
    debug_logger: logging.Logger,
    timestamp_start: float,
    global_max_memory_reserved: float,
    total_steps: int,
) -> typing.Dict:
    """Evaluate the global best result and return final metrics.

    Args:
        models: List of models
        tokenizer: Tokenizer
        best_tokens_dicts_list: List of best token dicts from each step
        average_logprobs_list: List of average logprobs from each step
        input_tokenized_data_list: Original input tokenized data list
        generation_config: Configuration for generation
        debug_logger: Logger instance
        timestamp_start: Timestamp when optimization started
        global_max_memory_reserved: Maximum GPU memory reserved
        total_steps: Total number of steps completed

    Returns:
        Dict containing final evaluation metrics
    """
    # Find global best based on average logprobs
    if not average_logprobs_list:
        return {
            "best_step": -1,
            "best_loss": float("inf"),
            "total_steps": total_steps,
            "total_runtime": time.time() - timestamp_start,
            "max_memory_reserved": global_max_memory_reserved,
            "samples_matching": 0,
            "total_samples": len(input_tokenized_data_list),
        }

    logprobs_tensor = torch.tensor(average_logprobs_list)
    best_idx = torch.argmin(logprobs_tensor).item()
    best_loss = average_logprobs_list[best_idx]
    best_tokens_dict = best_tokens_dicts_list[best_idx]

    debug_logger.info(f"Global best found at step {best_idx} with loss {best_loss:.4f}")

    # Apply best tokens to input data and evaluate
    best_input_tokenized_data_list = attack_utility.update_all_tokens(
        best_tokens_dict, input_tokenized_data_list
    )

    # Evaluate each sample
    num_elements_per_model = len(best_input_tokenized_data_list) // len(models)
    samples_matching = 0
    sample_results = []

    for model_idx, model in enumerate(models):
        start_idx = model_idx * num_elements_per_model
        end_idx = start_idx + num_elements_per_model

        for sample_idx, input_data in enumerate(
            best_input_tokenized_data_list[start_idx:end_idx]
        ):
            tokens = input_data["tokens"]
            masks_data = input_data["masks"]
            input_mask = masks_data["input_mask"]
            target_mask = masks_data["target_mask"]
            target_tokens = tokens[target_mask]

            # Generate text
            input_tokens_for_generation = tokens[input_mask]
            with torch.inference_mode():
                generated_tokens = model.generate(
                    torch.unsqueeze(input_tokens_for_generation, dim=0).to(
                        model.device
                    ),
                    attention_mask=torch.ones(
                        1, len(input_tokens_for_generation), device=model.device
                    ),
                    **generation_config,
                )
                input_length = len(input_tokens_for_generation)
                generated_text = tokenizer.decode(
                    generated_tokens[0, input_length:], skip_special_tokens=True
                )

            # Check if generation starts with target
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            starts_with_target = generated_text.strip().startswith(target_text.strip())

            if starts_with_target:
                samples_matching += 1

            sample_results.append(
                {
                    "model_idx": model_idx,
                    "sample_idx": sample_idx,
                    "starts_with_target": starts_with_target,
                    "generated_text": generated_text[:200],
                    "target_text": target_text[:100],
                }
            )

    # Extract and decode the best injection strings
    prefix_injection = tokenizer.decode(
        best_tokens_dict["prefix_tokens"], skip_special_tokens=False
    )
    suffix_injection = tokenizer.decode(
        best_tokens_dict["suffix_tokens"], skip_special_tokens=False
    )

    debug_logger.info(f"Best prefix injection: {prefix_injection}")
    debug_logger.info(f"Best suffix injection: {suffix_injection}")
    debug_logger.info(
        f"Samples matching target: {samples_matching}/{len(best_input_tokenized_data_list)}"
    )

    return {
        "best_step": best_idx,
        "best_loss": best_loss,
        "prefix_injection": prefix_injection,
        "suffix_injection": suffix_injection,
        "samples_matching": samples_matching,
        "total_samples": len(best_input_tokenized_data_list),
        "sample_results": sample_results,
        "total_steps": total_steps,
        "total_runtime": time.time() - timestamp_start,
        "max_memory_reserved": global_max_memory_reserved,
    }


def _extract_universal_best_tokens(
    candidate_tensors: typing.List[torch.Tensor],
    best_idx: int,
    masks_data_list: typing.List[typing.Dict[str, torch.Tensor]],
) -> typing.Dict[str, torch.Tensor]:
    """Extract prefix and suffix tokens from the best candidate.

    Args:
        candidate_tensors: List of candidate tensors (one per input sample)
        best_idx: Index of best candidate
        masks_data_list: List of masks

    Returns:
        Dict with 'prefix_tokens' and 'suffix_tokens'
    """
    best_tokens_dict = {
        "prefix_tokens": candidate_tensors[0][best_idx][
            masks_data_list[0]["prefix_mask"]
        ],
        "suffix_tokens": candidate_tensors[0][best_idx][
            masks_data_list[0]["suffix_mask"]
        ],
    }
    return best_tokens_dict


def weakly_universal_gcg(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    universal_gcg_hyperparameters: typing.Dict,
    *,
    eval_initial: bool = True,
    generation_config: typing.Dict = None,
    to_cache_logits: bool = True,
    to_cache_attentions: bool = True,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    # Logging parameters
    run_id: typing.Optional[str] = None,
    debug_log_dir: typing.Optional[str] = None,
    metrics_dir: typing.Optional[str] = None,
    metrics_every_n_steps: int = 1,
    # Decode-reencode validation (applied after loss computation)
    filter_tokenized_sequences: bool = True,
    decode_reencode_rejection_threshold: float = 0.5,
    # Early stopping parameters (using argmax match)
    early_stop: bool = False,
    identical_outputs_before_stop: int = 3,
    argmax_match_threshold: float = 1.0,
    # Random seed for reproducibility
    seed: typing.Optional[int] = None,
) -> typing.Tuple[typing.List[typing.Dict], typing.List[float], typing.Dict]:
    """
    Run universal GCG optimization attack across multiple samples/models.

    Args:
        models: List of models (can be on different GPUs)
        tokenizer: HuggingFace tokenizer
        input_tokenized_data_list: List of tokenized input data (must be normalized)
        universal_gcg_hyperparameters: Dict containing hyperparameters
        eval_initial: Whether to evaluate initial state
        generation_config: Configuration for text generation (used for final validation)
        to_cache_logits: Whether to cache logprobs
        to_cache_attentions: Whether to cache attentions
        clamp_tokens: Whether to clamp token IDs
        ascii_only: Whether to use ASCII-only tokens
        run_id: Unique identifier for this run
        debug_log_dir: Directory for debug logs
        metrics_dir: Directory for metrics
        metrics_every_n_steps: How often to log metrics
        filter_tokenized_sequences: Whether to filter candidates that fail decode-reencode
        decode_reencode_rejection_threshold: Reject candidate if >threshold samples fail (default 0.5)
        early_stop: Whether to enable early stopping based on argmax match
        identical_outputs_before_stop: Number of consecutive successes before stopping
        argmax_match_threshold: Fraction of samples that must have argmax match (default 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (best_tokens_dicts_list, average_logprobs_list, final_metrics) where:
        - best_tokens_dicts_list: List of dicts with 'prefix_tokens' and 'suffix_tokens'
        - average_logprobs_list: List of average logprobs values for each step
        - final_metrics: Dict containing final evaluation metrics
    """
    timestamp_start = time.time()

    # Setup logging infrastructure
    if run_id is None:
        run_id = f"universal_{int(timestamp_start)}"

    debug_logger, step_metrics_path = gcg_logging.setup_logging(
        run_id=run_id, debug_log_dir=debug_log_dir, metrics_dir=metrics_dir
    )

    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        debug_logger.info(f"Random seed set to {seed}")

    # Setup caching (att_cacher not used in universal GCG)
    average_target_logprobs, _ = _setup_universal_caching(
        to_cache_logits, to_cache_attentions
    )

    # Extract hyperparameters
    max_steps = universal_gcg_hyperparameters["max_steps"]
    topk = universal_gcg_hyperparameters["topk"]
    num_forward_evals = universal_gcg_hyperparameters["forward_eval_candidates"]

    signal_function = universal_gcg_hyperparameters.get(
        "signal_function", signals.average_target_logprobs_signal
    )
    true_loss_function = universal_gcg_hyperparameters.get(
        "true_loss_function", average_target_logprobs
    )
    substitution_validity_function = universal_gcg_hyperparameters.get(
        "substitution_validity_function", None
    )
    signal_kwargs = universal_gcg_hyperparameters.get("signal_kwargs", None)
    true_loss_kwargs = universal_gcg_hyperparameters.get("true_loss_kwargs", None)
    randomness_strategy = universal_gcg_hyperparameters.get(
        "randomness_strategy", DEFAULT_GCG_RANDOMNESS_STRATEGY
    )

    on_step_begin = universal_gcg_hyperparameters.get("on_step_begin", DEFAULT_ON_STEP)
    on_step_begin_kwargs = universal_gcg_hyperparameters.get("on_step_begin_kwargs", {})
    on_step_end = universal_gcg_hyperparameters.get("on_step_end", DEFAULT_ON_STEP)
    on_step_end_kwargs = universal_gcg_hyperparameters.get("on_step_end_kwargs", {})

    if true_loss_kwargs is None:
        true_loss_kwargs = {}

    # Set default generation config if not provided
    if generation_config is None:
        generation_config = {"do_sample": False, "max_new_tokens": 50}

    best_tokens_dicts_list = []
    average_logprobs_list = []
    successive_correct_outputs = 0  # For early stopping

    masks_data_list = [x["masks"] for x in input_tokenized_data_list]

    # Evaluate initial state if requested
    if eval_initial and step_metrics_path is not None:
        (
            best_tokens_dicts_list,
            average_logprobs_list,
            initial_metric,
        ) = _evaluate_universal_initial_state(
            models,
            tokenizer,
            input_tokenized_data_list,
            masks_data_list,
            true_loss_function,
            average_target_logprobs,
            true_loss_kwargs,
            debug_logger,
        )

        # Log initial metrics
        gcg_logging.log_step_metric(step_metrics_path, initial_metric)
    elif eval_initial:
        # Evaluate initial state without logging
        (
            best_tokens_dicts_list,
            average_logprobs_list,
            _,
        ) = _evaluate_universal_initial_state(
            models,
            tokenizer,
            input_tokenized_data_list,
            masks_data_list,
            true_loss_function,
            average_target_logprobs,
            true_loss_kwargs,
            debug_logger,
        )

    current_input_tokenized_data_list = input_tokenized_data_list

    # Track global maximum memory reserved across all steps
    global_max_memory_reserved = 0.0

    # Statistics for decode-reencode validation
    total_candidates_checked = 0
    total_candidates_invalid = 0

    # Create progress bar for universal optimization steps
    pbar = tqdm(
        range(max_steps),
        desc="Universal GCG Optimization",
        unit="step",
        disable=False,
    )

    step_num = 0

    for step_num in pbar:
        step_start_time = time.time()

        # Reset peak memory stats at start of each iteration
        for model in models:
            torch.cuda.reset_peak_memory_stats(device=model.device)

        # Call on_step_begin hook
        step_begin_state = on_step_begin(
            models,
            tokenizer,
            current_input_tokenized_data_list,
            universal_gcg_hyperparameters,
            debug_logger,
            step_num=step_num,
            **on_step_begin_kwargs,
        )

        # Compute signal (gradient-based top-k tokens)
        signal_start = time.time()
        best_tokens_indices = signal_function(
            models,
            tokenizer,
            current_input_tokenized_data_list,
            topk,
            debug_logger,
            step_num=step_num,
            clamp_tokens=clamp_tokens,
            ascii_only=ascii_only,
            **(signal_kwargs or {}),
        )
        signal_end = time.time()

        # Clear model gradients
        for model in models:
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Generate candidates
        candidate_gen_start = time.time()
        candidate_tensors = _generate_universal_candidates(
            best_tokens_indices,
            current_input_tokenized_data_list,
            randomness_strategy,
            substitution_validity_function,
            num_forward_evals,
            tokenizer,
            filter_tokenized_sequences=filter_tokenized_sequences,
            debug_logger=debug_logger,
        )
        candidate_gen_end = time.time()

        del best_tokens_indices
        gc.collect()
        torch.cuda.empty_cache()

        # Compute losses for all candidates
        loss_comp_start = time.time()
        true_losses = true_loss_function(
            models,
            tokenizer,
            candidate_tensors,
            masks_data_list,
            debug_logger,
            step_num=step_num,
            **(true_loss_kwargs or {}),
        )
        loss_comp_end = time.time()

        # Apply decode-reencode filter AFTER loss computation
        if filter_tokenized_sequences:
            num_checked, num_invalid = _apply_universal_decode_reencode_filter(
                candidate_tensors,
                tokenizer,
                true_losses,
                debug_logger,
                step_num,
                rejection_threshold=decode_reencode_rejection_threshold,
            )
            total_candidates_checked += num_checked
            total_candidates_invalid += num_invalid

        # Select best candidate
        best_idx = torch.argmin(true_losses)
        best_loss = true_losses[best_idx].item()

        # Extract best tokens
        best_tokens_dict = _extract_universal_best_tokens(
            candidate_tensors, best_idx, masks_data_list
        )
        best_tokens_dicts_list.append(best_tokens_dict)

        # Compute average logprobs for best candidate
        average_logprobs = average_target_logprobs(
            models,
            tokenizer,
            [torch.unsqueeze(x[best_idx], 0) for x in candidate_tensors],
            masks_data_list,
            debug_logger,
        )
        average_logprobs_value = average_logprobs.item()
        average_logprobs_list.append(average_logprobs_value)

        # Update current tokens with best candidate
        current_input_tokenized_data_list = attack_utility.update_all_tokens(
            best_tokens_dict, current_input_tokenized_data_list
        )

        # Compute argmax match fraction (always compute for progress bar)
        argmax_match = _compute_universal_argmax_match(
            models, tokenizer, current_input_tokenized_data_list
        )

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{best_loss:.4f}",
                "Avg_Loss": f"{average_logprobs_value:.4f}",
                "Best": f"{min(average_logprobs_list):.4f}",
                "Argmax": f"{argmax_match:.0%}",
            }
        )

        # Compute and save metrics if enabled
        if step_num % metrics_every_n_steps == 0 and step_metrics_path is not None:
            step_metric = _compute_universal_step_metrics(
                best_loss,
                average_logprobs_value,
                argmax_match,
                step_num,
                step_start_time,
                models,
            )

            # Update global max memory reserved
            if "max_memory_reserved" in step_metric:
                global_max_memory_reserved = max(
                    global_max_memory_reserved, step_metric["max_memory_reserved"]
                )

            # Log step metrics
            gcg_logging.log_step_metric(step_metrics_path, step_metric)

        # Check early stopping if enabled (using argmax match)
        if early_stop and step_num % metrics_every_n_steps == 0:
            should_stop, successive_correct_outputs = _check_universal_early_stopping(
                argmax_match,
                argmax_match_threshold,
                successive_correct_outputs,
                identical_outputs_before_stop,
                debug_logger,
                step_num,
            )

            # Update progress bar with success count
            pbar.set_postfix(
                {
                    "Loss": f"{best_loss:.4f}",
                    "Avg_Loss": f"{average_logprobs_value:.4f}",
                    "Best": f"{min(average_logprobs_list):.4f}",
                    "Argmax": f"{argmax_match:.0%}",
                    "Success": f"{successive_correct_outputs}",
                }
            )

            if should_stop:
                debug_logger.info(
                    f"Early stopping triggered at step {step_num}: "
                    f"argmax_match={argmax_match:.0%} >= threshold={argmax_match_threshold:.0%}"
                )
                # Log early stop in step metrics if enabled
                if step_metrics_path is not None:
                    gcg_logging.log_step_metric(
                        step_metrics_path,
                        {
                            "step": step_num,
                            "event": "early_stop",
                            "argmax_match": argmax_match,
                            "successive_correct_outputs": successive_correct_outputs,
                        },
                    )
                break

        # Call on_step_end hook
        step_end_state = on_step_end(
            models,
            tokenizer,
            current_input_tokenized_data_list,
            universal_gcg_hyperparameters,
            debug_logger,
            step_num=step_num,
            **on_step_end_kwargs,
        )

        gc.collect()
        torch.cuda.empty_cache()

    # Close progress bar
    pbar.close()

    # Evaluate the global best result
    final_metrics = _evaluate_universal_final_best(
        models,
        tokenizer,
        best_tokens_dicts_list,
        average_logprobs_list,
        input_tokenized_data_list,
        generation_config,
        debug_logger,
        timestamp_start,
        global_max_memory_reserved,
        step_num + 1,
    )

    # Log decode-reencode validation statistics
    if filter_tokenized_sequences and total_candidates_checked > 0:
        invalid_rate = (total_candidates_invalid / total_candidates_checked) * 100
        debug_logger.info(
            f"Decode-reencode validation: checked={total_candidates_checked}, "
            f"invalid={total_candidates_invalid}, rate={invalid_rate:.2f}%"
        )

    # Log final summary
    debug_logger.info(f"Universal GCG optimization completed in {step_num + 1} steps")
    debug_logger.info(
        f"Best average logprobs: {min(average_logprobs_list) if average_logprobs_list else 'N/A':.4f}"
    )
    debug_logger.info(f"Total runtime: {time.time() - timestamp_start:.2f}s")
    if global_max_memory_reserved > 0:
        debug_logger.info(f"Max memory reserved: {global_max_memory_reserved:.2f} GB")
    debug_logger.info(
        f"Samples matching target: {final_metrics['samples_matching']}/{final_metrics['total_samples']}"
    )

    # Log final metrics to step metrics file if enabled
    if step_metrics_path is not None:
        gcg_logging.log_step_metric(
            step_metrics_path,
            {
                "step": -2,  # -2 indicates final evaluation
                "event": "final_evaluation",
                **{k: v for k, v in final_metrics.items() if k != "sample_results"},
            },
        )

    return best_tokens_dicts_list, average_logprobs_list, final_metrics
