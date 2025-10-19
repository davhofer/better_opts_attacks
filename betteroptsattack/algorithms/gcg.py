import torch
import transformers
import typing
import numpy as np
from betteroptsattack.utils import attack_utility as attack_utility
import random
from betteroptsattack.utils import experiment_logger as experiment_logger
import gc
import json
import time
from pathlib import Path
from tqdm import tqdm


GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS


def check_argmax_match(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    current_tokens: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
    target_tokens: torch.Tensor,
) -> bool:
    """Check if argmax of logits matches target tokens."""
    with torch.no_grad():
        # Get logits for the current tokens
        logits = model(current_tokens.unsqueeze(0).to(model.device)).logits[0]

        # Get predictions at target positions (shift by 1 for causal LM)
        target_mask = masks_data["target_mask"]
        pred_logits = logits[target_mask - 1]

        # Get argmax predictions
        predictions = torch.argmax(pred_logits, dim=-1)

        # Check if they match target
        return torch.all(predictions.cpu() == target_tokens.cpu()).item()


def check_generation_starts_with_target(
    generated_text: str,
    target_tokens: torch.Tensor,
    tokenizer: transformers.AutoTokenizer,
) -> bool:
    """Check if generation starts with the target text (after stripping whitespace)."""
    target_text = tokenizer.decode(target_tokens)
    return generated_text.strip().startswith(target_text)


def check_generation_equals_target_exactly(
    generated_text: str,
    target_tokens: torch.Tensor,
    tokenizer: transformers.AutoTokenizer,
) -> bool:
    """Check if generation equals the target text exactly (after stripping whitespace)."""
    target_text = tokenizer.decode(target_tokens)
    return generated_text.strip() == target_text.strip()


def extract_adversarial_strings_context_aware(
    tokenizer: transformers.AutoTokenizer,
    best_tokens: torch.Tensor,
    initial_tokens: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
) -> typing.Tuple[str, str]:
    """
    Extract adversarial prefix and suffix from optimized tokens using full context.

    This function decodes the full token sequence and extracts the adversarial
    strings by using the known positions from masks. This avoids tokenization
    boundary issues that occur when decoding tokens in isolation.

    Args:
        tokenizer: HuggingFace tokenizer
        best_tokens: The optimized token sequence
        initial_tokens: The initial token sequence (for finding fixed parts)
        masks_data: Dictionary containing the masks

    Returns:
        tuple: (optimized_prefix, optimized_suffix) as strings
    """
    # Decode the full optimized sequence
    full_optimized_text = tokenizer.decode(best_tokens, skip_special_tokens=False)

    # Get masks
    prefix_mask = masks_data.get("prefix_mask")
    suffix_mask = masks_data.get("suffix_mask")
    payload_mask = masks_data.get("payload_mask")

    # Find the continuous range for prefix
    if prefix_mask is not None and len(prefix_mask) > 0:
        prefix_start_idx = prefix_mask.min().item()
        prefix_end_idx = prefix_mask.max().item() + 1

        # Decode tokens from beginning up to prefix start to get context before
        if prefix_start_idx > 0:
            text_before_prefix = tokenizer.decode(
                best_tokens[:prefix_start_idx], skip_special_tokens=False
            )
        else:
            text_before_prefix = ""

        # Decode tokens from prefix start to end of sequence
        text_from_prefix_start = tokenizer.decode(
            best_tokens[prefix_start_idx:], skip_special_tokens=False
        )

        # The prefix is the part of text_from_prefix_start up to where payload starts
        if payload_mask is not None and len(payload_mask) > 0:
            payload_start_idx = payload_mask.min().item()
            # Decode from prefix_end to payload_start to find the separator
            text_between = tokenizer.decode(
                best_tokens[prefix_end_idx:payload_start_idx], skip_special_tokens=False
            )
            # Decode just the prefix range
            prefix_text = tokenizer.decode(
                best_tokens[prefix_start_idx:prefix_end_idx], skip_special_tokens=False
            )
        else:
            prefix_text = tokenizer.decode(
                best_tokens[prefix_mask], skip_special_tokens=False
            )
    else:
        prefix_text = ""
        text_before_prefix = ""

    # Find the continuous range for suffix
    if suffix_mask is not None and len(suffix_mask) > 0:
        suffix_start_idx = suffix_mask.min().item()
        suffix_end_idx = suffix_mask.max().item() + 1

        # Decode from suffix to get the actual suffix text in context
        text_from_suffix = tokenizer.decode(
            best_tokens[suffix_start_idx:], skip_special_tokens=False
        )

        # Find where suffix ends (before target or end of sequence)
        target_mask = masks_data.get("target_mask")
        if target_mask is not None and len(target_mask) > 0:
            target_start_idx = target_mask.min().item()
            # Decode between suffix and target
            text_after_suffix = tokenizer.decode(
                best_tokens[suffix_end_idx:target_start_idx], skip_special_tokens=False
            )
            # The suffix is from suffix_start to suffix_end
            suffix_text = tokenizer.decode(
                best_tokens[suffix_start_idx:suffix_end_idx], skip_special_tokens=False
            )
        else:
            suffix_text = tokenizer.decode(
                best_tokens[suffix_mask], skip_special_tokens=False
            )
    else:
        suffix_text = ""

    # Return the extracted strings
    return prefix_text, suffix_text


def extract_full_injection_string(
    tokenizer: transformers.AutoTokenizer,
    best_tokens: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
) -> str:
    """
    Extract the complete injection string (prefix + payload + suffix) from optimized tokens.

    Args:
        tokenizer: HuggingFace tokenizer
        best_tokens: The optimized token sequence
        masks_data: Dictionary containing the masks

    Returns:
        str: The complete injection string
    """
    prefix_mask = masks_data.get("prefix_mask")
    suffix_mask = masks_data.get("suffix_mask")
    payload_mask = masks_data.get("payload_mask")

    # Find the range that encompasses prefix, payload, and suffix
    all_indices = []
    if prefix_mask is not None and len(prefix_mask) > 0:
        all_indices.append(prefix_mask)
    if payload_mask is not None and len(payload_mask) > 0:
        all_indices.append(payload_mask)
    if suffix_mask is not None and len(suffix_mask) > 0:
        all_indices.append(suffix_mask)

    if len(all_indices) == 0:
        return ""

    all_indices = torch.cat(all_indices)

    # Get min and max to find the contiguous range
    min_idx = all_indices.min().item()
    max_idx = all_indices.max().item() + 1

    # Extract and decode the full injection range
    injection_tokens = best_tokens[min_idx:max_idx]
    full_injection = tokenizer.decode(injection_tokens, skip_special_tokens=False)

    return full_injection


def og_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    step_num,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    debug: bool = False,
    **kwargs,
):
    optim_mask: torch.Tensor = masks_data["optim_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]

    # Debug: Print the decoded input if debug flag is set
    if debug:
        decoded_text = tokenizer.decode(input_points, skip_special_tokens=False)
        print(f"\n{'=' * 80}")
        print(f"DEBUG og_gcg_signal - Step {step_num}")
        print(f"{'=' * 80}")
        print(f"Input tokens shape: {input_points.shape}")
        print(f"Optim mask positions: {optim_mask.tolist()}")
        print(f"Target mask positions: {target_mask.tolist()}")
        print(f"{'=' * 80}")
        print(f"DECODED INPUT TEXT (last 1000 characters):")
        print(decoded_text[-1000:])
        print(f"{'=' * 80}")
        print("TOKENS SEEN DURING OPTIMIZATION:")
        for i in range(0, len(input_points), 10):
            print(input_points[i : i + 10])
        print(f"{'=' * 80}")

        # Also show what's being optimized separately
        if len(optim_mask) > 0:
            # Split optim_mask into prefix and suffix if possible
            prefix_mask = masks_data.get("prefix_mask", None)
            suffix_mask = masks_data.get("suffix_mask", None)
            if prefix_mask is not None and suffix_mask is not None:
                print(
                    f"PREFIX tokens (according to prefix mask) ({len(prefix_mask)}): {tokenizer.decode(input_points[prefix_mask])}"
                )
                print(
                    f"SUFFIX tokens (according to suffix mask) ({len(suffix_mask)}): {tokenizer.decode(input_points[suffix_mask])}"
                )
            print(
                f"OPTIMIZED tokens (according to optim mask) ({len(optim_mask)}): {tokenizer.decode(input_points[optim_mask])}"
            )

        print(f"{'=' * 80}\n")

    # Get vocabulary size from embedding layer (modern approach)
    vocab_size = model.get_input_embeddings().weight.shape[0]

    # Check if any tokens exceed vocab_size and clamp if requested
    max_token_id = input_points.max().item()
    if max_token_id >= vocab_size:
        if clamp_tokens:
            if logger:
                logger.log(
                    f"WARNING: Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping tokens",
                    event_type="warning",
                )
            # Clamp tokens to valid range
            input_points = input_points.clamp(max=vocab_size - 1)
        else:
            if logger:
                logger.log(
                    f"ERROR: Token ID {max_token_id} exceeds vocab size {vocab_size}, but clamping is disabled",
                    event_type="error",
                )
            # Return random indices as fallback
            return torch.stack(
                [
                    torch.randperm(vocab_size)[:gcg_topk]
                    for _ in range(optim_mask.shape[0])
                ]
            )

    one_hot_tensor = torch.nn.functional.one_hot(
        input_points.clone().detach(), num_classes=vocab_size
    ).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight
    inputs_embeds = torch.unsqueeze(
        one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0
    )

    # Add NaN check for logits
    logits = model(inputs_embeds=inputs_embeds).logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        if logger:
            logger.log_event("WARNING: NaN or Inf detected in logits")
        # Return random indices as fallback
        return torch.stack(
            [torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])]
        )

    loss_tensor = GCG_LOSS_FUNCTION(
        logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)
    ).sum()

    # Add NaN check for loss
    if torch.isnan(loss_tensor).item():
        if logger:
            logger.log_event("WARNING: NaN detected in loss")
        # Return random indices as fallback
        return torch.stack(
            [torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])]
        )

    loss_tensor.backward()

    # Add NaN check for gradients
    if one_hot_tensor.grad is None or torch.isnan(one_hot_tensor.grad).any():
        if logger:
            logger.log_event("WARNING: NaN detected in gradients")
        # Return random indices as fallback
        return torch.stack(
            [torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])]
        )

    grad_optims = -(one_hot_tensor.grad[optim_mask, :])

    # Always exclude special tokens from being selected
    special_toks = attack_utility.get_special_toks(tokenizer, device=grad_optims.device)
    if len(special_toks) > 0:
        grad_optims[:, special_toks] = float("-inf")

    # Apply ASCII-only filtering if requested (in addition to special token filtering)
    if ascii_only:
        nonascii_toks = attack_utility.get_nonascii_toks(
            tokenizer, device=grad_optims.device
        )
        # Set gradients for non-ASCII tokens to -inf so they won't be selected
        grad_optims[:, nonascii_toks] = float("-inf")

    best_tokens_indices = grad_optims.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices


def neg_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    **kwargs,
):
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]

    # Get vocabulary size from embedding layer (modern approach)
    vocab_size = model.get_input_embeddings().weight.shape[0]

    # Check if any tokens exceed vocab_size and clamp if requested
    max_token_id = input_points.max().item()
    if max_token_id >= vocab_size:
        if clamp_tokens:
            if logger:
                logger.log(
                    f"WARNING: Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping tokens",
                    event_type="warning",
                )
            # Clamp tokens to valid range
            input_points = input_points.clamp(max=vocab_size - 1)
        else:
            if logger:
                logger.log(
                    f"ERROR: Token ID {max_token_id} exceeds vocab size {vocab_size}, but clamping is disabled",
                    event_type="error",
                )
            # Return random indices as fallback
            return torch.stack(
                [
                    torch.randperm(vocab_size)[:gcg_topk]
                    for _ in range(optim_mask.shape[0])
                ]
            )

    one_hot_tensor = torch.nn.functional.one_hot(
        input_points.clone().detach(), num_classes=vocab_size
    ).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight
    inputs_embeds = torch.unsqueeze(
        one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0
    )
    logits = model(inputs_embeds=inputs_embeds).logits
    loss_tensor = GCG_LOSS_FUNCTION(
        logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)
    ).sum()
    loss_tensor.backward()
    grad_optims = one_hot_tensor.grad[optim_mask, :]

    # Always exclude special tokens from being selected
    special_toks = attack_utility.get_special_toks(tokenizer, device=grad_optims.device)
    if len(special_toks) > 0:
        grad_optims[:, special_toks] = float("-inf")

    # Apply ASCII-only filtering if requested (in addition to special token filtering)
    if ascii_only:
        nonascii_toks = attack_utility.get_nonascii_toks(
            tokenizer, device=grad_optims.device
        )
        # Set gradients for non-ASCII tokens to -inf so they won't be selected
        grad_optims[:, nonascii_toks] = float("-inf")

    best_tokens_indices = grad_optims.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices


def rand_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    clamp_tokens: bool = True,
    **kwargs,
):
    optim_mask: torch.tensor = masks_data["optim_mask"]

    # Get vocabulary size from embedding layer (modern approach)
    vocab_size = model.get_input_embeddings().weight.shape[0]

    # Note: rand_gcg_signal doesn't use input_points, so no need to check for token clamping
    # This is a random signal function
    best_tokens_indices = torch.stack(
        [torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])]
    )
    return best_tokens_indices


def universal_rand_gcg_signal(
    models,
    tokenizer,
    input_tokenized_data_list,
    gcg_topk,
    logger,
    *,
    clamp_tokens: bool = True,
    **kwargs,
):
    optim_mask = input_tokenized_data_list[0]["masks"]["optim_mask"]

    # Get vocabulary size from embedding layer of first model (modern approach)
    vocab_size = models[0].get_input_embeddings().weight.shape[0]

    # Note: universal_rand_gcg_signal doesn't use input_points, so no need to check for token clamping
    # This is a random signal function
    best_tokens_indices = torch.stack(
        [torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])]
    )
    return best_tokens_indices


@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def custom_gcg(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Dict,
    custom_gcg_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    early_stop,
    identical_outputs_before_stop,
    generation_config,
    to_cache_logits,
    to_cache_attentions,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    # Metrics parameters
    compute_metrics: bool = False,
    metrics_every_n_steps: int = 1,
    save_metrics_path: typing.Optional[str] = None,
    save_adv_string_every_n_steps: int = 25,
    # Decode-reencode validation
    filter_tokenized_sequences: bool = False,
    # Exact target only mode: optimize to make model produce ONLY the target string followed by EOS token
    # (instead of just starting with the target string)
    exact_target_only: bool = False,
    # Debug mode: print decoded text at each optimization step
    debug_mode: bool = False,
):
    logger.log(input_tokenized_data)

    # Validate flags
    if early_stop and not compute_metrics:
        raise ValueError("early_stop requires compute_metrics=True")

    # Setup metrics collection if enabled
    per_step_metrics = []
    if compute_metrics:
        if save_metrics_path is None:
            raise ValueError("save_metrics_path must be provided when compute_metrics=True")
        metrics_file = Path(save_metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

    if to_cache_logits:
        target_logprobs = attack_utility.CachedTargetLogprobs(to_cache=True)
    else:
        target_logprobs = attack_utility.target_logprobs

    if to_cache_attentions:
        att_cacher = attack_utility.CachedBulkForward(to_cache=True)
    else:
        att_cacher = None

    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    eval_input_mask: torch.tensor = masks_data["input_mask"]

    # Handle exact_target_only mode: extend target to include EOS token
    if exact_target_only:
        # Store original target tokens (without EOS)
        original_target_tokens = input_tokens[target_mask].clone()

        # Extend target mask to include EOS position
        eos_position = target_mask[-1] + 1
        target_mask_extended = torch.cat(
            [
                target_mask,
                torch.tensor(
                    [eos_position], device=target_mask.device, dtype=target_mask.dtype
                ),
            ]
        )

        # Extend input_tokens to include EOS at the appropriate position
        # We need to insert EOS token at the position after the last target token
        # First, ensure input_tokens is long enough
        if eos_position >= len(input_tokens):
            # Pad input_tokens if necessary (shouldn't normally happen)
            padding_needed = eos_position - len(input_tokens) + 1
            input_tokens = torch.cat(
                [
                    input_tokens,
                    torch.full(
                        (padding_needed,),
                        tokenizer.pad_token_id or 0,
                        device=input_tokens.device,
                        dtype=input_tokens.dtype,
                    ),
                ]
            )

        # Set the EOS token at the appropriate position
        input_tokens[eos_position] = tokenizer.eos_token_id

        # Update the target mask to use the extended version
        target_mask = target_mask_extended

        logger.log(
            f"exact_target_only enabled: Extended target to include EOS token at position {eos_position}",
            event_type="info",
        )
    else:
        original_target_tokens = input_tokens[target_mask].clone()

    signal_function = custom_gcg_hyperparams.get("signal_function", og_gcg_signal)
    true_loss_function = custom_gcg_hyperparams.get(
        "true_loss_function", target_logprobs
    )
    substitution_validity_function = custom_gcg_hyperparams.get(
        "substitution_validity_function", None
    )
    signal_kwargs = custom_gcg_hyperparams.get("signal_kwargs", None)
    true_loss_kwargs = custom_gcg_hyperparams.get("true_loss_kwargs", None)

    current_best_tokens = input_tokens.clone()
    best_output_sequences = []
    logprobs_sequences = []
    successive_correct_outputs = 0

    # Statistics for decode-reencode validation
    total_candidates_checked = 0
    total_candidates_invalid = 0

    if true_loss_kwargs is None:
        true_loss_kwargs = {}
    true_loss_kwargs["att_cacher"] = att_cacher

    # Evaluate initial state if metrics are enabled
    if compute_metrics:
        step_start_time = time.time()
        initial_true_loss = true_loss_function(
            model,
            tokenizer,
            torch.unsqueeze(current_best_tokens, 0),
            masks_data,
            input_tokens[target_mask],
            logger,
            **true_loss_kwargs,
        )
        logger.log(initial_true_loss, step_num=-1)
        best_output_sequences.append(current_best_tokens.clone())
        logger.log(current_best_tokens, step_num=-1)
        initial_logprobs = target_logprobs(
            model,
            tokenizer,
            torch.unsqueeze(current_best_tokens, 0),
            masks_data,
            input_tokens[target_mask],
            logger,
        )
        initial_logprobs = initial_logprobs.item()
        logger.log(initial_logprobs, step_num=-1)
        logprobs_sequences.append(initial_logprobs)
        input_tokens_for_generation = current_best_tokens[eval_input_mask]

        print(f"{'=' * 80}")
        print("TOKENS SEEN DURING OPTIMIZATION (AT GENERATION EVAL):")
        toks = input_tokens_for_generation.tolist()
        for i in range(0, len(toks), 10):
            print(toks[i : i + 10])
        print(f"{'=' * 80}")

        generated_output_tokens = model.generate(
            torch.unsqueeze(input_tokens_for_generation, dim=0).to(model.device),
            attention_mask=torch.unsqueeze(
                torch.ones(input_tokens_for_generation.shape), dim=0
            ).to(model.device),
            **generation_config,
        )
        # Get the actual number of input tokens used for generation
        input_length = len(input_tokens_for_generation)
        generated_output_string = tokenizer.batch_decode(
            generated_output_tokens[:, input_length:]
        )[0]
        logger.log(generated_output_string, step_num=-1)

        # Compute metrics for initial state
        target_tokens = input_tokens[target_mask]
        argmax_matches = check_argmax_match(
            model, tokenizer, current_best_tokens, masks_data, target_tokens
        )
        starts_with_target = check_generation_starts_with_target(
            generated_output_string, target_tokens, tokenizer
        )

        initial_metric = {
            "step": -1,
            "loss": initial_logprobs,
            "argmax_matches_target": argmax_matches,
            "generation_starts_with_target": starts_with_target,
            "generated_text": generated_output_string[:100],
            "time_elapsed": time.time() - step_start_time,
        }
        per_step_metrics.append(initial_metric)
        with open(metrics_file, "w") as f:
            f.write(json.dumps(initial_metric) + "\n")

    step_num = 0

    best_tokens_chunk = []
    true_losses_chunk = []
    substitution_data_chunk = []
    current_best_true_loss_chunk = []
    current_best_tokens_chunk = []
    logprobs_chunk = []

    # Create progress bar for optimization steps
    pbar = tqdm(
        range(custom_gcg_hyperparams["max_steps"]),
        desc="GCG Optimization",
        unit="step",
        disable=False,
    )

    for step_num in pbar:
        step_start_time = time.time()

        # Add debug flag to signal_kwargs if debug mode is enabled
        current_signal_kwargs = signal_kwargs or {}
        if debug_mode:
            current_signal_kwargs["debug"] = True

        best_tokens_indices = signal_function(
            model,
            tokenizer,
            current_best_tokens,
            masks_data,
            custom_gcg_hyperparams["topk"],
            logger,
            step_num=step_num,
            clamp_tokens=clamp_tokens,
            ascii_only=ascii_only,
            **current_signal_kwargs,
        )

        indices_to_sample = set()
        indices_to_exclude = set()
        substitutions_set = set()

        if isinstance(custom_gcg_hyperparams["forward_eval_candidates"], str):
            if custom_gcg_hyperparams["forward_eval_candidates"] == "all":
                for first_coordinate in range(best_tokens_indices.shape[0]):
                    for second_coordinate in range(best_tokens_indices.shape[1]):
                        substitution_make = current_best_tokens.clone()
                        substitution_make[optim_mask[first_coordinate]] = (
                            best_tokens_indices[(first_coordinate, second_coordinate)]
                        )
                        substitutions_set.add(substitution_make)
                substitution_data = torch.stack(list(substitutions_set))
        else:
            assert isinstance(custom_gcg_hyperparams["forward_eval_candidates"], int), (
                "Only strings or ints"
            )
            num_forward_evals = custom_gcg_hyperparams["forward_eval_candidates"]
            while len(indices_to_sample) < num_forward_evals:
                first_coordinate = (
                    torch.randint(0, best_tokens_indices.shape[0], (1,))
                    .to(torch.int32)
                    .item()
                )
                second_coordinate = (
                    torch.randint(0, best_tokens_indices.shape[1], (1,))
                    .to(torch.int32)
                    .item()
                )
                if (first_coordinate, second_coordinate) in indices_to_sample:
                    continue
                if (first_coordinate, second_coordinate) in indices_to_exclude:
                    continue
                random_substitution_make = current_best_tokens.clone()
                random_substitution_make[optim_mask[first_coordinate]] = (
                    best_tokens_indices[(first_coordinate, second_coordinate)]
                )
                if (substitution_validity_function is None) or (
                    substitution_validity_function(
                        random_substitution_make,
                        tokenizer=tokenizer,
                        masks_data=masks_data,
                    )
                ):
                    indices_to_sample.add((first_coordinate, second_coordinate))
                    substitutions_set.add(random_substitution_make)
                else:
                    # SUBSTITUTION_INVALID_STRING = "substitution_invalid"
                    # logger.log(SUBSTITUTION_INVALID_STRING)
                    indices_to_exclude.add((first_coordinate, second_coordinate))
            substitution_data = torch.stack(list(substitutions_set))

        del best_tokens_indices
        gc.collect()
        torch.cuda.empty_cache()
        substitution_data_chunk.append(substitution_data)

        true_losses = true_loss_function(
            model,
            tokenizer,
            substitution_data,
            masks_data,
            input_tokens[target_mask],
            logger,
            **true_loss_kwargs,
        )

        # Decode-reencode validation: filter out candidates that change during tokenization cycle
        if filter_tokenized_sequences:
            valid_mask = torch.ones(len(substitution_data), dtype=torch.bool)

            for idx, candidate_tokens in enumerate(substitution_data):
                # Decode full sequence
                decoded_text = tokenizer.decode(
                    candidate_tokens.cpu(), skip_special_tokens=False
                )

                # Re-encode
                reencoded_tokens = tokenizer.encode(
                    decoded_text, return_tensors="pt", add_special_tokens=False
                )[0]

                # Check if tokenization is preserved
                if not torch.equal(candidate_tokens.cpu(), reencoded_tokens.cpu()):
                    valid_mask[idx] = False
                    # Set loss to infinity so it won't be selected
                    true_losses[idx] = float("inf")

            # Update statistics
            num_invalid = (~valid_mask).sum().item()
            total_candidates_checked += len(substitution_data)
            total_candidates_invalid += num_invalid

            if num_invalid > 0:
                logger.log(
                    f"Step {step_num}: Filtered {num_invalid}/{len(substitution_data)} candidates due to decode-reencode mismatch",
                    step_num=step_num,
                )

            # Check if all candidates were invalidated
            if num_invalid == len(substitution_data):
                logger.log(
                    f"WARNING Step {step_num}: ALL candidates failed decode-reencode validation! Using best of invalid candidates.",
                    step_num=step_num,
                )
                # In this case, true_losses are all inf, so we'll just use the first candidate
                # This is a fallback - ideally this shouldn't happen often

        true_losses_chunk.append(true_losses)
        current_best_true_loss = true_losses[torch.argmin(true_losses)]
        current_best_true_loss_chunk.append(current_best_true_loss)
        current_best_tokens = substitution_data[torch.argmin(true_losses)].clone()
        current_best_tokens_chunk.append(current_best_tokens)
        best_output_sequences.append(current_best_tokens.clone())
        logprobs = target_logprobs(
            model,
            tokenizer,
            torch.unsqueeze(current_best_tokens, 0),
            masks_data,
            input_tokens[target_mask],
            logger,
        )
        logprobs = logprobs.item()
        logprobs_chunk.append(logprobs)
        logprobs_sequences.append(logprobs)

        # Update progress bar with current loss
        pbar.set_postfix(
            {
                "Loss": f"{logprobs:.4f}",
                "Best": f"{min(logprobs_sequences):.4f}",
                "Success": f"{successive_correct_outputs}",
            }
        )

        # Compute and save metrics if enabled
        if compute_metrics and step_num % metrics_every_n_steps == 0:
            target_tokens = input_tokens[target_mask]

            # Check argmax match
            argmax_matches = check_argmax_match(
                model, tokenizer, current_best_tokens, masks_data, target_tokens
            )

            # Generate text to check if it starts with target
            with torch.no_grad():
                input_tokens_for_generation = current_best_tokens[eval_input_mask]
                generated_tokens = model.generate(
                    torch.unsqueeze(input_tokens_for_generation, dim=0).to(
                        model.device
                    ),
                    attention_mask=torch.unsqueeze(
                        torch.ones(input_tokens_for_generation.shape), dim=0
                    ).to(model.device),
                    **generation_config,
                )
                # Get the actual number of input tokens used for generation
                input_length = len(input_tokens_for_generation)
                generated_text = tokenizer.batch_decode(
                    generated_tokens[:, input_length:]
                )[0]

            starts_with_target = check_generation_starts_with_target(
                generated_text, target_tokens, tokenizer
            )

            # Prepare step metrics
            step_metric = {
                "step": step_num,
                "loss": logprobs,
                "argmax_matches_target": argmax_matches,
                "generation_starts_with_target": starts_with_target,
                "generated_text": generated_text[:100],
                "time_elapsed": time.time() - step_start_time,
            }

            # Add adversarial string periodically
            if step_num % save_adv_string_every_n_steps == 0:
                # Get prefix and suffix tokens separately
                prefix_tokens = current_best_tokens[masks_data["prefix_mask"]]
                suffix_tokens = current_best_tokens[masks_data["suffix_mask"]]
                # Decode with separator to show as it appears in the prompt
                prefix_str = tokenizer.decode(prefix_tokens)
                suffix_str = tokenizer.decode(suffix_tokens)
                step_metric["current_adv_string"] = f"{prefix_str} . {suffix_str}"

            # Early stopping logic (uses generated_text from metrics)
            # Check this BEFORE saving so we can add early_stop flag if needed
            should_stop = False
            if early_stop:
                # Choose validation function based on exact_target_only mode
                if exact_target_only:
                    # In exact_target_only mode, require exact match
                    target_match = check_generation_equals_target_exactly(
                        generated_text,
                        original_target_tokens,  # Use original target without EOS
                        tokenizer,
                    )
                else:
                    # In normal mode, check if target appears at the beginning
                    target_match = check_generation_starts_with_target(
                        generated_text,
                        target_tokens,
                        tokenizer,
                    )

                if target_match:
                    successive_correct_outputs += 1
                    if successive_correct_outputs >= identical_outputs_before_stop:
                        # Add final adversarial string if not already included
                        if "current_adv_string" not in step_metric:
                            prefix_tokens = current_best_tokens[
                                masks_data["prefix_mask"]
                            ]
                            suffix_tokens = current_best_tokens[
                                masks_data["suffix_mask"]
                            ]
                            prefix_str = tokenizer.decode(prefix_tokens)
                            suffix_str = tokenizer.decode(suffix_tokens)
                            step_metric["current_adv_string"] = (
                                f"{prefix_str} . {suffix_str}"
                            )

                        step_metric["early_stop"] = True
                        should_stop = True

                        # Update progress bar for early stopping
                        pbar.set_description("GCG Optimization (Early Stop)")
                        pbar.set_postfix(
                            {
                                "Loss": f"{logprobs:.4f}",
                                "Best": f"{min(logprobs_sequences):.4f}",
                                "Success": f"{successive_correct_outputs}",
                                "Status": "SUCCESS",
                            }
                        )
                else:
                    successive_correct_outputs = 0

            # Save metrics (now includes early_stop flag if applicable)
            per_step_metrics.append(step_metric)
            with open(metrics_file, "a") as f:
                f.write(json.dumps(step_metric) + "\n")

            # Break after saving if early stop was triggered
            if should_stop:
                break

        if (step_num + 1) % 10 == 0:
            logger.log(substitution_data_chunk, step_num=step_num)
            logger.log(true_losses_chunk, step_num=step_num)
            logger.log(current_best_true_loss_chunk, step_num=step_num)
            logger.log(current_best_tokens_chunk, step_num=step_num)
            logger.log(best_tokens_chunk, step_num=step_num)
            logger.log(logprobs_chunk, step_num=step_num)

            substitution_data_chunk = []
            true_losses_chunk = []
            current_best_true_loss_chunk = []
            current_best_tokens_chunk = []
            best_tokens_chunk = []
            logprobs_chunk = []

    # Close progress bar
    pbar.close()

    logger.log(successive_correct_outputs, num_steps=step_num)

    # Log decode-reencode validation statistics
    if filter_tokenized_sequences and total_candidates_checked > 0:
        invalid_rate = (total_candidates_invalid / total_candidates_checked) * 100
        logger.log(f"\n{'=' * 80}")
        logger.log(f"DECODE-REENCODE VALIDATION STATISTICS:")
        logger.log(f"{'=' * 80}")
        logger.log(f"Total candidates checked: {total_candidates_checked}")
        logger.log(f"Total candidates invalidated: {total_candidates_invalid}")
        logger.log(f"Invalidation rate: {invalid_rate:.2f}%")
        logger.log(f"{'=' * 80}\n")

    # Return extended results if metrics were enabled
    if compute_metrics:
        return {
            "logprobs_sequences": logprobs_sequences,
            "best_output_sequences": best_output_sequences,
            "per_step_metrics": per_step_metrics,
            "final_success": successive_correct_outputs
            >= identical_outputs_before_stop,
            "total_steps": step_num + 1,
        }
    else:
        # Maintain backward compatibility - return original format
        return logprobs_sequences, best_output_sequences


def average_target_logprobs_signal(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    step_num,
    canonical_device_idx=0,
    normalize_grads_before_accumulation=True,
    ascii_only: bool = False,
    **kwargs,
):
    num_elements_per_batch = len(input_tokenized_data_list) // len(models)
    input_tokenized_data_list_batches = [
        input_tokenized_data_list[
            x * num_elements_per_batch : (x + 1) * num_elements_per_batch
        ]
        for x in range(len(models))
    ]

    grads_list = []
    for model, input_tokenized_data_list_batch in zip(
        models, input_tokenized_data_list_batches
    ):
        grads_list_batch = []
        for input_tokenized_data in input_tokenized_data_list_batch:
            input_points = input_tokenized_data["tokens"]
            masks_data = input_tokenized_data["masks"]

            optim_mask: torch.Tensor = masks_data["optim_mask"]
            target_mask: torch.Tensor = masks_data["target_mask"]

            # Get vocabulary size from embedding layer (modern approach)
            vocab_size = model.get_input_embeddings().weight.shape[0]

            # Check if any tokens exceed vocab_size and clamp if requested
            max_token_id = input_points.max().item()
            if max_token_id >= vocab_size:
                if clamp_tokens:
                    if logger:
                        logger.log(
                            f"WARNING: Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping tokens",
                            event_type="warning",
                        )
                    # Clamp tokens to valid range
                    input_points = input_points.clamp(max=vocab_size - 1)
                else:
                    if logger:
                        logger.log(
                            f"ERROR: Token ID {max_token_id} exceeds vocab size {vocab_size}, but clamping is disabled",
                            event_type="error",
                        )
                    # Skip this input and continue with next
                    continue

            one_hot_tensor = torch.nn.functional.one_hot(
                input_points.clone().detach(), num_classes=vocab_size
            ).to(dtype=model.dtype)
            one_hot_tensor.requires_grad_()
            embedding_tensor = model.get_input_embeddings().weight
            inputs_embeds = torch.unsqueeze(
                one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0
            )
            logits = model(inputs_embeds=inputs_embeds).logits
            loss_tensor = GCG_LOSS_FUNCTION(
                logits[0, target_mask - 1, :],
                input_points[target_mask].to(logits.device),
            ).sum()
            loss_tensor.backward()
            if normalize_grads_before_accumulation:
                normalized_grad = one_hot_tensor.grad[
                    optim_mask, :
                ] / one_hot_tensor.grad[optim_mask, :].norm(dim=-1, keepdim=True)
                grads_list_batch.append(normalized_grad)
            else:
                grads_list_batch.append(one_hot_tensor.grad[optim_mask, :])
        grads_list.append(torch.stack(grads_list_batch))

    device_moved_grad_list = []
    for grads_list_batch_tensor in grads_list:
        device_moved_grad_list.append(grads_list_batch_tensor.to(canonical_device_idx))

    final_grads = -torch.cat(device_moved_grad_list, dim=0).mean(dim=0)

    # Always exclude special tokens from being selected
    special_toks = attack_utility.get_special_toks(tokenizer, device=final_grads.device)
    if len(special_toks) > 0:
        final_grads[:, special_toks] = float("-inf")

    # Apply ASCII-only filtering if requested (in addition to special token filtering)
    if ascii_only:
        nonascii_toks = attack_utility.get_nonascii_toks(
            tokenizer, device=final_grads.device
        )
        # Set gradients for non-ASCII tokens to -inf so they won't be selected
        final_grads[:, nonascii_toks] = float("-inf")

    best_tokens_indices = final_grads.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices


def DEFAULT_GCG_RANDOMNESS_STRATEGY(
    tokenizer,
    best_tokens_indices,
    input_tokenized_data_list,
    substitution_validity_function,
    max_candidate_size,
):
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

            if (substitution_validity_function is None) or (
                substitution_validity_function(
                    random_substitution_make, tokenizer=tokenizer, masks_data=masks_data
                )
            ):
                pass
            else:
                # SUBSTITUTION_INVALID_STRING = "substitution_invalid"
                # logger.log(SUBSTITUTION_INVALID_STRING)
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


@experiment_logger.log_parameters(exclude=["models", "tokenizer"])
def weakly_universal_gcg(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    universal_gcg_hyperparameters: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    eval_initial,
    generation_config,
    to_cache_logits,
    to_cache_attentions,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
):
    logger.log(input_tokenized_data_list)

    if to_cache_logits:
        average_target_logprobs = attack_utility.CachedAverageLogprobs()
    else:
        raise ValueError(f"Just cache ffs. Or write your own implementation.")

    if to_cache_attentions:
        att_cacher = None
        # att_cacher = attack_utility.CachedAverageBulkForward()
    else:
        raise ValueError(f"Just cache ffs. Or write your own implementation.")

    signal_function = universal_gcg_hyperparameters.get(
        "signal_function", average_target_logprobs_signal
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
    true_loss_kwargs["att_cacher"] = att_cacher

    best_tokens_dicts_list = []
    average_logprobs_list = []

    masks_data_list = [x["masks"] for x in input_tokenized_data_list]

    if eval_initial:
        initial_true_loss = true_loss_function(
            models,
            tokenizer,
            [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list],
            masks_data_list,
            logger,
            **true_loss_kwargs,
        )
        logger.log(initial_true_loss, step_num=-1)
        initial_average_logprobs = average_target_logprobs(
            models,
            tokenizer,
            [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list],
            masks_data_list,
            logger,
        )
        initial_average_logprobs = initial_average_logprobs.item()
        logger.log(initial_average_logprobs, step_num=-1)
        average_logprobs_list.append(initial_average_logprobs)
        best_tokens_dicts_list.append(
            attack_utility.form_best_tokens_dict(input_tokenized_data_list)
        )

    best_tokens_dicts_chunk = []
    true_losses_chunk = []
    current_best_true_loss_chunk = []
    logprobs_chunk = []

    current_input_tokenized_data_list = input_tokenized_data_list

    # Create progress bar for universal optimization steps
    pbar = tqdm(
        range(universal_gcg_hyperparameters["max_steps"]),
        desc="Universal GCG Optimization",
        unit="step",
        disable=False,
    )

    for step_num in pbar:
        step_begin_state = on_step_begin(
            models,
            tokenizer,
            current_input_tokenized_data_list,
            universal_gcg_hyperparameters,
            logger,
            step_num=step_num,
            **on_step_begin_kwargs,
        )

        best_tokens_indices = signal_function(
            models,
            tokenizer,
            current_input_tokenized_data_list,
            universal_gcg_hyperparameters["topk"],
            logger,
            step_num=step_num,
            clamp_tokens=clamp_tokens,
            ascii_only=ascii_only,
            **(signal_kwargs or {}),
        )
        forward_eval_candidates = randomness_strategy(
            tokenizer,
            best_tokens_indices,
            current_input_tokenized_data_list,
            substitution_validity_function,
            universal_gcg_hyperparameters["forward_eval_candidates"],
        )
        true_losses = true_loss_function(
            models,
            tokenizer,
            forward_eval_candidates,
            masks_data_list,
            logger,
            step_num=step_num,
            **(true_loss_kwargs or {}),
        )
        true_losses_chunk.append(true_losses)
        best_idx = torch.argmin(true_losses)
        best_loss = true_losses[best_idx]
        current_best_true_loss_chunk.append(best_loss)
        best_tokens_dict = {
            "prefix_tokens": forward_eval_candidates[0][best_idx][
                masks_data_list[0]["prefix_mask"]
            ],
            "suffix_tokens": forward_eval_candidates[0][best_idx][
                masks_data_list[0]["suffix_mask"]
            ],
        }
        best_tokens_dicts_chunk.append(best_tokens_dict)
        best_tokens_dicts_list.append(best_tokens_dict)
        average_logprobs = average_target_logprobs(
            models,
            tokenizer,
            [torch.unsqueeze(x[best_idx], 0) for x in forward_eval_candidates],
            masks_data_list,
            logger,
        )
        logprobs_chunk.append(average_logprobs.item())
        average_logprobs_list.append(average_logprobs.item())
        current_input_tokenized_data_list = attack_utility.update_all_tokens(
            best_tokens_dict, current_input_tokenized_data_list
        )

        # Update progress bar with current loss
        pbar.set_postfix(
            {
                "Loss": f"{best_loss:.4f}",
                "Avg_Loss": f"{average_logprobs.item():.4f}",
                "Best": f"{min(average_logprobs_list):.4f}",
            }
        )

        if (step_num + 1) % 10 == 0:
            logger.log(true_losses_chunk, step_num=step_num)
            logger.log(current_best_true_loss_chunk, step_num=step_num)
            logger.log(best_tokens_dicts_chunk, step_num=step_num)
            logger.log(logprobs_chunk, step_num=step_num)

            true_losses_chunk = []
            current_best_true_loss_chunk = []
            logprobs_chunk = []
            best_tokens_dicts_chunk = []

        step_end_state = on_step_end(
            models,
            tokenizer,
            current_input_tokenized_data_list,
            universal_gcg_hyperparameters,
            logger,
            step_num=step_num,
            **on_step_end_kwargs,
        )

        gc.collect()
        torch.cuda.empty_cache()

    # Close progress bar
    pbar.close()

    return best_tokens_dicts_list, average_logprobs_list
