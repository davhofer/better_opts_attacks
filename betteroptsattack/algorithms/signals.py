import torch
import transformers
import typing
from betteroptsattack.utils import attack_utility as attack_utility
import logging

GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS


def og_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
    gcg_topk: int,
    debug_logger: logging.Logger,
    *,
    step_num,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    **kwargs,
):
    optim_mask: torch.Tensor = masks_data["optim_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]

    # Get vocabulary size from embedding layer (modern approach)
    vocab_size = model.get_input_embeddings().weight.shape[0]

    # Check if any tokens exceed vocab_size and clamp if requested
    max_token_id = input_points.max().item()
    if max_token_id >= vocab_size:
        if clamp_tokens:
            debug_logger.warning(
                f"Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping tokens (step={step_num})"
            )
            # Clamp tokens to valid range
            input_points = input_points.clamp(max=vocab_size - 1)
        else:
            debug_logger.error(
                f"Token ID {max_token_id} exceeds vocab size {vocab_size}, but clamping is disabled (step={step_num})"
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
        debug_logger.warning(f"NaN or Inf detected in logits (step={step_num})")
        # Return random indices as fallback
        return torch.stack(
            [torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])]
        )

    loss_tensor = GCG_LOSS_FUNCTION(
        logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)
    ).sum()

    # Add NaN check for loss
    if torch.isnan(loss_tensor).item():
        debug_logger.warning(f"NaN detected in loss (step={step_num})")
        # Return random indices as fallback
        return torch.stack(
            [torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])]
        )

    loss_tensor.backward()

    # Add NaN check for gradients
    if one_hot_tensor.grad is None or torch.isnan(one_hot_tensor.grad).any():
        debug_logger.warning(f"NaN detected in gradients (step={step_num})")
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
    debug_logger: logging.Logger,
    *,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    step_num: int = 0,
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
            debug_logger.warning(
                f"Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping tokens (step={step_num})"
            )
            # Clamp tokens to valid range
            input_points = input_points.clamp(max=vocab_size - 1)
        else:
            debug_logger.error(
                f"Token ID {max_token_id} exceeds vocab size {vocab_size}, but clamping is disabled (step={step_num})"
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
    debug_logger: logging.Logger,
    *,
    step_num: int = 0,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    **kwargs,
):
    optim_mask: torch.tensor = masks_data["optim_mask"]

    # Get vocabulary size from embedding layer (modern approach)
    vocab_size = model.get_input_embeddings().weight.shape[0]

    # Note: rand_gcg_signal doesn't use input_points, so no need to check for token clamping
    # This is a random signal function that samples from valid tokens only

    # Create a mask for valid tokens (start with all tokens valid)
    device = model.device
    valid_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

    # Always exclude special tokens from being selected
    special_toks = attack_utility.get_special_toks(tokenizer, device=device)
    if len(special_toks) > 0:
        valid_mask[special_toks] = False

    # Apply ASCII-only filtering if requested (in addition to special token filtering)
    if ascii_only:
        nonascii_toks = attack_utility.get_nonascii_toks(tokenizer, device=device)
        valid_mask[nonascii_toks] = False

    # Get indices of valid tokens
    valid_indices = torch.where(valid_mask)[0]

    # Check if we have enough valid tokens
    actual_topk = min(gcg_topk, len(valid_indices))
    if actual_topk < gcg_topk:
        debug_logger.warning(
            f"Step {step_num}: Only {actual_topk} valid tokens available (requested {gcg_topk})"
        )

    # Generate random indices by randomly permuting valid token indices
    best_tokens_indices = torch.stack(
        [
            valid_indices[
                torch.randperm(len(valid_indices), device=device)[:actual_topk]
            ]
            for _ in range(optim_mask.shape[0])
        ]
    )

    return best_tokens_indices


def universal_rand_gcg_signal(
    models,
    tokenizer,
    input_tokenized_data_list,
    gcg_topk,
    debug_logger,
    *,
    step_num: int = 0,
    clamp_tokens: bool = True,
    ascii_only: bool = False,
    **kwargs,
):
    optim_mask = input_tokenized_data_list[0]["masks"]["optim_mask"]

    # Get vocabulary size from embedding layer of first model (modern approach)
    vocab_size = models[0].get_input_embeddings().weight.shape[0]

    # Note: universal_rand_gcg_signal doesn't use input_points, so no need to check for token clamping
    # This is a random signal function that samples from valid tokens only

    # Create a mask for valid tokens (start with all tokens valid)
    device = models[0].device
    valid_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

    # Always exclude special tokens from being selected
    special_toks = attack_utility.get_special_toks(tokenizer, device=device)
    if len(special_toks) > 0:
        valid_mask[special_toks] = False

    # Apply ASCII-only filtering if requested (in addition to special token filtering)
    if ascii_only:
        nonascii_toks = attack_utility.get_nonascii_toks(tokenizer, device=device)
        valid_mask[nonascii_toks] = False

    # Get indices of valid tokens
    valid_indices = torch.where(valid_mask)[0]

    # Check if we have enough valid tokens
    actual_topk = min(gcg_topk, len(valid_indices))
    if actual_topk < gcg_topk:
        debug_logger.warning(
            f"Step {step_num}: Only {actual_topk} valid tokens available (requested {gcg_topk})"
        )

    # Generate random indices by randomly permuting valid token indices
    best_tokens_indices = torch.stack(
        [
            valid_indices[
                torch.randperm(len(valid_indices), device=device)[:actual_topk]
            ]
            for _ in range(optim_mask.shape[0])
        ]
    )

    return best_tokens_indices


def average_target_logprobs_signal(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    gcg_topk: int,
    debug_logger: logging.Logger,
    *,
    step_num,
    canonical_device_idx=0,
    normalize_grads_before_accumulation=True,
    ascii_only: bool = False,
    clamp_tokens: bool = True,
    **kwargs,
):
    """
    Compute averaged gradients across multiple samples/models for universal GCG.

    Args:
        models: List of models
        tokenizer: Tokenizer
        input_tokenized_data_list: List of input tokenized data
        gcg_topk: Number of top-k tokens to return
        debug_logger: Logger instance
        step_num: Current step number
        canonical_device_idx: Device to move gradients to for averaging
        normalize_grads_before_accumulation: Whether to normalize gradients before averaging
        ascii_only: Whether to filter non-ASCII tokens
        clamp_tokens: Whether to clamp token IDs to valid range

    Returns:
        Tensor of top-k token indices (shape: [optim_positions, topk])
    """
    num_elements_per_batch = len(input_tokenized_data_list) // len(models)
    input_tokenized_data_list_batches = [
        input_tokenized_data_list[
            x * num_elements_per_batch : (x + 1) * num_elements_per_batch
        ]
        for x in range(len(models))
    ]

    # Get vocabulary size from first model
    vocab_size = models[0].get_input_embeddings().weight.shape[0]

    grads_list = []
    skipped_samples = 0

    for model, input_tokenized_data_list_batch in zip(
        models, input_tokenized_data_list_batches
    ):
        grads_list_batch = []
        for sample_idx, input_tokenized_data in enumerate(
            input_tokenized_data_list_batch
        ):
            input_points = input_tokenized_data["tokens"]
            masks_data = input_tokenized_data["masks"]

            optim_mask: torch.Tensor = masks_data["optim_mask"]
            target_mask: torch.Tensor = masks_data["target_mask"]

            # Check if any tokens exceed vocab_size and clamp if requested
            max_token_id = input_points.max().item()
            if max_token_id >= vocab_size:
                if clamp_tokens:
                    debug_logger.warning(
                        f"Step {step_num}, Sample {sample_idx}: Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping tokens"
                    )
                    # Clamp tokens to valid range
                    input_points = input_points.clamp(max=vocab_size - 1)
                else:
                    debug_logger.error(
                        f"Step {step_num}, Sample {sample_idx}: Token ID {max_token_id} exceeds vocab size {vocab_size}, but clamping is disabled. Skipping sample."
                    )
                    skipped_samples += 1
                    continue

            one_hot_tensor = torch.nn.functional.one_hot(
                input_points.clone().detach(), num_classes=vocab_size
            ).to(dtype=model.dtype)
            one_hot_tensor.requires_grad_()
            embedding_tensor = model.get_input_embeddings().weight
            inputs_embeds = torch.unsqueeze(
                one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0
            )

            # Forward pass with NaN/Inf checking
            logits = model(inputs_embeds=inputs_embeds).logits

            # Check for NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                debug_logger.warning(
                    f"Step {step_num}, Sample {sample_idx}: NaN or Inf detected in logits. Skipping sample."
                )
                skipped_samples += 1
                continue

            loss_tensor = GCG_LOSS_FUNCTION(
                logits[0, target_mask - 1, :],
                input_points[target_mask].to(logits.device),
            ).sum()

            # Check for NaN in loss
            if torch.isnan(loss_tensor).item():
                debug_logger.warning(
                    f"Step {step_num}, Sample {sample_idx}: NaN detected in loss. Skipping sample."
                )
                skipped_samples += 1
                continue

            loss_tensor.backward()

            # Check for NaN in gradients
            if one_hot_tensor.grad is None or torch.isnan(one_hot_tensor.grad).any():
                debug_logger.warning(
                    f"Step {step_num}, Sample {sample_idx}: NaN detected in gradients. Skipping sample."
                )
                skipped_samples += 1
                continue

            # Extract gradients for optimization positions
            grad_optims = one_hot_tensor.grad[optim_mask, :]

            if normalize_grads_before_accumulation:
                # Normalize gradients with safety check
                grad_norm = grad_optims.norm(dim=-1, keepdim=True)
                # Avoid division by zero
                grad_norm = torch.clamp(grad_norm, min=1e-10)
                normalized_grad = grad_optims / grad_norm
                grads_list_batch.append(normalized_grad)
            else:
                grads_list_batch.append(grad_optims)

        if len(grads_list_batch) > 0:
            grads_list.append(torch.stack(grads_list_batch))

    # Check if we have any valid gradients
    if len(grads_list) == 0:
        debug_logger.error(
            f"Step {step_num}: All samples were skipped due to errors. Returning random tokens."
        )
        # Return random valid token indices as fallback
        return torch.stack(
            [
                torch.randperm(vocab_size)[:gcg_topk]
                for _ in range(
                    input_tokenized_data_list[0]["masks"]["optim_mask"].shape[0]
                )
            ]
        )

    if skipped_samples > 0:
        debug_logger.warning(
            f"Step {step_num}: Skipped {skipped_samples} samples due to errors"
        )

    # Move gradients to canonical device and average
    device_moved_grad_list = []
    for grads_list_batch_tensor in grads_list:
        device_moved_grad_list.append(
            grads_list_batch_tensor.to(f"cuda:{canonical_device_idx}")
        )

    final_grads = -torch.cat(device_moved_grad_list, dim=0).mean(dim=0)

    # Check for NaN/Inf in final averaged gradients
    if torch.isnan(final_grads).any() or torch.isinf(final_grads).any():
        debug_logger.error(
            f"Step {step_num}: NaN or Inf in final averaged gradients. Returning random tokens."
        )
        return torch.stack(
            [torch.randperm(vocab_size)[:gcg_topk] for _ in range(final_grads.shape[0])]
        )

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
