import torch
import transformers
import typing
import numpy as np
import utils.attack_utility as attack_utility
import random
import utils.experiment_logger as experiment_logger
import gc
import json
import time
from pathlib import Path


GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS


def check_argmax_match(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    current_tokens: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
    target_tokens: torch.Tensor
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
    tokenizer: transformers.AutoTokenizer
) -> bool:
    """Check if generation starts with the target text (after stripping whitespace)."""
    target_text = tokenizer.decode(target_tokens)
    return generated_text.strip().startswith(target_text)

def og_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    step_num,
    **kwargs
):
    optim_mask: torch.Tensor = masks_data["optim_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]

    # Get vocabulary size properly
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'get_vocab'):
        vocab_size = len(tokenizer.get_vocab())
    else:
        vocab_size = len(tokenizer.vocab)
    
    # Ensure vocab size doesn't exceed embedding size
    embedding_size = model.get_input_embeddings().weight.shape[0]
    vocab_size = min(vocab_size, embedding_size)

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=vocab_size).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:vocab_size]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    
    # Add NaN check for logits
    logits = model(inputs_embeds=inputs_embeds).logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        if logger:
            logger.log_event("WARNING: NaN or Inf detected in logits")
        # Return random indices as fallback
        return torch.stack([torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])])
    
    loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).sum()
    
    # Add NaN check for loss
    if torch.isnan(loss_tensor).item():
        if logger:
            logger.log_event("WARNING: NaN detected in loss")
        # Return random indices as fallback
        return torch.stack([torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])])
    
    loss_tensor.backward()
    
    # Add NaN check for gradients
    if one_hot_tensor.grad is None or torch.isnan(one_hot_tensor.grad).any():
        if logger:
            logger.log_event("WARNING: NaN detected in gradients")
        # Return random indices as fallback
        return torch.stack([torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])])
    
    grad_optims = - (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices

def neg_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
):
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]

    # Get vocabulary size properly
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'get_vocab'):
        vocab_size = len(tokenizer.get_vocab())
    else:
        vocab_size = len(tokenizer.vocab)
    
    # Ensure vocab size doesn't exceed embedding size
    embedding_size = model.get_input_embeddings().weight.shape[0]
    vocab_size = min(vocab_size, embedding_size)

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=vocab_size).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:vocab_size]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    logits = model(inputs_embeds=inputs_embeds).logits
    loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).sum()
    loss_tensor.backward()
    grad_optims = (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices

def rand_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    **kwargs
):
    optim_mask: torch.tensor = masks_data["optim_mask"]

    # Get vocabulary size properly
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'get_vocab'):
        vocab_size = len(tokenizer.get_vocab())
    else:
        vocab_size = len(tokenizer)

    best_tokens_indices = torch.stack([torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])])
    return best_tokens_indices

def universal_rand_gcg_signal(
    models,
    tokenizer,
    input_tokenized_data_list,
    gcg_topk,
    logger,
    **kwargs
):
    optim_mask = input_tokenized_data_list[0]["masks"]["optim_mask"]

    # Get vocabulary size properly
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'get_vocab'):
        vocab_size = len(tokenizer.get_vocab())
    else:
        vocab_size = len(tokenizer)

    best_tokens_indices = torch.stack([torch.randperm(vocab_size)[:gcg_topk] for _ in range(optim_mask.shape[0])])
    return best_tokens_indices

    

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def custom_gcg(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Dict,
    custom_gcg_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    eval_every_step,
    early_stop,
    eval_initial,
    identical_outputs_before_stop,
    generation_config,
    to_cache_logits,
    to_cache_attentions,
    # Extended logging parameters
    save_metrics_path: typing.Optional[str] = None,
    check_extended_metrics_every_n_steps: int = 10,
    save_adv_string_every_n_steps: int = 25
):

    logger.log(input_tokenized_data)
    
    # Setup metrics file if path provided
    per_step_metrics = []
    if save_metrics_path:
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

    signal_function = custom_gcg_hyperparams.get("signal_function", og_gcg_signal)
    true_loss_function = custom_gcg_hyperparams.get("true_loss_function", target_logprobs)
    substitution_validity_function = custom_gcg_hyperparams.get("substitution_validity_function", None)
    signal_kwargs = custom_gcg_hyperparams.get("signal_kwargs", None)
    true_loss_kwargs = custom_gcg_hyperparams.get("true_loss_kwargs", None)

    current_best_tokens = input_tokens.clone()
    best_output_sequences = []
    logprobs_sequences = []
    successive_correct_outputs = 0

    if true_loss_kwargs is None:
        true_loss_kwargs = {}
    true_loss_kwargs["att_cacher"] = att_cacher
    if eval_initial:
        step_start_time = time.time()
        initial_true_loss = true_loss_function(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger, **true_loss_kwargs)
        logger.log(initial_true_loss, step_num=-1)
        best_output_sequences.append(current_best_tokens.clone())
        logger.log(current_best_tokens, step_num=-1)
        initial_logprobs = target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
        initial_logprobs = initial_logprobs.item()
        logger.log(initial_logprobs, step_num=-1)
        logprobs_sequences.append(initial_logprobs)
        input_tokens_for_generation = current_best_tokens[eval_input_mask]
        generated_output_tokens = model.generate(torch.unsqueeze(input_tokens_for_generation, dim=0).to(model.device), attention_mask=torch.unsqueeze(torch.ones(input_tokens_for_generation.shape), dim=0).to(model.device), **generation_config)
        # Get the actual number of input tokens used for generation
        input_length = len(input_tokens_for_generation)
        generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, input_length:])[0]
        logger.log(generated_output_string, step_num=-1)
        
        # Extended metrics for initial state
        if save_metrics_path:
            target_tokens = input_tokens[target_mask]
            argmax_matches = check_argmax_match(model, tokenizer, current_best_tokens, masks_data, target_tokens)
            starts_with_target = check_generation_starts_with_target(generated_output_string, target_tokens, tokenizer)
            
            initial_metric = {
                "step": -1,
                "loss": initial_logprobs,
                "argmax_matches_target": argmax_matches,
                "generation_starts_with_target": starts_with_target,
                "generated_text": generated_output_string[:100],
                "time_elapsed": time.time() - step_start_time
            }
            per_step_metrics.append(initial_metric)
            with open(metrics_file, 'w') as f:
                f.write(json.dumps(initial_metric) + '\n')

    step_num = 0

    best_tokens_chunk = []
    true_losses_chunk = []
    substitution_data_chunk = []
    current_best_true_loss_chunk = []
    current_best_tokens_chunk = []
    logprobs_chunk = []
    generated_output_string_chunk = []

    for step_num in range(custom_gcg_hyperparams["max_steps"]):
        step_start_time = time.time()
        
        best_tokens_indices = signal_function(model, tokenizer, current_best_tokens, masks_data, custom_gcg_hyperparams["topk"], logger, step_num=step_num, **(signal_kwargs or {}))
        
        indices_to_sample = set()
        indices_to_exclude = set()
        substitutions_set = set()

        if isinstance(custom_gcg_hyperparams["forward_eval_candidates"], str):
            if custom_gcg_hyperparams["forward_eval_candidates"] == "all":
                for first_coordinate in range(best_tokens_indices.shape[0]):
                    for second_coordinate in range(best_tokens_indices.shape[1]):
                        substitution_make = current_best_tokens.clone()
                        substitution_make[optim_mask[first_coordinate]] = best_tokens_indices[(first_coordinate, second_coordinate)]
                        substitutions_set.add(substitution_make)
                substitution_data = torch.stack(list(substitutions_set))
        else:
            assert isinstance(custom_gcg_hyperparams["forward_eval_candidates"], int), "Only strings or ints"
            num_forward_evals = custom_gcg_hyperparams["forward_eval_candidates"]
            while len(indices_to_sample) < num_forward_evals:
                first_coordinate = torch.randint(0, best_tokens_indices.shape[0], (1,)).to(torch.int32).item()
                second_coordinate = torch.randint(0, best_tokens_indices.shape[1], (1,)).to(torch.int32).item()
                if (first_coordinate, second_coordinate) in indices_to_sample:
                    continue
                if (first_coordinate, second_coordinate) in indices_to_exclude:
                    continue
                random_substitution_make = current_best_tokens.clone()
                random_substitution_make[optim_mask[first_coordinate]] = best_tokens_indices[(first_coordinate, second_coordinate)]
                if (substitution_validity_function is None) or (substitution_validity_function(random_substitution_make, tokenizer=tokenizer, masks_data=masks_data)):
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
        

        true_losses = true_loss_function(model, tokenizer, substitution_data, masks_data, input_tokens[target_mask], logger, **true_loss_kwargs)
        true_losses_chunk.append(true_losses)
        current_best_true_loss = true_losses[torch.argmin(true_losses)]
        current_best_true_loss_chunk.append(current_best_true_loss)
        current_best_tokens = substitution_data[torch.argmin(true_losses)].clone()
        current_best_tokens_chunk.append(current_best_tokens)
        best_output_sequences.append(current_best_tokens.clone())
        logprobs = target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
        logprobs = logprobs.item()
        logprobs_chunk.append(logprobs)
        logprobs_sequences.append(logprobs)
        
        # Extended metrics collection
        if save_metrics_path:
            target_tokens = input_tokens[target_mask]
            
            # Always check argmax match
            argmax_matches = check_argmax_match(model, tokenizer, current_best_tokens, masks_data, target_tokens)
            
            # Prepare step metrics
            step_metric = {
                "step": step_num,
                "loss": logprobs,
                "argmax_matches_target": argmax_matches,
                "time_elapsed": time.time() - step_start_time
            }
            
            # Check extended metrics periodically
            if step_num % check_extended_metrics_every_n_steps == 0:
                # Generate text for extended checks
                with torch.no_grad():
                    input_tokens_for_generation = current_best_tokens[eval_input_mask]
                    generated_tokens = model.generate(
                        torch.unsqueeze(input_tokens_for_generation, dim=0).to(model.device),
                        attention_mask=torch.unsqueeze(torch.ones(input_tokens_for_generation.shape), dim=0).to(model.device),
                        **generation_config
                    )
                    # Get the actual number of input tokens used for generation
                    input_length = len(input_tokens_for_generation)
                    extended_generated_text = tokenizer.batch_decode(generated_tokens[:, input_length:])[0]
                
                starts_with_target = check_generation_starts_with_target(extended_generated_text, target_tokens, tokenizer)
                
                step_metric["generation_starts_with_target"] = starts_with_target
                step_metric["generated_text"] = extended_generated_text[:100]
            
            # Save adversarial string periodically
            if step_num % save_adv_string_every_n_steps == 0:
                # Get prefix and suffix tokens separately
                prefix_tokens = current_best_tokens[masks_data["prefix_mask"]]
                suffix_tokens = current_best_tokens[masks_data["suffix_mask"]]
                # Decode with separator to show as it appears in the prompt
                prefix_str = tokenizer.decode(prefix_tokens)
                suffix_str = tokenizer.decode(suffix_tokens)
                step_metric["current_adv_string"] = f"{prefix_str} . {suffix_str}"
            
            # Save metrics incrementally
            per_step_metrics.append(step_metric)
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(step_metric) + '\n')
        
        if eval_every_step:
            input_tokens_for_generation = current_best_tokens[eval_input_mask]
            generated_output_tokens = model.generate(torch.unsqueeze(input_tokens_for_generation, dim=0).to(model.device), attention_mask=torch.unsqueeze(torch.ones(input_tokens_for_generation.shape), dim=0).to(model.device), **generation_config)
            # Get the actual number of input tokens used for generation
            input_length = len(input_tokens_for_generation)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, input_length:])[0]
            generated_output_string_chunk.append(generated_output_string)
            if early_stop:
                # Check if target appears at the beginning of the generated string
                if check_generation_starts_with_target(generated_output_string, input_tokenized_data["tokens"][target_mask], tokenizer):
                    successive_correct_outputs += 1
                    if successive_correct_outputs >= identical_outputs_before_stop:
                        # Log early stopping with extended metrics if enabled
                        if save_metrics_path and "current_adv_string" not in step_metric:
                            # Add final adversarial string before stopping
                            prefix_tokens = current_best_tokens[masks_data["prefix_mask"]]
                            suffix_tokens = current_best_tokens[masks_data["suffix_mask"]]
                            prefix_str = tokenizer.decode(prefix_tokens)
                            suffix_str = tokenizer.decode(suffix_tokens)
                            step_metric["current_adv_string"] = f"{prefix_str} . {suffix_str}"
                            step_metric["early_stop"] = True
                            # Re-save the metric with early stop indicator
                            per_step_metrics[-1] = step_metric
                            # Rewrite the last line in the file
                            with open(metrics_file, 'r') as f:
                                lines = f.readlines()
                            lines[-1] = json.dumps(step_metric) + '\n'
                            with open(metrics_file, 'w') as f:
                                f.writelines(lines)
                        break
                else:
                    successive_correct_outputs = 0
    
        if (step_num + 1) % 10 == 0:
            logger.log(substitution_data_chunk, step_num=step_num)
            logger.log(true_losses_chunk, step_num=step_num)
            logger.log(current_best_true_loss_chunk, step_num=step_num)
            logger.log(current_best_tokens_chunk, step_num=step_num)
            logger.log(best_tokens_chunk, step_num=step_num)
            logger.log(logprobs_chunk, step_num=step_num)
            logger.log(generated_output_string_chunk, step_num=step_num)

            substitution_data_chunk = []
            true_losses_chunk = []
            current_best_true_loss_chunk = []
            current_best_tokens_chunk = []
            best_tokens_chunk = []
            logprobs_chunk = []
            generated_output_string_chunk = []

    logger.log(successive_correct_outputs, num_steps=step_num)
    
    # Return extended results if metrics were collected
    if save_metrics_path:
        return {
            "logprobs_sequences": logprobs_sequences,
            "best_output_sequences": best_output_sequences,
            "per_step_metrics": per_step_metrics,
            "final_success": successive_correct_outputs >= identical_outputs_before_stop,
            "total_steps": step_num + 1
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
    canonical_device_idx = 0,
    normalize_grads_before_accumulation = True,
    **kwargs
):
    
    num_elements_per_batch = len(input_tokenized_data_list) // len(models)
    input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

    grads_list = []
    for model, input_tokenized_data_list_batch in zip(models, input_tokenized_data_list_batches):
        grads_list_batch = []
        for input_tokenized_data in input_tokenized_data_list_batch:
            input_points = input_tokenized_data["tokens"]
            masks_data = input_tokenized_data["masks"]

            optim_mask: torch.Tensor = masks_data["optim_mask"]
            target_mask: torch.Tensor = masks_data["target_mask"]
            
            # Get vocabulary size properly
            if hasattr(tokenizer, 'vocab_size'):
                vocab_size = tokenizer.vocab_size
            elif hasattr(tokenizer, 'get_vocab'):
                vocab_size = len(tokenizer.get_vocab())
            else:
                vocab_size = len(tokenizer.vocab)
            
            # Ensure vocab size doesn't exceed embedding size
            embedding_size = model.get_input_embeddings().weight.shape[0]
            vocab_size = min(vocab_size, embedding_size)
            
            one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=vocab_size).to(dtype=model.dtype)
            one_hot_tensor.requires_grad_()
            embedding_tensor = model.get_input_embeddings().weight[:vocab_size]
            inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
            logits = model(inputs_embeds=inputs_embeds).logits
            loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).sum()
            loss_tensor.backward()
            if normalize_grads_before_accumulation:
                normalized_grad = one_hot_tensor.grad[optim_mask, :] / one_hot_tensor.grad[optim_mask, :].norm(dim=-1, keepdim=True)
                grads_list_batch.append(normalized_grad)
            else:
                grads_list_batch.append(one_hot_tensor.grad[optim_mask, :])    
        grads_list.append(torch.stack(grads_list_batch))
    
    device_moved_grad_list = []
    for grads_list_batch_tensor in grads_list:
        device_moved_grad_list.append(grads_list_batch_tensor.to(canonical_device_idx))
    
    final_grads = - torch.cat(device_moved_grad_list, dim=0).mean(dim=0)
    best_tokens_indices = final_grads.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices

def DEFAULT_GCG_RANDOMNESS_STRATEGY(tokenizer, best_tokens_indices, input_tokenized_data_list, substitution_validity_function, max_candidate_size):
    
    indices_to_sample = set()
    indices_to_exclude = set()

    while len(indices_to_sample) < max_candidate_size:
        first_coordinate = torch.randint(0, best_tokens_indices.shape[0], (1,)).to(torch.int32).item()
        second_coordinate = torch.randint(0, best_tokens_indices.shape[1], (1,)).to(torch.int32).item()
        if (first_coordinate, second_coordinate) in indices_to_sample:
            continue
        if (first_coordinate, second_coordinate) in indices_to_exclude:
            continue

        all_substitutions_valid = True
        for input_tokenized_data in input_tokenized_data_list:
            masks_data = input_tokenized_data["masks"]
            optim_mask = masks_data["optim_mask"]
            random_substitution_make = input_tokenized_data["tokens"].clone()  
            random_substitution_make[optim_mask[first_coordinate]] = best_tokens_indices[(first_coordinate, second_coordinate)]

            if (substitution_validity_function is None) or (substitution_validity_function(random_substitution_make, tokenizer=tokenizer, masks_data=masks_data)):
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
            random_substitution_make[optim_mask[index_to_sample[0]]] = best_tokens_indices[(index_to_sample[0], index_to_sample[1])]
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
    to_cache_attentions    
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
    
    signal_function = universal_gcg_hyperparameters.get("signal_function", average_target_logprobs_signal)
    true_loss_function = universal_gcg_hyperparameters.get("true_loss_function", average_target_logprobs)
    substitution_validity_function = universal_gcg_hyperparameters.get("substitution_validity_function", None)
    signal_kwargs = universal_gcg_hyperparameters.get("signal_kwargs", None)
    true_loss_kwargs = universal_gcg_hyperparameters.get("true_loss_kwargs", None)
    randomness_strategy = universal_gcg_hyperparameters.get("randomness_strategy", DEFAULT_GCG_RANDOMNESS_STRATEGY)

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
        initial_true_loss = true_loss_function(models, tokenizer, [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list], masks_data_list, logger, **true_loss_kwargs)
        logger.log(initial_true_loss, step_num=-1)
        initial_average_logprobs = average_target_logprobs(models, tokenizer, [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list], masks_data_list, logger)
        initial_average_logprobs = initial_average_logprobs.item()
        logger.log(initial_average_logprobs, step_num=-1)
        average_logprobs_list.append(initial_average_logprobs)
        best_tokens_dicts_list.append(attack_utility.form_best_tokens_dict(input_tokenized_data_list))

    best_tokens_dicts_chunk = []
    true_losses_chunk = []
    current_best_true_loss_chunk = []
    logprobs_chunk = []


    current_input_tokenized_data_list = input_tokenized_data_list
    for step_num in range(universal_gcg_hyperparameters["max_steps"]):

        step_begin_state = on_step_begin(models, tokenizer, current_input_tokenized_data_list, universal_gcg_hyperparameters, logger, step_num=step_num, **on_step_begin_kwargs)

        best_tokens_indices = signal_function(models, tokenizer, current_input_tokenized_data_list, universal_gcg_hyperparameters["topk"], logger, step_num=step_num, **(signal_kwargs or {}))
        forward_eval_candidates = randomness_strategy(tokenizer, best_tokens_indices, current_input_tokenized_data_list, substitution_validity_function, universal_gcg_hyperparameters["forward_eval_candidates"])
        true_losses = true_loss_function(models, tokenizer, forward_eval_candidates, masks_data_list, logger, step_num=step_num, **(true_loss_kwargs or {}))
        true_losses_chunk.append(true_losses)
        best_idx = torch.argmin(true_losses)
        best_loss = true_losses[best_idx]
        current_best_true_loss_chunk.append(best_loss)
        best_tokens_dict = {
            "prefix_tokens": forward_eval_candidates[0][best_idx][masks_data_list[0]["prefix_mask"]],
            "suffix_tokens": forward_eval_candidates[0][best_idx][masks_data_list[0]["suffix_mask"]]
        }
        best_tokens_dicts_chunk.append(best_tokens_dict)
        best_tokens_dicts_list.append(best_tokens_dict)
        average_logprobs = average_target_logprobs(models, tokenizer, [torch.unsqueeze(x[best_idx], 0) for x in forward_eval_candidates], masks_data_list, logger)
        logprobs_chunk.append(average_logprobs.item())
        average_logprobs_list.append(average_logprobs.item())
        current_input_tokenized_data_list = attack_utility.update_all_tokens(best_tokens_dict, current_input_tokenized_data_list)

        if (step_num + 1) % 10 == 0:
            logger.log(true_losses_chunk, step_num=step_num)
            logger.log(current_best_true_loss_chunk, step_num=step_num)
            logger.log(best_tokens_dicts_chunk, step_num=step_num)
            logger.log(logprobs_chunk, step_num=step_num)

            true_losses_chunk = []
            current_best_true_loss_chunk = []
            logprobs_chunk = []
            best_tokens_dicts_chunk = []
        
        step_end_state = on_step_end(models, tokenizer, current_input_tokenized_data_list, universal_gcg_hyperparameters, logger, step_num=step_num, **on_step_end_kwargs)

        gc.collect()
        torch.cuda.empty_cache()

    return best_tokens_dicts_list, average_logprobs_list
