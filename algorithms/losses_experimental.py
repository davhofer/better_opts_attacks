import torch
import transformers
import gc
import typing
import peft
import datasets
import random
import copy
import sys

import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
from secalign_refactored import secalign


def process_batch_attentions(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.Tensor,
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    start_idx: int,
    end_idx: int,
    layer_weight_strategy: str,
) -> float:
    """
    Process a batch of data, automatically splitting if OOM occurs.
    Returns the accumulated attention weight loss for the batch.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        input_points: Input token indices
        target_mask: Mask indicating target positions (full sequence)
        attention_mask: Mask indicating positions to attend to (full sequence)
        start_idx: Start index of current batch
        end_idx: End index of current batch
        layer_weight_strategy: Strategy for weighting attention layers
    """
    input_points.requires_grad_(False)
    batch_input_points = input_points[start_idx:end_idx]
    
    with torch.inference_mode():
        gc.collect()
        torch.cuda.synchronize()   
        torch.cuda.empty_cache()
        torch.cuda.synchronize()   
        # Get embeddings
        batch_inputs_embeds = model.get_input_embeddings()(batch_input_points)
        
        # Forward pass for the batch
        try:
            batch_output = model(
                inputs_embeds=batch_inputs_embeds,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if end_idx - start_idx <= 1:
                raise torch.cuda.OutOfMemoryError("Cannot process even a single sample")
            
            del batch_inputs_embeds
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Split batch in half and process recursively
            mid_idx = start_idx + (end_idx - start_idx) // 2
            first_half = process_batch_attentions(
                model, tokenizer, input_points, target_mask, attention_mask,
                start_idx, mid_idx, layer_weight_strategy
            )
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()   

            second_half = process_batch_attentions(
                model, tokenizer, input_points, target_mask, attention_mask,
                mid_idx, end_idx, layer_weight_strategy
            )
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            return torch.cat((first_half, second_half), dim=0)
        
        batch_attentions = batch_output.attentions
        
        # Process attention outputs based on strategy
        if layer_weight_strategy == "uniform":
            batch_final = torch.stack(list((
                layer_attention[
                    :,
                    :,
                    target_mask - 1
                ][..., attention_mask].sum(dim=[1,2,3])
                for layer_attention in batch_attentions
            ))).sum(dim=0)
        elif layer_weight_strategy == "only_last":
            batch_final = batch_attentions[-1][
                :,
                :,
                target_mask - 1
            ][..., attention_mask].sum(dim=[1,2,3])
        elif layer_weight_strategy == "only_first":
            batch_final = batch_attentions[0][
                :,
                :,
                target_mask - 1
            ][..., attention_mask].sum(dim=[1,2,3])
        elif layer_weight_strategy in ("increasing", "decreasing"):
            num_weights = len(batch_attentions)
            weights = range(1, num_weights + 1)
            if layer_weight_strategy == "decreasing":
                weights = reversed(weights)
            batch_final = torch.stack(list((
                weight * layer_attention[
                    :,
                    :,
                    target_mask - 1
                ][..., attention_mask].sum(dim=[1,2,3])
                for weight, layer_attention in zip(weights, batch_attentions)
            ))).sum(dim=0)
        elif isinstance(layer_weight_strategy, list):
            batch_final = torch.stack(list((
                weight * layer_attention[
                    :,
                    :,
                    target_mask - 1
                ][..., attention_mask].sum(dim=[1,2,3])
                for weight, layer_attention in zip(layer_weight_strategy, batch_attentions)
            ))).sum(dim=0)
        # Clean up

        del batch_attentions
        del batch_output
        del batch_inputs_embeds
        gc.collect()
        torch.cuda.empty_cache()

        return batch_final
        
def attention_weight_signal_v1(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    topk,
    logger: experiment_logger.ExperimentLogger,
    *,
    layer_weight_strategy: str = "uniform",
    attention_mask_strategy: str = "payload_only",
    **kwargs
):
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    payload_mask: torch.tensor = masks_data["payload_mask"]
    control_mask: torch.tensor = masks_data["control_mask"]

    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
    elif attention_mask_strategy == "payload_and_control":
        attention_mask = torch.cat(payload_mask, control_mask)

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    model_output = model(inputs_embeds=inputs_embeds, output_attentions=True, return_dict=True)
    attentions = model_output.attentions
    relevant_attentions = torch.stack([layer_attention[0, :, target_mask - 1][:, :, attention_mask] for layer_attention in attentions])
    if layer_weight_strategy == "uniform":
        final_tensor = relevant_attentions.sum()
    elif layer_weight_strategy == "only_last":
        final_tensor = relevant_attentions[-1].sum()
    elif layer_weight_strategy == "only_first":
        final_tensor = relevant_attentions[0].sum()
    elif layer_weight_strategy == "increasing":
        num_weights = relevant_attentions.shape[0]
        weights_tensor = torch.tensor(list(range(num_weights))) + 1
        final_tensor = (weights_tensor.view(num_weights, 1, 1, 1).to(relevant_attentions.device) * relevant_attentions).sum()
    elif layer_weight_strategy == "decreasing":
        num_weights = relevant_attentions.shape[0]
        weights_tensor = torch.tensor(list(reversed(list(range(num_weights))))) + 1
        final_tensor = (weights_tensor.view(num_weights, 1, 1, 1).to(relevant_attentions.device) * relevant_attentions).sum()
    elif isinstance(layer_weight_strategy, list):
        num_weights = relevant_attentions.shape[0]
        weights_tensor = torch.tensor(layer_weight_strategy)
        final_tensor = (weights_tensor.view(num_weights, 1, 1, 1).to(relevant_attentions.device) * relevant_attentions).sum()
    else:
        raise ValueError(f"layer_weight_strategy parameter {layer_weight_strategy} not recognized")
    final_tensor.backward()
    grad_optims = (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(topk, dim=-1).indices
    return best_tokens_indices

def attention_weight_loss_v1(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    topk,
    logger: experiment_logger.ExperimentLogger,
    *,
    layer_weight_strategy: str = "uniform",
    attention_mask_strategy: str = "payload_only",
    **kwargs
):
    """
    Compute attention weight loss with dynamic batching that automatically adjusts
    to available memory. Maintains the same API and semantics as the original function.
    
    The function starts with the initial_batch_size and automatically reduces batch size
    when OOM occurs, processing data in appropriately-sized chunks.
    """

    target_mask: torch.tensor = masks_data["target_mask"]
    payload_mask: torch.tensor = masks_data["payload_mask"]
    control_mask: torch.tensor = masks_data["control_mask"]

    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
    elif attention_mask_strategy == "payload_and_control":
        attention_mask = torch.cat(payload_mask, control_mask)
    
    seq_length = input_points.shape[0]

    final_result = process_batch_attentions(
        model, tokenizer, input_points, target_mask, attention_mask,
        0, seq_length, layer_weight_strategy
    )    
    return -final_result

def smart_layer_weight_strategy(
    model,
    tokenizer,
    layer_weight_strategy,
    ideal_attentions,
    input_points = None,
    masks_data = None,
    **kwargs
):
    assert isinstance(ideal_attentions, torch.Tensor) and ideal_attentions.dim() == 5 # (layer, batch, attention head, row, column)

    if isinstance(layer_weight_strategy, str):
        if layer_weight_strategy == "uniform":
            layer_weight_strategy = torch.ones(ideal_attentions.shape[:-1], dtype=ideal_attentions.dtype)
        elif layer_weight_strategy == "increasing":            
            layer_weight_strategy = torch.stack([(1 + i) * torch.ones(ideal_attentions.shape[1:-1]) for i in range(int(ideal_attentions.shape[0]))])
        elif layer_weight_strategy == "decreasing":
            layer_weight_strategy = torch.stack(list(reversed([(1 + i) * torch.ones(ideal_attentions.shape[1:-1]) for i in range(int(ideal_attentions.shape[0]))])))
    elif isinstance(layer_weight_strategy, int):
        assert layer_weight_strategy < ideal_attentions.shape[0]
        layer_weight_strategy_tensor = torch.zeros(ideal_attentions.shape[:-1], dtype=ideal_attentions.dtype)
        layer_weight_strategy_tensor[layer_weight_strategy, :, :, :] = 1
        layer_weight_strategy = layer_weight_strategy_tensor
    elif isinstance(layer_weight_strategy, torch.Tensor):
        assert layer_weight_strategy.shape == ideal_attentions.shape[:-1]
        pass
    elif callable(layer_weight_strategy): # This is the fine-grained case where we do more detailed stuff, hopefully we never need this
        layer_weight_strategy = layer_weight_strategy(model, tokenizer, input_points, masks_data, **kwargs)

    assert isinstance(layer_weight_strategy, torch.Tensor) and layer_weight_strategy.dim() == 4 and layer_weight_strategy.shape == ideal_attentions.shape[:-1]
    
    return layer_weight_strategy

def smart_ideal_attentions(
    model,
    tokenizer,
    ideal_attentions,
    input_points = None,
    masks_data = None,
    **kwargs    
):
    if isinstance(ideal_attentions, torch.Tensor):
        return ideal_attentions
    elif callable(ideal_attentions):
        return ideal_attentions(model, tokenizer, input_points, masks_data, **kwargs)

def attention_metricized_signal_v2(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    topk,
    logger: experiment_logger.ExperimentLogger,
    *,
    prob_dist_metric,
    ideal_attentions,
    layer_weight_strategy,
    **kwargs
):
    
    ideal_attention_kwargs = kwargs.get("ideal_attentions_kwargs", {})
    ideal_attentions = smart_ideal_attentions(model, tokenizer, ideal_attentions, input_points, masks_data, **ideal_attention_kwargs)

    layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions, input_points, masks_data)

    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    model_output = model(inputs_embeds=inputs_embeds, output_attentions=True, return_dict=True)
    true_attentions = torch.stack([attention[:, :, target_mask - 1, :] for attention in model_output.attentions])
    loss_tensor = prob_dist_metric(model, tokenizer, input_points, masks_data, ideal_attentions, true_attentions, logger=logger, layer_weight_strategy=layer_weight_strategy)
    loss_tensor.backward()
    grad_optims = - (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(topk, dim=-1).indices
    return best_tokens_indices    

def attention_metricized_v2_true_loss(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    target_tokens,
    logger: experiment_logger.ExperimentLogger,
    *,
    prob_dist_metric,
    ideal_attentions,
    layer_weight_strategy,
    **kwargs
):
    ideal_attention_kwargs = kwargs.get("ideal_attentions_kwargs", {})
    input_points_to_send = input_points

    ideal_attentions = smart_ideal_attentions(model, tokenizer, ideal_attentions, input_points_to_send, masks_data, **ideal_attention_kwargs)
    layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions, input_points, masks_data)

    target_mask: torch.tensor = masks_data["target_mask"]
    loss_tensors_list = []
    num_processed = 0
    for batch_logits, batch_true_attentions in attack_utility.bulk_forward_iter(model, input_points_to_send):
        true_attentions = torch.stack([attention[:, :, -(len(target_mask) + 1):- 1, :] for attention in batch_true_attentions])
        loss_tensor = prob_dist_metric(model, tokenizer, input_points, masks_data, ideal_attentions[:, num_processed:num_processed + true_attentions.shape[1], ...], true_attentions, logger=logger, layer_weight_strategy=layer_weight_strategy[:, num_processed:num_processed + true_attentions.shape[1], ...])
        num_processed += true_attentions.shape[1]
        loss_tensors_list.append(loss_tensor)
    
    loss_tensor = torch.cat(loss_tensors_list)
    return loss_tensor

def kl_divergence_payload_only(
    model,
    tokenizer,
    input_points,
    masks_data,
    ideal_attentions,
    true_attentions,
    logger,
    *,
    layer_weight_strategy
):
    assert true_attentions.shape == ideal_attentions.shape

    payload_mask = masks_data["payload_mask"]
    true_attentions = true_attentions[:, :, :, :, payload_mask]
    ideal_attentions = ideal_attentions[:, :, :, :, payload_mask]
    true_attentions_log_space = torch.log(true_attentions + 1e-6)
    divvied_up_losses = torch.nn.functional.kl_div(true_attentions_log_space, ideal_attentions.to(true_attentions_log_space.device), reduction="none")
    batch_first_att_strategy = torch.transpose(layer_weight_strategy, 1, 0)
    batch_first_losses = torch.nansum(torch.transpose(divvied_up_losses, 1, 0), dim=-1)
    product = batch_first_att_strategy.to(batch_first_losses.device) * batch_first_losses
    result = product.sum(dim=(1, 2, 3))
    return result

def cross_entropy_payload_only(
    model,
    tokenizer,
    input_points,
    masks_data,
    ideal_attentions,
    true_attentions,
    *,
    layer_weight_strategy
):
    assert true_attentions.shape == ideal_attentions.shape

    payload_mask = masks_data["payload_mask"]
    true_attentions = true_attentions[:, :, :, :, payload_mask]
    ideal_attentions = ideal_attentions[:, :, :, :, payload_mask]
    true_attentions_log_space = torch.log(true_attentions + 1e-6)
    divvied_up_losses = torch.nn.functional.cross_entropy(true_attentions_log_space, ideal_attentions.to(true_attentions_log_space.device), reduction="none")
    batch_first_att_strategy = torch.transpose(layer_weight_strategy, 1, 0)
    batch_first_losses = torch.nansum(torch.transpose(divvied_up_losses, 1, 0), dim=-1)
    product = batch_first_att_strategy.to(batch_first_losses.device) * batch_first_losses
    result = product.sum(dim=(1, 2, 3))
    return result


def pointwise_sum_of_differences_payload_only(
    model,
    tokenizer,
    input_points,
    masks_data,
    ideal_attentions,
    true_attentions,
    logger: experiment_logger.ExperimentLogger,
    *,
    layer_weight_strategy
):
    assert true_attentions.shape == ideal_attentions.shape
    payload_mask = masks_data["payload_mask"]
    true_attentions = true_attentions[:, :, :, :, payload_mask]
    ideal_attentions = ideal_attentions[:, :, :, :, payload_mask]
    divvied_up_losses = ideal_attentions.to(true_attentions.device) - true_attentions
    batch_first_att_strategy = torch.transpose(layer_weight_strategy, 1, 0)
    batch_first_losses = torch.nansum(torch.transpose(divvied_up_losses, 1, 0), dim=-1)
    product = batch_first_att_strategy.to(batch_first_losses.device) * batch_first_losses
    result = product.sum(dim=(1, 2, 3))
    return result


def secalign_ideal_attention_v1(
    model,
    tokenizer,
    input_points,
    masks_data,
    *,
    attention_mask_strategy,
    **kwargs
):
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    
    payload_mask: torch.Tensor = masks_data["payload_mask"]
    control_mask: torch.Tensor = masks_data["control_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]

    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
        attention_mask_formatted = torch.zeros(input_points.shape)
        attention_mask_formatted[:, attention_mask] = 1
        attentions = model(input_ids=input_points.to(model.device), attention_mask=attention_mask_formatted.to(model.device), output_attentions=True).attentions
    elif attention_mask_strategy == "payload_and_control":
        attention_mask = torch.cat((payload_mask, control_mask))
        attention_mask_formatted = torch.zeros(input_points.shape)
        attention_mask_formatted[:, attention_mask] = 1
        attentions = model(input_ids=input_points.to(model.device), attention_mask=attention_mask_formatted.to(model.device), output_attentions=True).attentions
    # Removing these two, because it might be that we can emulate the squeezed
    # versions with just modifying position_ids???
    # 
    # elif attention_mask_strategy == "payload_control_squeezed":
    #     attention_mask = torch.cat((payload_mask, control_mask))
    #     attentions = model(input_ids=input_points[:, sorted(attention_mask)], output_attentions=True).attentions
    # elif attention_mask_strategy == "payload_squeezed":
    #     attention_mask = payload_mask
    #     attentions = model(input_ids=input_points[:, sorted(attention_mask)], output_attentions=True).attentions
    else:
        raise ValueError(f"attention_mask_strategy {attention_mask_strategy} is not implemented yet.")
    return torch.stack(attentions)[:, :, :, target_mask - 1, :]

def uniform_ideal_attentions(
    model,
    tokenizer,
    input_points,
    masks_data,
    *,
    attention_mask_strategy,
):
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    payload_mask: torch.Tensor = masks_data["payload_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]
    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
    elif attention_mask_strategy == "payload_and_control":
        control_mask: torch.Tensor = masks_data["control_mask"]    
        attention_mask = torch.cat((payload_mask, control_mask))
    else:
        raise ValueError(f"attention_mask_strategy {attention_mask_strategy} is not implemented yet.")
    dummy_attentions = torch.stack(model(input_ids=torch.unsqueeze(input_points[0], dim=0).to(model.device), output_attentions=True).attentions)
    ideal_shape = dummy_attentions.shape
    ideal_shape = (ideal_shape[0], input_points.shape[0], ideal_shape[2], ideal_shape[3], ideal_shape[4])
    attentions = torch.zeros(ideal_shape)
    attentions[:, :, :, :, attention_mask] = 1 / len(attention_mask)
    return attentions[:, :, :, -(len(target_mask) + 1):-1, :]


def get_dolly_data(logger, tokenizer, prompt_template, init_config, target, harmful_inst, convert_to_secalign_format=False):

    assert (init_config is not None) and (target is not None)

    dolly_15k_raw = datasets.load_dataset("databricks/databricks-dolly-15k")
    dolly_15k_filtered = [x for x in dolly_15k_raw["train"] if (x["context"] != "" and x["instruction"] != "")]
    dolly_data = [x for x in dolly_15k_filtered if len(x["context"]) <= 200 and len(x["instruction"]) < 300]
    dolly_data = [
        [
            {
                "role": "system",
                "content": x["instruction"]
            },
            {
                "role": "user",
                "content": x["context"]
            }
        ]
        for x in dolly_data
    ]
    if convert_to_secalign_format:
        dolly_data = [secalign._convert_to_secalign_format(input_conv, prompt_template, tokenizer, harmful_inst) for input_conv in dolly_data]

    random_input_conv = random.choice(dolly_data)
    input_tokenized_data, true_init_config = attack_utility.generate_valid_input_tokenized_data(tokenizer, random_input_conv, target, init_config, logger)
    true_prefix_tokens = input_tokenized_data["tokens"][input_tokenized_data["masks"]["prefix_mask"]]
    true_suffix_tokens = input_tokenized_data["tokens"][input_tokenized_data["masks"]["suffix_mask"]]
    new_dolly_data = []
    for dolly_data_point in dolly_data:
        current_input_tokenized_data, _ = attack_utility.generate_valid_input_tokenized_data(tokenizer, dolly_data_point, target, init_config, logger)
        current_prefix_mask = current_input_tokenized_data["masks"]["prefix_mask"]
        current_suffix_mask = current_input_tokenized_data["masks"]["suffix_mask"]
        new_tokens = copy.deepcopy(current_input_tokenized_data["tokens"])
        new_tokens[current_prefix_mask[-len(true_prefix_tokens):]] = true_prefix_tokens[-len(true_prefix_tokens):]
        new_tokens[current_suffix_mask[:len(true_suffix_tokens)]] = true_suffix_tokens[:len(true_suffix_tokens)]
        new_dolly_data.append(
            {
                "tokens": new_tokens,
                "masks": current_input_tokenized_data["masks"]
            }
        )
    return new_dolly_data, true_init_config





def abs_grad_dolly_layer_weights(model, tokenizer, input_tokenized_data, logger, **kwargs):
    dolly_data = datasets.load_dataset("")