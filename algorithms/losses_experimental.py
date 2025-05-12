import torch
import transformers
import gc
import typing
import peft
import datasets
import random
import copy
import sys
import time

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
    input_points,
    masks_data,
    logger,
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
        layer_weight_strategy = layer_weight_strategy(model, tokenizer, input_points, masks_data, logger, **kwargs)

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

    layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions, input_points, masks_data, logger)

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
    layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions, input_points, masks_data, logger)

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


def get_dolly_data(tokenizer, input_tokenized_data, logger):

    tokens = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    target = tokenizer.decode(tokens[0][masks_data["target_mask"]], clean_up_tokenization_spaces=False)

    prefix_length = len(masks_data["prefix_mask"])
    suffix_length = len(masks_data["suffix_mask"])

    init_config = {
        "strategy_type": "random",
        "prefix_length": prefix_length,
        "suffix_length": suffix_length,
        "seed": int(time.time())
    }

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

class SingleAttentionGradHook:
    def __init__(self, model, input_tokenized_data):
        self.model = model
        self.num_layers = len(attack_utility._get_layer_obj(model))
        self.attention_weights = [None] * self.num_layers
        self.attention_grads = [None] * self.num_layers
        self.input_tokenized_data = input_tokenized_data
        
    def accumulate_grads(self):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        with torch.enable_grad():
            input_ids = self.input_tokenized_data["tokens"]
            device = next(self.model.parameters()).device
            input_tensor = torch.unsqueeze(input_ids.to(device), dim=0)
            
            outputs = self.model(input_ids=input_tensor, output_attentions=True)
            for attn_weight in outputs.attentions:
                attn_weight.retain_grad()
            self.attention_weights = outputs.attentions

            target_mask = self.input_tokenized_data["masks"]["target_mask"]
            target_logits = outputs.logits[0, target_mask - 1, :]
            true_labels = self.input_tokenized_data["tokens"][target_mask].to(device)
            loss = torch.nn.CrossEntropyLoss()(target_logits, true_labels)                
            loss.backward()
            for i in range(self.num_layers):
                if self.attention_weights[i] is not None and hasattr(self.attention_weights[i], 'grad'):
                    self.attention_grads[i] = self.attention_weights[i].grad.detach().to("cpu")
            # self.attention_grads[i] is the gradient wrt the attention matrix i
            # self.attention_grads[i] is of shape (batch size (always 1), num_heads, context_length, context_length)

class MultiAttentionGradHook:
    def __init__(self, model, input_tokenized_data_list):
        self.model = model
        self.input_tokenized_data_list = input_tokenized_data_list
        self.num_layers = len(attack_utility._get_layer_obj(model))
        self.single_attention_grad_hooks_list = [SingleAttentionGradHook(model, x) for x in input_tokenized_data_list]
        self.grads = [None] * len(self.single_attention_grad_hooks_list)
        self.accumulated = False

    def accumulate_gradients(self):
        if self.accumulated:
            raise ValueError(f"Don't call accumulate when already accumulated")
        for i, attn_hook in enumerate(self.single_attention_grad_hooks_list):
            attn_hook.accumulate_grads()
            self.grads[i] = attn_hook.attention_grads
            gc.collect()
            torch.cuda.empty_cache()
        self.accumulated = True
        gc.collect()

    def _are_we_same_example(self):
        masks_data_list = []
        non_optim_tokens_list = []
        for input_tokenized_data in self.input_tokenized_data_list:
            tokens = input_tokenized_data["tokens"]
            optim_mask = input_tokenized_data["masks"]["optim_mask"]
            non_optim_tokens = tokens[[i for i in range(len(tokens)) if i not in optim_mask]]
            non_optim_tokens_list.append(non_optim_tokens)
            masks_data_list.append(input_tokenized_data["masks"])
        return (all([x.data.tolist() == non_optim_tokens_list[0].data.tolist() for x in non_optim_tokens_list])) and (all([x == masks_data_list[0] for x in masks_data_list]))

    def layer_wise_abs_grads(self):
        if not self._are_we_same_example():
            raise ValueError("This only makes sense when the inputs are all of the same structure")
        
        if not self.accumulated:
            self.accumulate_gradients()

        target_mask = self.input_tokenized_data_list[0]["masks"]["target_mask"]
        layer_wise_abs_grads_sums = []
        for layer_idx in range(self.num_layers):
            layer_wise_abs_grads_sums.append([])
            for per_ex_grad_val in self.grads:
                layer_wise_abs_grads_sums[layer_idx].append(torch.abs(per_ex_grad_val[layer_idx][0][:, target_mask - 1, :]).mean(dim=-1).sum(dim=-1))
        layer_wise_abs_grads_means = [torch.mean(torch.stack(layer_wise_abs_grads_sums[layer_idx]), dim=0) for layer_idx in range(self.num_layers)]
        return layer_wise_abs_grads_means

def generate_random_inits_for_one_example(model, tokenizer, input_tokenized_data, num_randoms):
    new_input_tokenized_data_list = []
    for _ in range(num_randoms):
        optim_mask = input_tokenized_data["masks"]["optim_mask"]
        new_random = torch.randint_like(optim_mask, 0, tokenizer.vocab_size)
        new_tokens = input_tokenized_data["tokens"]
        new_tokens[optim_mask] = new_random
        new_input_tokenized_data = {
            "tokens": new_tokens,
            "masks": input_tokenized_data["masks"]
        }
        new_input_tokenized_data_list.append(new_input_tokenized_data)
    return new_input_tokenized_data_list

def attention_heads_across_training_examples(model, tokenizer, dolly_full_data, num_examples, num_randoms_per_example):

    dolly_relevant_examples = random.sample(dolly_full_data, num_examples)    

    per_example_output_means = []
    for example, relevant_example in enumerate(dolly_relevant_examples):
        random_perturbs = generate_random_inits_for_one_example(model, tokenizer, relevant_example, num_randoms_per_example)
        mgh = MultiAttentionGradHook(model, random_perturbs)
        mgh.accumulate_gradients()
        output_mean = mgh.layer_wise_abs_grads()
        per_example_output_means.append(output_mean)
        gc.collect()
        torch.cuda.empty_cache()
    return per_example_output_means

def abs_grad_dolly_layer_weights(model, tokenizer, input_tokenized_data, logger):
    dolly_convolved_dataset, _ = get_dolly_data(tokenizer, input_tokenized_data, logger)
    means = attention_heads_across_training_examples(model, tokenizer, dolly_convolved_dataset, 5, 5)
    example_mean_all = []
    for example_num, example_output_mean in enumerate(means):
        example_mean_all.append(torch.stack(example_output_mean))

    final_mean = torch.mean(torch.stack(example_mean_all), dim=0)
    return final_mean


CACHED_DOLLY_LAYER_WEIGHT_OBJ = None
def cached_abs_grad_dolly_layer_weights(model, tokenizer, input_points, masks_data, logger):
    global CACHED_DOLLY_LAYER_WEIGHT_OBJ
    if CACHED_DOLLY_LAYER_WEIGHT_OBJ is None:
        input_tokenized_data = {
            "tokens": input_points,
            "masks": masks_data
        }
        CACHED_DOLLY_LAYER_WEIGHT_OBJ = abs_grad_dolly_layer_weights(model, tokenizer, input_tokenized_data, logger)
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    batch_size = input_points.shape[0]
    final_tensor = torch.transpose(torch.unsqueeze(CACHED_DOLLY_LAYER_WEIGHT_OBJ, dim=0).expand(batch_size, -1, -1).unsqueeze(dim=-1).expand(-1, -1, -1, len(masks_data["target_mask"])), 0, 1)
    return final_tensor