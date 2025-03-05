import torch
import transformers
import json
import typing
import pickle as pkl
from contextlib import contextmanager
import time
import datetime
import gc
import shutil
import pandas as pd

import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger

import algorithms.gcg as gcg
import algorithms.autodan as autodan

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
                    
        # Clean up

        del batch_attentions
        del batch_output
        del batch_inputs_embeds
        gc.collect()
        torch.cuda.empty_cache()

        return batch_final
        

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def attention_weight_signal_v1(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    topk,
    logger: experiment_logger.ExperimentLogger,
    *,
    layer_weight_strategy: str = "uniform",
    attention_mask_strategy: str = "payload_only"
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

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def adversarial_opt(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_template: str | typing.List[typing.Dict[str, str]],
    target_output_str: str,
    adversarial_parameters_dict: typing.Dict,
    logger: experiment_logger.ExperimentLogger
):

    init_config = adversarial_parameters_dict["init_config"]
    adv_prefix_init, adv_suffix_init = attack_utility.initialize_adversarial_strings(tokenizer, init_config)
    if isinstance(input_template, str):
        input_tokenized_data = attack_utility.string_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
    elif isinstance(input_template, list):
        input_tokenized_data = attack_utility.conversation_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)


    attack_algorithm = adversarial_parameters_dict["attack_algorithm"]
    if attack_algorithm == "gcg":
        loss_sequences, best_output_sequences = gcg.gcg(model, tokenizer, input_tokenized_data, adversarial_parameters_dict["attack_hyperparameters"], logger)
        logger.log(loss_sequences)
        logger.log(best_output_sequences)
        return loss_sequences, best_output_sequences, None
    elif attack_algorithm == "custom_gcg":
        early_stop = adversarial_parameters_dict.get("early_stop", True)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", True),
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)

        logprobs_sequences, best_output_sequences = gcg.custom_gcg(model,
            tokenizer,
            input_tokenized_data,
            adversarial_parameters_dict["attack_hyperparameters"],
            logger,
            early_stop=early_stop,
            eval_every_step=eval_every_step,
            identical_outputs_before_stop=identical_outputs_before_stop,
            generation_config=generation_config
        )
        logger.log(logprobs_sequences)
        logger.log(best_output_sequences)
        return logprobs_sequences, best_output_sequences
    elif attack_algorithm == "autodan":
        early_stop = adversarial_parameters_dict.get("early_stop", True)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", True)
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)

        logprobs_sequences, best_output_sequences = autodan.autodan(model,
            tokenizer,
            input_tokenized_data,
            adversarial_parameters_dict["attack_hyperparameters"],
            logger,
            early_stop=early_stop,
            eval_every_step=eval_every_step,
            identical_outputs_before_stop=identical_outputs_before_stop,
            generation_config=generation_config
        )
        logger.log(logprobs_sequences)
        logger.log(best_output_sequences)
        return logprobs_sequences, best_output_sequences

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def attack_purplellama_indirect(
    purplellama_data,
    example_num,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    logger: experiment_logger.ExperimentLogger,
    *,
    add_eot_to_target=True
):
    purplellama_example = purplellama_data[example_num]
    input_conversation = [
            {
                "role": "system",
                "content":  purplellama_example["test_case_prompt"]
            },
            {
                "role": "user",
                "content": purplellama_example["user_input_wrapper"]
            }
        ]

    target_string = purplellama_example["target"]
    if add_eot_to_target:
        if "lama" in model.__repr__():
            target_string = target_string + "<|eot_id|>"
    
    initial_config = {
        "strategy_type": "random",
        "prefix_length": 25,
        "suffix_length": 25,
        "seed": int(time.time())
    }

    
    custom_gcg_hyperparameters_1 = {
        "signal_function": attention_weight_signal_v1,
        "signal_kwargs": {
            "layer_weight_strategy": "only_first",
            "attention_mask_strategy": "payload_only"
        },
        "true_loss_function": attention_weight_loss_v1,
        "true_loss_kwargs": {
            "layer_weight_strategy": "only_first",
            "attention_mask_strategy": "payload_only"
        },
        "max_steps": 150,
        "topk": 256,
        "forward_eval_candidates": 512,
    }

    adversarial_parameters_dict_1 = {
        "init_config": initial_config,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": custom_gcg_hyperparameters_1,
        "early_stop": False,
        "eval_every_step": True
    }

    logger.log(adversarial_parameters_dict_1, example_num=example_num)
    loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict_1, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_output_sequences, example_num=example_num)

    custom_gcg_hyperparameters_2 = {
        "signal_function": attention_weight_signal_v1,
        "signal_kwargs": {
            "layer_weight_strategy": "uniform",
            "attention_mask_strategy": "payload_only"
        },
        "true_loss_function": attention_weight_loss_v1,
        "true_loss_kwargs": {
            "layer_weight_strategy": "uniform",
            "attention_mask_strategy": "payload_only"
        },
        "max_steps": 150,
        "topk": 256,
        "forward_eval_candidates": 512,
    }
    adversarial_parameters_dict_2 = {
        "init_config": initial_config,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": custom_gcg_hyperparameters_2,
        "early_stop": False,
        "eval_every_step": True
    }

    logger.log(adversarial_parameters_dict_2, example_num=example_num)
    loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict_2, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_output_sequences, example_num=example_num)

    custom_gcg_hyperparameters_3 = {
        "signal_function": gcg.og_gcg_signal,
        "max_steps": 150,
        "topk": 256,
        "forward_eval_candidates": 512,
    }

    adversarial_parameters_dict_3 = {
        "init_config": initial_config,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": custom_gcg_hyperparameters_3,
        "early_stop": False,
        "eval_every_step": True
    }

    logger.log(adversarial_parameters_dict_3, example_num=example_num)
    loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict_3, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_output_sequences, example_num=example_num)


# def attack_agentharm(
#     attack_idx,
#     agentharm_path = "/data/AgentHarm/benchmark/harmful_behaviors_test_public.json"
# ):
#     with open(agentharm_path) as agentharm_dataset_file:
#         agentharm_dataset = json.load(agentharm_dataset_file)["behaviors"]
    
#     for target_behvaior in agentharm_dataset:

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def attack_advbench(
    advbench_data_path: str,
    example_num: int,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    logger: experiment_logger.ExperimentLogger
):
    adv_bench_data = pd.read_csv(advbench_data_path)
    adv_bench_example = adv_bench_data.iloc[example_num].to_dict()
    malicious_inst = "<ADV_PREFIX>" + adv_bench_example["goal"] + "<ADV_SUFFIX>"
    target_string = adv_bench_example["target"]
    input_conversation = [
        {
            "role": "user",
            "content": malicious_inst
        }
    ]

    initial_config = {
        "strategy_type": "random",
        "prefix_length": 0,
        "suffix_length": 1,
        "seed": int(time.time())
    }
    autodan_hyperparameters = {
        "signal_function": autodan.og_autodan_signal,
        "topk": 4096,
        "autodan_signal_weight": 100,
        "autodan_true_weight": 100,
        "sampling_temperature": 1,
        "max_steps": 200,
        "forward_eval_candidates": 512,
    }
    autodan_adversarial_parameters_dict = {
        "init_config": initial_config,
        "attack_algorithm": "autodan",
        "attack_hyperparameters": autodan_hyperparameters,
        "early_stop": False,
        "eval_every_step": True
    }

    logger.log(autodan_adversarial_parameters_dict, example_num=example_num)
    loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_conversation, target_string, autodan_adversarial_parameters_dict, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_output_sequences, example_num=example_num)
