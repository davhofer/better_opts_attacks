import torch
import transformers
import typing
import numpy as np
import attack_utility
import random
import experiment_logger
import gc


GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS
DEFAULT_GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 200
}
@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def gcg(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Dict[str, torch.tensor],
    gcg_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    loss_function = GCG_LOSS_FUNCTION,
    eval_every_step = True,
    generation_config = DEFAULT_GENERATION_CONFIG
):

    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    optim_mask: torch.tensor = input_tokenized_data["masks"]["optim_mask"]
    target_mask: torch.tensor = input_tokenized_data["masks"]["target_mask"]
    eval_input_mask: torch.tensor = input_tokenized_data["masks"]["input_mask"]

    gradient_signal = gcg_hyperparams["gradient_signal"]

    loss_sequences = []
    current_best_tokens = input_tokens.clone()
    best_output_sequences = [current_best_tokens]
    for step_num in range(gcg_hyperparams["max_steps"]):
        one_hot_tensor = torch.nn.functional.one_hot(current_best_tokens.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
        one_hot_tensor.requires_grad_()
        embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
        inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
        logits = model.forward(inputs_embeds=inputs_embeds).logits
        loss_tensor = loss_function(logits[0, target_mask - 1, :], input_tokens[target_mask].to(logits.device)).sum()
        loss_tensor.backward()
        loss_val = loss_tensor.item()
        loss_sequences.append(loss_val)
        logger.log(loss_val, step_num=step_num)

        if gradient_signal == "neg_grad":
            grad_optims = - (one_hot_tensor.grad[optim_mask, :])
            best_tokens_indices = grad_optims.topk(gcg_hyperparams["topk"], dim=-1).indices
        elif gradient_signal == "random":
            best_tokens_indices = torch.stack([torch.randperm(len(tokenizer))[:gcg_hyperparams["topk"]] for _ in range(optim_mask.shape[0])])
        elif gradient_signal == "pos_grad":
            grad_optims = (one_hot_tensor.grad[optim_mask, :])
            best_tokens_indices = grad_optims.topk(gcg_hyperparams["topk"], dim=-1).indices
        else:
            raise ValueError(f"gradient_signal not recognized")

        indices_to_sample = set()
        while len(indices_to_sample) < gcg_hyperparams["forward_eval_candidates"]:
            first_coordinate = torch.randint(0, best_tokens_indices.shape[0], (1,)).to(torch.int32).item()
            second_coordinate = torch.randint(0, best_tokens_indices.shape[1], (1,)).to(torch.int32).item()
            if (first_coordinate, second_coordinate) in indices_to_sample:
                continue
            indices_to_sample.add((first_coordinate, second_coordinate))

        substitutions_list = []
        for index_to_sample in list(indices_to_sample):
            random_substitution_make = current_best_tokens.clone()
            random_substitution_make[optim_mask[index_to_sample[0]]] = best_tokens_indices[index_to_sample]
            substitutions_list.append(random_substitution_make)
        
        substitution_data = torch.stack(substitutions_list)
        inference_logits = attack_utility.bulk_logits(model, substitution_data)
        true_losses = loss_function(inference_logits[:, target_mask - 1, :].transpose(1, 2), input_tokens.repeat(inference_logits.shape[0], 1)[:, target_mask]).sum(dim=1)
        current_best_tokens = substitution_data[torch.argmin(true_losses)].clone()
        logger.log(current_best_tokens, step_num=step_num)
        best_output_sequences.append(current_best_tokens.clone())
        if eval_every_step:
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
            logger.log(generated_output_string, step_num=step_num)

    return loss_sequences, best_output_sequences

def og_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
):
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    logits = model.forward(inputs_embeds=inputs_embeds).logits
    loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).sum()
    loss_tensor.backward()
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

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    logits = model.forward(inputs_embeds=inputs_embeds).logits
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
):
    optim_mask: torch.tensor = masks_data["optim_mask"]

    best_tokens_indices = torch.stack([torch.randperm(len(tokenizer))[:gcg_topk] for _ in range(optim_mask.shape[0])])
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
    identical_outputs_before_stop,
    generation_config
):

    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    eval_input_mask: torch.tensor = masks_data["input_mask"]

    signal_function = custom_gcg_hyperparams.get("signal_function", og_gcg_signal)
    true_loss_function = custom_gcg_hyperparams.get("true_loss_function", attack_utility.target_logprobs)
    substitution_validity_function = custom_gcg_hyperparams.get("substitution_function", None)
    signal_kwargs = custom_gcg_hyperparams.get("signal_kwargs", None)
    true_loss_kwargs = custom_gcg_hyperparams.get("true_loss_kwargs", None)

    current_best_tokens = input_tokens.clone()
    best_output_sequences = []
    logprobs_sequences = []
    successive_correct_outputs = 0

    initial_true_loss = true_loss_function(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger, **(true_loss_kwargs or {}))
    logger.log(initial_true_loss, step_num=-1)
    best_output_sequences.append(current_best_tokens.clone())
    logger.log(current_best_tokens, step_num=-1)
    initial_logprobs = attack_utility.target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
    initial_logprobs = initial_logprobs.item()
    logger.log(initial_logprobs, step_num=-1)
    logprobs_sequences.append(initial_logprobs)
    generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0), **generation_config)
    generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
    logger.log(generated_output_string, step_num=-1)

    for step_num in range(custom_gcg_hyperparams["max_steps"]):

        best_tokens_indices = signal_function(model, tokenizer, current_best_tokens, masks_data, custom_gcg_hyperparams["topk"], logger, **(signal_kwargs or {}))
        
        indices_to_sample = set()
        indices_to_exclude = set()
        substitutions_set = set()
        while len(indices_to_sample) < custom_gcg_hyperparams["forward_eval_candidates"]:
            first_coordinate = torch.randint(0, best_tokens_indices.shape[0], (1,)).to(torch.int32).item()
            second_coordinate = torch.randint(0, best_tokens_indices.shape[1], (1,)).to(torch.int32).item()
            if (first_coordinate, second_coordinate) in indices_to_sample:
                continue
            if (first_coordinate, second_coordinate) in indices_to_exclude:
                continue
            random_substitution_make = current_best_tokens.clone()
            random_substitution_make[optim_mask[first_coordinate]] = best_tokens_indices[(first_coordinate, second_coordinate)]
            if (substitution_validity_function is None) or (substitution_validity_function(random_substitution_make)):
                indices_to_sample.add((first_coordinate, second_coordinate))
                substitutions_set.add(random_substitution_make)
            else:
                indices_to_exclude.add((first_coordinate, second_coordinate))
        substitution_data = torch.stack(list(substitutions_set))
        del best_tokens_indices
        gc.collect()
        torch.cuda.empty_cache()
        true_losses = true_loss_function(model, tokenizer, substitution_data, masks_data, input_tokens[target_mask], logger, **(true_loss_kwargs or {}))
        current_best_true_loss = true_losses[torch.argmin(true_losses)]
        logger.log(current_best_true_loss, step_num=step_num)
        current_best_tokens = substitution_data[torch.argmin(true_losses)].clone()
        logger.log(current_best_tokens, step_num=step_num)
        best_output_sequences.append(current_best_tokens.clone())
        if eval_every_step:
            logprobs = attack_utility.target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
            logprobs = logprobs.item()
            logger.log(logprobs, step_num=step_num)
            logprobs_sequences.append(logprobs)
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
            logger.log(generated_output_string, step_num=step_num)
            if early_stop:
                if generated_output_string == tokenizer.decode(input_tokenized_data["tokens"][target_mask]):
                    successive_correct_outputs += 1
                    if successive_correct_outputs >= identical_outputs_before_stop:
                        break
                else:
                    successive_correct_outputs = 0

    logger.log(successive_correct_outputs, num_steps=step_num)
    return logprobs_sequences, best_output_sequences
