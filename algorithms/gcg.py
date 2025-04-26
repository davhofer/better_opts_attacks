import torch
import transformers
import typing
import numpy as np
import utils.attack_utility as attack_utility
import random
import utils.experiment_logger as experiment_logger
import gc


GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS
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
    generation_config = attack_utility.DEFAULT_TEXT_GENERATION_CONFIG
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
        logits = model(inputs_embeds=inputs_embeds).logits
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

        if isinstance(gcg_hyperparams["forward_eval_candidates"], str):
            if gcg_hyperparams["forward_eval_candidates"] == "all":
                num_forward_evals = len(optim_mask) * gcg_hyperparams["topk"] 
        else:
            assert isinstance(gcg_hyperparams["forward_eval_candidates"], int), "Only strings or ints"
            num_forward_evals = gcg_hyperparams["forward_eval_candidates"]

        while len(indices_to_sample) < num_forward_evals:
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
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0).to(model.device), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape).to(model.device), dim=0), **generation_config)
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
    *,
    step_num,
    **kwargs
):
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    logits = model(inputs_embeds=inputs_embeds).logits
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
    eval_initial,
    identical_outputs_before_stop,
    generation_config,
    use_kv_caching = True
):

    logger.log(input_tokenized_data)

    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    eval_input_mask: torch.tensor = masks_data["input_mask"]

    signal_function = custom_gcg_hyperparams.get("signal_function", og_gcg_signal)
    true_loss_function = custom_gcg_hyperparams.get("true_loss_function", attack_utility.target_logprobs)
    substitution_validity_function = custom_gcg_hyperparams.get("substitution_validity_function", None)
    signal_kwargs = custom_gcg_hyperparams.get("signal_kwargs", None)
    true_loss_kwargs = custom_gcg_hyperparams.get("true_loss_kwargs", None)

    current_best_tokens = input_tokens.clone()
    best_output_sequences = []
    logprobs_sequences = []
    successive_correct_outputs = 0

    if eval_initial:
        initial_true_loss = true_loss_function(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger, **(true_loss_kwargs or {}))
        logger.log(initial_true_loss, step_num=-1)
        best_output_sequences.append(current_best_tokens.clone())
        logger.log(current_best_tokens, step_num=-1)
        initial_logprobs = attack_utility.target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
        initial_logprobs = initial_logprobs.item()
        logger.log(initial_logprobs, step_num=-1)
        logprobs_sequences.append(initial_logprobs)
        generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0).to(model.device), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0).to(model.device), **generation_config)
        generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
        logger.log(generated_output_string, step_num=-1)

    step_num = 0

    best_tokens_chunk = []
    true_losses_chunk = []
    substitution_data_chunk = []
    current_best_true_loss_chunk = []
    current_best_tokens_chunk = []
    logprobs_chunk = []
    generated_output_string_chunk = []

    if use_kv_caching:
        import pdb
        pdb.set_trace()
        min_static_index = min(optim_mask) - 1
        most_common_input_tokens = input_tokens[:min_static_index]
        outputs = model(
            input_ids=torch.unsqueeze(most_common_input_tokens, dim=0).to(model.device),
            use_cache=True
        )
        cache = outputs.past_key_values
        cache = attack_utility.truncate_cache(cache, min_static_index)
    else:
        min_static_index = 0
        cache = None
    
    for step_num in range(custom_gcg_hyperparams["max_steps"]):
        
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

        if true_loss_kwargs is None:
            true_loss_kwargs = {}
        true_loss_kwargs["past_key_values"] = cache
        true_loss_kwargs["min_static_index"] = min_static_index
        
        true_losses = true_loss_function(model, tokenizer, substitution_data, masks_data, input_tokens[target_mask], logger, **true_loss_kwargs)
        true_losses_chunk.append(true_losses)
        current_best_true_loss = true_losses[torch.argmin(true_losses)]
        current_best_true_loss_chunk.append(current_best_true_loss)
        current_best_tokens = substitution_data[torch.argmin(true_losses)].clone()
        current_best_tokens_chunk.append(current_best_tokens)
        best_output_sequences.append(current_best_tokens.clone())
        if eval_every_step:
            logprobs = attack_utility.target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
            logprobs = logprobs.item()
            logprobs_chunk.append(logprobs)
            logprobs_sequences.append(logprobs)
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0).to(model.device), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0).to(model.device), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
            generated_output_string_chunk.append(generated_output_string)
            if early_stop:
                if generated_output_string == tokenizer.decode(input_tokenized_data["tokens"][target_mask]):
                    successive_correct_outputs += 1
                    if successive_correct_outputs >= identical_outputs_before_stop:
                        break
                else:
                    successive_correct_outputs = 0
    
        if (step_num + 1) % 1 == 0:
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
    return logprobs_sequences, best_output_sequences
