import torch
import transformers
import typing
import numpy as np
import attack_utility
import logging
import random

DEBUG_LOGGER = logging.getLogger(__name__)
DEBUG_LOGGER.setLevel(logging.INFO)
DEBUG_LOGGER_FILE_HANDLER = logging.FileHandler(f"logs/defaults/{__name__}")
DEBUG_LOGGER_FORMATTER = logging.Formatter(fmt='[%(asctime)s][%(filename)s][%(callId)s][%(funcName)s][%(levelname)s]: <I> %(message)s </I>', datefmt="%Y-%m-%d %H:%M:%S")
DEBUG_LOGGER_FILE_HANDLER.setFormatter(DEBUG_LOGGER_FORMATTER)
DEBUG_LOGGER.addHandler(DEBUG_LOGGER_FILE_HANDLER)
call_id_dict = {"callId": f"random_{random.randint(0, 2**32)}"}
DEBUG_LOGGER = logging.LoggerAdapter(DEBUG_LOGGER, call_id_dict)
DEBUG_LOGGER.info(f"callId={call_id_dict['callId']}")

GCG_LOSS_FUNCTION = torch.nn.CrossEntropyLoss(reduction="none")
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0,
    "max_new_tokens": 200
}
def gcg(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor],
    gcg_hyperparams: typing.Dict,
    logger,
    *,
    loss_function = GCG_LOSS_FUNCTION,
    eval_every_step = True,
    generation_config = DEFAULT_GENERATION_CONFIG
):

    input_tokens: torch.tensor
    optim_mask: torch.tensor
    target_mask: torch.tensor

    input_tokens, optim_mask, _, _, target_mask, eval_input_mask = input_tokenized_data

    logger.info(f"input_tokens={input_tokens.tolist()}, optim_mask={optim_mask.tolist()}, target_mask={target_mask.tolist()}")
    logger.info(f"gcg_hyperparam={gcg_hyperparams}")

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
        loss_sequences.append(loss_tensor.item())
        logger.info(f"step_num={step_num}, loss_tensor={loss_tensor.item()}")
        grad_optims = - (one_hot_tensor.grad[optim_mask, :])
        best_tokens_indices = grad_optims.topk(gcg_hyperparams["topk"], dim=-1).indices
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
        logger.info(f"step_num={step_num}, current_best_tokens={current_best_tokens.tolist()}")
        best_output_sequences.append(current_best_tokens.clone())
        if eval_every_step:
            current_input_string = tokenizer.decode(current_best_tokens[eval_input_mask])
            logger.info(f"step_num={step_num}, current_input_string={current_input_string}")
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
            logger.info(f"step_num={step_num}, generated_output_string={generated_output_string}")

    return loss_sequences, best_output_sequences

def gcg_abl(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor],
    gcg_hyperparams: typing.Dict,
    logger,
    *,
    loss_function = GCG_LOSS_FUNCTION,
    eval_every_step = True,
    generation_config = DEFAULT_GENERATION_CONFIG
):
    input_tokens: torch.tensor
    optim_mask: torch.tensor
    target_mask: torch.tensor

    input_tokens, optim_mask, _, _, target_mask, eval_input_mask = input_tokenized_data

    logger.info(f"input_tokens={input_tokens.tolist()}, optim_mask={optim_mask.tolist()}, target_mask={target_mask.tolist()}")
    logger.info(f"gcg_hyperparam={gcg_hyperparams}")

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
        # loss_tensor.backward()
        loss_sequences.append(loss_tensor.item())
        logger.info(f"step_num={step_num}, loss_tensor={loss_tensor.item()}")
        # grad_optims = - (one_hot_tensor.grad[optim_mask, :])
        # best_tokens_indices = grad_optims.topk(gcg_hyperparams["topk"], dim=-1).indices # best_tokens_indices.shape == optim_mask.shape[0], topk

        best_tokens_indices = torch.stack([torch.randperm(len(tokenizer))[:gcg_hyperparams["topk"]] for _ in range(optim_mask.shape[0])])
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
        logger.info(f"step_num={step_num}, current_best_tokens={current_best_tokens.tolist()}")
        best_output_sequences.append(current_best_tokens.clone())
        if eval_every_step:
            current_input_string = tokenizer.decode(current_best_tokens[eval_input_mask])
            logger.info(f"step_num={step_num}, current_input_string={current_input_string}")
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
            logger.info(f"step_num={step_num}, generated_output_string={generated_output_string}")

    return loss_sequences, best_output_sequences

def not_gcg(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor],
    gcg_hyperparams: typing.Dict,
    logger,
    *,
    loss_function = GCG_LOSS_FUNCTION,
    eval_every_step = True,
    generation_config = DEFAULT_GENERATION_CONFIG
):

    input_tokens: torch.tensor
    optim_mask: torch.tensor
    target_mask: torch.tensor

    input_tokens, optim_mask, _, _, target_mask, eval_input_mask = input_tokenized_data

    logger.info(f"input_tokens={input_tokens.tolist()}, optim_mask={optim_mask.tolist()}, target_mask={target_mask.tolist()}")
    logger.info(f"gcg_hyperparam={gcg_hyperparams}")

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
        loss_sequences.append(loss_tensor.item())
        logger.info(f"step_num={step_num}, loss_tensor={loss_tensor.item()}")
        grad_optims = (one_hot_tensor.grad[optim_mask, :])
        best_tokens_indices = grad_optims.topk(gcg_hyperparams["topk"], dim=-1).indices
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
        logger.info(f"step_num={step_num}, current_best_tokens={current_best_tokens.tolist()}")
        best_output_sequences.append(current_best_tokens.clone())
        if eval_every_step:
            current_input_string = tokenizer.decode(current_best_tokens[eval_input_mask])
            logger.info(f"step_num={step_num}, current_input_string={current_input_string}")
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
            logger.info(f"step_num={step_num}, generated_output_string={generated_output_string}")

    return loss_sequences, best_output_sequences

