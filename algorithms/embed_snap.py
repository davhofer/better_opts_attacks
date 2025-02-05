import torch
import transformers
import json
import pickle
import typing
import time
import datetime
import algorithms.autodan as autodan

import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
import algorithms.gcg as gcg


def embed_opt(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Dict,
    embed_snap_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    eval_every_step,
    early_stop,
    eval_initial,
    identical_outputs_before_stop,
    generation_config
):

    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    eval_input_mask: torch.tensor = masks_data["input_mask"]

    true_loss_function = embed_snap_hyperparams.get("true_loss_function", attack_utility.target_logprobs)
    true_loss_kwargs = embed_snap_hyperparams.get("true_loss_kwargs", None)

    logprobs_sequences = []
    embeds_sequences = []
    if eval_initial:
        initial_true_loss = true_loss_function(model, tokenizer, torch.unsqueeze(input_tokens, 0), masks_data, input_tokens[target_mask], logger, **(true_loss_kwargs or {}))
        logger.log(initial_true_loss, step_num=-1)
        logger.log(input_tokens, step_num=-1)
        initial_logprobs = attack_utility.target_logprobs(model, tokenizer, torch.unsqueeze(input_tokens, 0), masks_data, input_tokens[target_mask], logger)
        initial_logprobs = initial_logprobs.item()
        logger.log(initial_logprobs, step_num=-1)
        logprobs_sequences.append(initial_logprobs)
        generated_output_tokens = model.generate(torch.unsqueeze(input_tokens[eval_input_mask].to(model.device), dim=0), attention_mask=torch.unsqueeze(torch.ones(input_tokens[eval_input_mask].to(model.device).shape), dim=0), **generation_config)
        generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
        logger.log(generated_output_string, step_num=-1)

    embedding_map = model.get_input_embeddings()
    inputs_embeds = embedding_map(input_tokens).detach().clone()
    for step_num in range(embed_snap_hyperparams["max_steps"]):
        embeds_sequences.append(inputs_embeds)
        inputs_embeds.requires_grad_()
        logits = model(inputs_embeds=torch.unsqueeze(inputs_embeds, 0), generation_config=attack_utility.DEFAULT_GENERATION_PARAMS).logits
        loss_tensor = attack_utility.UNREDUCED_CE_LOSS(logits[0, target_mask - 1, :], input_tokens[target_mask].to(logits.device)).sum()
        loss_tensor.backward()
        inputs_grads = inputs_embeds.grad
        inputs_embeds.requires_grad_(False)
        inputs_embeds[optim_mask] = inputs_embeds[optim_mask] - embed_snap_hyperparams["step_size"] * inputs_grads[optim_mask]
        inputs_embeds.grad.zero_()
        model.zero_grad()
        if eval_every_step:
            logprobs = attack_utility.target_logprobs_from_embeds(model, tokenizer, torch.unsqueeze(inputs_embeds, 0), masks_data, input_tokens[target_mask], logger)
            logprobs = logprobs.item()
            logger.log(logprobs, step_num=step_num)
            logprobs_sequences.append(logprobs)
            generated_output_tokens = model.generate(inputs_embeds=torch.unsqueeze(inputs_embeds[eval_input_mask, :], dim=0), attention_mask=torch.unsqueeze(torch.ones((len(eval_input_mask),)), dim=0), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens)[0]
            logger.log(generated_output_string, step_num=step_num)
            if early_stop:
                if generated_output_string == tokenizer.decode(input_tokenized_data["tokens"][target_mask]):
                    successive_correct_outputs += 1
                    if successive_correct_outputs >= identical_outputs_before_stop:
                        break
                else:
                    successive_correct_outputs = 0

    return logprobs_sequences, embeds_sequences