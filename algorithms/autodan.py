import torch
import transformers
import typing
import utils.experiment_logger as experiment_logger
import gc
import utils.attack_utility as attack_utility
import algorithms.gcg as gcg
import copy
import random

AUTODAN_LOSS_FUNCTION = None

def _add_adv_token_and_extend(model, tokenizer, current_tokens, original_masks_data, substitution_validity_function = None):
    last_token_optim_idx = original_masks_data["optim_mask"][-1]
    frozen_tensor_pre = current_tokens[:last_token_optim_idx + 1]
    frozen_tensor_post = current_tokens[last_token_optim_idx + 1:]
    if substitution_validity_function is not None:
        while True:
            temp_next_tokens = torch.cat((
                frozen_tensor_pre,
                torch.unsqueeze(torch.tensor(random.randint(0, len(tokenizer.vocab))), dim=0),
                frozen_tensor_post
            ))
            if substitution_validity_function(temp_next_tokens):
                break
    else:
        temp_next_tokens = torch.cat((
            frozen_tensor_pre,
            torch.unsqueeze(torch.tensor(random.randint(0, len(tokenizer.vocab))), dim=0),
            frozen_tensor_post
        ))
    
    new_masks_data = copy.deepcopy(original_masks_data)
    new_masks_data["optim_mask"] = torch.cat([original_masks_data["optim_mask"], torch.tensor([original_masks_data["optim_mask"][-1] + 1])])
    new_masks_data["suffix_mask"] = torch.cat([original_masks_data["suffix_mask"], torch.tensor([original_masks_data["optim_mask"][-1] + 1])])
    new_masks_data["target_mask"] = original_masks_data["target_mask"] + 1
    new_masks_data["input_mask"] = torch.cat([original_masks_data["input_mask"], torch.tensor([original_masks_data["input_mask"][-1] + 1])])
    new_masks_data["content_mask"] = torch.cat([original_masks_data["content_mask"], torch.tensor([original_masks_data["content_mask"][-1] + 1])])
    # TODO: Add logic for control mask
    return temp_next_tokens, new_masks_data

def og_autodan_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    autodan_topk: int,
    autodan_signal_weight: float,
    logger: experiment_logger.ExperimentLogger,    
):
    token_to_optim = masks_data["optim_mask"][-1]
    target_mask: torch.tensor = masks_data["target_mask"]

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    logits = model.forward(inputs_embeds=inputs_embeds).logits
    gcg_loss_tensor = gcg.GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).mean()
    gcg_loss_tensor.backward()
    gcg_grad_optims = - (one_hot_tensor.grad[token_to_optim, :])
    readability_loss = - torch.log_softmax(model.forward(input_ids=torch.unsqueeze(input_points[:token_to_optim], dim=0)).logits[0, -1], dim=-1)
    final_loss = autodan_signal_weight * gcg_grad_optims + readability_loss
    best_tokens_candidates = final_loss.topk(autodan_topk, dim=-1).indices
    return torch.unsqueeze(best_tokens_candidates, dim=0)

def og_autodan_true_loss(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    autodan_true_weight,
    logger: experiment_logger.ExperimentLogger,
):
    token_to_optim = masks_data["optim_mask"][-1]
    target_mask: torch.tensor = masks_data["target_mask"]

    target_loss = attack_utility.target_logprobs(model, tokenizer, input_points, masks_data, input_points[0, target_mask], logger)
    target_loss /= target_mask.shape[0]

    readability_losses = []
    accum_idx = 0
    for logit_piece in attack_utility.bulk_logits_iter(model, input_points[:, :token_to_optim]):
        readability_loss = attack_utility.UNREDUCED_CE_LOSS(torch.transpose(logit_piece, 1, 2)[..., -1], input_points[accum_idx:accum_idx + logit_piece.shape[0], token_to_optim].to(logit_piece.device))
        accum_idx += logit_piece.shape[0]
        readability_losses.append(readability_loss)
    readability_loss = torch.cat(readability_losses)

    final_loss = autodan_true_weight * target_loss + readability_loss
    return final_loss

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def autodan(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Dict,
    autodan_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    eval_every_step,
    early_stop,
    identical_outputs_before_stop,
    generation_config    
):
    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    masks_data = copy.deepcopy(input_tokenized_data["masks"])
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    eval_input_mask: torch.tensor = masks_data["input_mask"]

    signal_function = autodan_hyperparams.get("signal_function", og_autodan_signal)
    true_loss_function = autodan_hyperparams.get("true_loss_function", og_autodan_true_loss)
    substitution_validity_function = autodan_hyperparams.get("substitution_function", None)
    signal_kwargs = autodan_hyperparams.get("signal_kwargs", None)
    true_loss_kwargs = autodan_hyperparams.get("true_loss_kwargs", None)
    topk = autodan_hyperparams.get("autodan_topk", 4096)
    signal_weight = autodan_hyperparams.get("autodan_signal_weight", 0.1)
    true_weight = autodan_hyperparams.get("autodan_true_weight", 0.1)

    current_best_tokens = input_tokens.clone()
    best_output_sequences = []
    logprobs_sequences = []
    successive_correct_outputs = 0

    initial_true_loss = true_loss_function(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, autodan_hyperparams["autodan_true_weight"], logger, **(true_loss_kwargs or {}))
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

    for step_num in range(autodan_hyperparams["max_steps"]):
        optim_mask: torch.tensor = torch.tensor([masks_data["optim_mask"][-1]])
        target_mask: torch.tensor = masks_data["target_mask"]
        eval_input_mask: torch.tensor = masks_data["input_mask"]
        
        best_tokens_indices = signal_function(model, tokenizer, current_best_tokens, masks_data, autodan_hyperparams["topk"], autodan_hyperparams["autodan_signal_weight"], logger, **(signal_kwargs or {}))
        
        indices_to_sample = set()
        indices_to_exclude = set()
        substitutions_set = set([current_best_tokens])
        while len(indices_to_sample) < autodan_hyperparams["forward_eval_candidates"] - 1:
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
        true_losses = true_loss_function(model, tokenizer, substitution_data, masks_data, autodan_hyperparams["autodan_true_weight"], logger, **(true_loss_kwargs or {}))
        current_best_true_loss = true_losses[torch.argmin(true_losses)]
        logger.log(current_best_true_loss, step_num=step_num)
        current_best_tokens = substitution_data[torch.argmin(true_losses)].clone()
        logger.log(current_best_tokens, step_num=step_num)
        best_output_sequences.append(current_best_tokens.clone())

        if eval_every_step:
            logprobs = attack_utility.target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, current_best_tokens[target_mask], logger)
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
        
        current_best_tokens, masks_data = _add_adv_token_and_extend(model, tokenizer, current_best_tokens, masks_data, substitution_validity_function)

    logger.log(masks_data)
    return logprobs_sequences, best_output_sequences