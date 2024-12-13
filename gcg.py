import torch
import transformers
import typing
import numpy as np
import attack_utility
import random
import experiment_logger


GCG_LOSS_FUNCTION = torch.nn.CrossEntropyLoss(reduction="none")
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0,
    "max_new_tokens": 200
}
@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def gcg(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor],
    gcg_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    loss_function = GCG_LOSS_FUNCTION,
    eval_every_step = True,
    generation_config = DEFAULT_GENERATION_CONFIG
):

    input_tokens: torch.tensor
    optim_mask: torch.tensor
    target_mask: torch.tensor

    input_tokens, optim_mask, _, _, target_mask, eval_input_mask = input_tokenized_data
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