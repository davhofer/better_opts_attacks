import torch
import transformers
import json
import pickle
import typing
import time
import datetime

import attack_utility
import experiment_logger
import gcg

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
        return loss_sequences, best_output_sequences
    elif attack_algorithm == "custom_gcg":
        logprobs_sequences, best_output_sequences = gcg.custom_gcg(model, tokenizer, input_tokenized_data, adversarial_parameters_dict["attack_hyperparameters"], logger)
        logger.log(logprobs_sequences)
        logger.log(best_output_sequences)
        return logprobs_sequences, best_output_sequences
    elif attack_algorithm == "embed_snap":
        logprobs_sequences, best_output_sequences = embed_snap(model, tokenizer, input_tokenized_data, adversarial_parameters_dict["attack_hyperparameters"], logger)
        return logprobs_sequences, best_output_sequences

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def attack_purplellama_indirect(example_num,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    logger: experiment_logger.ExperimentLogger,
    *,
    add_eot_to_target=True
):
    with open(f"data/purplellama_indirect.json", "r") as purplellama_indirect_file:
        purplellama_data = json.load(purplellama_indirect_file)
    
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
    
    initial_config_1 = {
        "strategy_type": "random",
        "prefix_length": 25,
        "suffix_length": 25,
        "seed": int(time.time())
    }

    embed_snap_hyperparameters = {
        "step_size": 0.01,
    }    

    adversarial_parameters_dict_4 = {
        "init_config": initial_config_1,
        "attack_algorithm": "embed_snap",
        "attack_hyperparameters": embed_snap_hyperparameters
    }


    logger.log(adversarial_parameters_dict_4, example_num=example_num)
    loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict_4, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_output_sequences, example_num=example_num)

def approximate_closest_neighbour()

GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS
def embed_snap(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Dict,
    embed_snap_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger
):

    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    eval_input_mask: torch.tensor = masks_data["input_mask"]

    embedding_map = model.get_input_embeddings()
    inputs_embeds = embedding_map(input_tokens)
    inputs_embeds.retain_grad()
    logits = model.forward(inputs_embeds=torch.unsqueeze(inputs_embeds, 0)).logits
    loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_tokens[target_mask].to(logits.device)).sum()
    loss_tensor.backward()
    inputs_grads = inputs_embeds.grad
    new_embeds = inputs_embeds.clone()
    new_embeds[optim_mask] = inputs_embeds[optim_mask] - embed_snap_hyperparams["step_size"] * inputs_grads[optim_mask]
    return

MODEL_PATH = "/data/models/hf/Llama-3.2-1B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
model.generation_config.pad_token_id = tokenizer.pad_token_id

expt_id = f"run_{str(datetime.datetime.now()).replace("-","").replace(" ","").replace(":","").replace(".","")}"
logger = experiment_logger.ExperimentLogger(f"logs/debug_logs/{expt_id}")
logger.log(model.__repr__())
attack_purplellama_indirect(0, model, tokenizer, logger)
