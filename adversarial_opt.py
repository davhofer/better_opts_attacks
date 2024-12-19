import transformers
import torch
import typing
import datetime
import json

import gcg
import attack_utility
import experiment_logger


@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def adversarial_opt(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_string_template: str,
    target_output_str: str,
    adversarial_parameters_dict: typing.Dict,
    logger: experiment_logger.ExperimentLogger
):

    init_config = adversarial_parameters_dict["init_config"]
    adv_prefix_init, adv_suffix_init = attack_utility.initialize_adversarial_strings(tokenizer, init_config)
    input_tokenized_data = attack_utility.string_masks(tokenizer, input_string_template, adv_prefix_init, adv_suffix_init, target_output_str)
    
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
    input_string_template = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content":  purplellama_example["test_case_prompt"]
            },
            {
                "role": "user",
                "content": purplellama_example["user_input_wrapper"]
            }
        ],
        add_generation_prompt=True,
        tokenize=False
    )
    target_string = purplellama_example["target"]
    if add_eot_to_target:
        if "lama" in model.__repr__():
            target_string = target_string + "<|eot_id|>"
    
    initial_config_1 = {
        "strategy_type": "random",
        "prefix_length": 25,
        "suffix_length": 25,
        "seed": 0
    }
    
    custom_gcg_hyperparameters_2 = {
        "max_steps": 400,
        "topk": 256,
        "forward_eval_candidates": 512
    }

    # custom_gcg_hyperparameters_3 = {
    #     "max_steps": 400,
    #     "topk": 256,
    #     "forward_eval_candidates": 512
    # }

    adversarial_parameters_dict_2 = {
        "init_config": initial_config_1,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": custom_gcg_hyperparameters_2
    }

    # adversarial_parameters_dict_3 = {
    #     "init_config": initial_config_1,
    #     "attack_algorithm": "custom_gcg",
    #     "attack_hyperparameters": custom_gcg_hyperparameters_3
    # }



    logger.log(adversarial_parameters_dict_2, example_num=example_num)
    loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_string_template, target_string, adversarial_parameters_dict_2, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_output_sequences, example_num=example_num)

    # logger.log(adversarial_parameters_dict_3, example_num=example_num)
    # loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_string_template, target_string, adversarial_parameters_dict, logger)
    # logger.log(loss_sequences, example_num=example_num)
    # logger.log(best_output_sequences, example_num=example_num)



if __name__ == "__main__":
    MODEL_PATH = "/data/models/hf/Meta-Llama-3-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    for i in range(2, 5):
        expt_id = f"run_{str(datetime.datetime.now()).replace("-","").replace(" ","").replace(":","").replace(".","")}"
        logger = experiment_logger.ExperimentLogger(f"logs/runs/{expt_id}")
        logger.log(model.__repr__(), example_num=i)
        attack_purplellama_indirect(i, model, tokenizer, logger)