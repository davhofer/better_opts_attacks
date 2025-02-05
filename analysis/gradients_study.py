import transformers
import torch
import shutil
import datetime
import time
import typing
import algorithms.autodan as autodan
import json
import os
import pandas as pd
import itertools

import algorithms.gcg as gcg
import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger

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
    
    TOPK_VALUES = [16, 64, 256]
    ATTACK_DIMENSION_VALUES = [40, 80, 120]


    for ex_index, (topk, attack_dimension) in enumerate(itertools.product(TOPK_VALUES, ATTACK_DIMENSION_VALUES)):
        initial_config = {
            "strategy_type": "random",
            "prefix_length": 0,
            "suffix_length": attack_dimension,
            "seed": int(time.time())
        }
        custom_gcg_hyperparameters_3 = {
            "signal_function": gcg.og_gcg_signal,
            "max_steps": 150,
            "topk": topk,
            "forward_eval_candidates": 512,
        }
        adversarial_parameters_dict_3 = {
            "init_config": initial_config,
            "attack_algorithm": "custom_gcg",
            "attack_hyperparameters": custom_gcg_hyperparameters_3,
            "early_stop": False,
            "eval_every_step": True
        }

        logger.log(adversarial_parameters_dict_3, example_num=example_num, ex_index=ex_index)
        loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict_3, logger)
        logger.log(loss_sequences, example_num=example_num)
        logger.log(best_output_sequences, example_num=example_num)

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

    TOPK_VALUES = [16, 64, 256]
    ATTACK_DIMENSION_VALUES = [40, 80, 120]

    for ex_index, (topk, attack_dimension) in enumerate(itertools.product(TOPK_VALUES, ATTACK_DIMENSION_VALUES)):
        initial_config = {
            "strategy_type": "random",
            "prefix_length": 0,
            "suffix_length": attack_dimension,
            "seed": int(time.time())
        }
        custom_gcg_hyperparameters_3 = {
            "signal_function": gcg.og_gcg_signal,
            "max_steps": 150,
            "topk": topk,
            "forward_eval_candidates": 512,
        }
        adversarial_parameters_dict_3 = {
            "init_config": initial_config,
            "attack_algorithm": "custom_gcg",
            "attack_hyperparameters": custom_gcg_hyperparameters_3,
            "early_stop": False,
            "eval_every_step": True
        }

        logger.log(adversarial_parameters_dict_3, example_num=example_num, ex_index=ex_index)
        loss_sequences, best_output_sequences = adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict_3, logger)
        logger.log(loss_sequences, example_num=example_num)
        logger.log(best_output_sequences, example_num=example_num)

if __name__ == "__main__":
    MODEL_PATH = "/data/models/hf/Meta-Llama-3-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    ADVBENCH_PATH = "data/advbench_harmful_behaviors.csv"    
    EXPT_FOLDER = "logs/runs19"
    if not os.path.exists(EXPT_FOLDER):
        os.mkdir(EXPT_FOLDER)
    shutil.copy(__file__, EXPT_FOLDER)
    
    with open(f"data/purplellama_indirect.json", "r") as purplellama_file:
        purplellama_data = json.load(purplellama_file)
    for i in range(5):
        for rand_restart in range(3):
            expt_id = f"run_{str(datetime.datetime.now()).replace("-","").replace(" ","").replace(":","").replace(".","")}"
            logger = experiment_logger.ExperimentLogger(f"{EXPT_FOLDER}/{expt_id}")
            logger.log(model.__repr__(), example_num=i, rand_restart=rand_restart)
            attack_purplellama_indirect(purplellama_data, i, model, tokenizer, logger)
