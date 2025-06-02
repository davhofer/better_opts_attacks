import transformers
import torch
import pandas as pd
import json
import time
import itertools
import datetime
import shutil
import os
import gc
import traceback
import random
import copy
import multiprocessing
import sys
import peft
import argparse

import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
import algorithms.gcg as gcg
from secalign_refactored import secalign
import adversarial_opt
import algorithms.losses_experimental as losses_experimental


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
        "prefix_length": 0,
        "suffix_length": 25,
    }

    input_tokenized_data, true_init_config = attack_utility.generate_valid_input_tokenized_data(tokenizer, input_conversation, target_string, initial_config, logger)
    logger.log(true_init_config)

    guided_hyperparameters = {
        "signal_function": gcg.og_gcg_signal,
        "max_steps": 500,
        "topk": 256,
        "forward_eval_candidates": 512
    }    

    guided_parameters_dict = {
        "input_tokenized_data": input_tokenized_data,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": guided_hyperparameters,
        "early_stop": False,
        "eval_every_step": False,
        "to_cache_logits": True,
        "to_cache_attentions": True
    }

    logger.log(guided_parameters_dict, example_num=example_num)
    loss_sequences_guided, best_outputs_sequences_guided = adversarial_opt.adversarial_opt(model, tokenizer, input_conversation, target_string, guided_parameters_dict, logger)
    logger.log(loss_sequences_guided, example_num=example_num)
    logger.log(best_outputs_sequences_guided, example_num=example_num)

    unguided_hyperparameters = {
        "signal_function": gcg.rand_gcg_signal,
        "max_steps": 500,
        "topk": 256,
        "forward_eval_candidates": 512
    }    

    unguided_parameters_dict = {
        "input_tokenized_data": input_tokenized_data,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": unguided_hyperparameters,
        "early_stop": False,
        "eval_every_step": False,
        "to_cache_logits": True,
        "to_cache_attentions": True
    }

    logger.log(unguided_parameters_dict, example_num=example_num)
    loss_sequences_unguided, best_outputs_sequences_unguided = adversarial_opt.adversarial_opt(model, tokenizer, input_conversation, target_string, unguided_parameters_dict, logger)
    logger.log(loss_sequences_unguided, example_num=example_num)
    logger.log(best_outputs_sequences_unguided, example_num=example_num)

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
    malicious_inst = attack_utility.ADV_PREFIX_INDICATOR + adv_bench_example["goal"] + attack_utility.ADV_SUFFIX_INDICATOR
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
        "suffix_length": 25,
        "seed": int(time.time())
    }

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
    loss_sequences, best_output_sequences = adversarial_opt.adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict_3, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_output_sequences, example_num=example_num)

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def attack_secalign_model(
    example_target,
    model,
    tokenizer,
    frontend_delimiters,
    logger: experiment_logger.ExperimentLogger,
    *,
    convert_to_secalign_format = True
):
    
    input_conv = example_target["input_conv"]
    target = example_target["target"]

    prompt_template = secalign.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
    inst_delm, data_delm, resp_delm = secalign.DELIMITERS[frontend_delimiters]

    if convert_to_secalign_format:
        input_conv = secalign._convert_to_secalign_format(input_conv, prompt_template, tokenizer)
    else:
        input_conv = [
            {
                "role": input_conv[0]["role"],
                "content": input_conv[0]["content"]
            },
            {
                "role": input_conv[1]["role"],
                "content": input_conv[1]["content"] + " " + attack_utility.ADV_PREFIX_INDICATOR + " " +  secalign.SECALIGN_COMMON_INSTRUCTION  + " " + attack_utility.ADV_SUFFIX_INDICATOR
            }
        ]

    initial_config = {
        "strategy_type": "random",
        "prefix_length": 0,
        "suffix_length": 20,
        "prefix_filter": secalign.secalign_filter,
        "suffix_filter": secalign.secalign_filter,
        "filter_metadata": {
            "tokenizer": tokenizer
        }
    }

    input_tokenized_data, true_init_config = attack_utility.generate_valid_input_tokenized_data(tokenizer, input_conv, target, initial_config, logger)
    logger.log(true_init_config)

    gcg_guided_params = {
        "signal_function": gcg.og_gcg_signal,
        "max_steps": 500,
        "topk": 256,
        "forward_eval_candidates": 512,
        "substitution_validity_function": secalign.secalign_filter
    }
    adversarial_parameters_dict_guided = {
        "input_tokenized_data": input_tokenized_data,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": gcg_guided_params,
        "early_stop": False,
        "eval_every_step": False,
        "to_cache_logits": True,
        "to_cache_attentions": True,
    }


    logger.log(adversarial_parameters_dict_guided)
    loss_sequences_guided, best_output_sequences_guided = adversarial_opt.adversarial_opt(model, tokenizer, input_conv, target, adversarial_parameters_dict_guided, logger)
    logger.log(loss_sequences_guided)
    logger.log(best_output_sequences_guided)
    final_inputs_strings_guided = tokenizer.batch_decode(best_output_sequences_guided, clean_up_tokenization_spaces=False)
    logger.log(final_inputs_strings_guided)

    del loss_sequences_guided, best_output_sequences_guided, final_inputs_strings_guided
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    gcg_unguided_params = {
        "signal_function": gcg.rand_gcg_signal,
        "max_steps": 500,
        "topk": 256,
        "forward_eval_candidates": 512,
        "substitution_validity_function": secalign.secalign_filter
    }
    adversarial_parameters_dict_unguided = {
        "input_tokenized_data": input_tokenized_data,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": gcg_unguided_params,
        "early_stop": False,
        "eval_every_step": False,
        "to_cache_logits": True,
        "to_cache_attentions": True,
    }


    logger.log(adversarial_parameters_dict_unguided)
    loss_sequences_unguided, best_output_sequences_unguided = adversarial_opt.adversarial_opt(model, tokenizer, input_conv, target, adversarial_parameters_dict_unguided, logger)
    logger.log(loss_sequences_unguided)
    logger.log(best_output_sequences_unguided)
    final_inputs_strings_unguided = tokenizer.batch_decode(best_output_sequences_unguided, clean_up_tokenization_spaces=False)
    logger.log(final_inputs_strings_unguided)


def run_secalign_eval_on_single_gpu(expt_folder_prefix: str, self_device_idx, example_targets, **kwargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(self_device_idx)
    expt_folder = f"{expt_folder_prefix}/expt_{str(self_device_idx)}"
    if not os.path.exists(expt_folder):
        os.mkdir(expt_folder)
    shutil.copy(__file__, expt_folder)
    model_name = "meta-llama-instruct"
    defence = "secalign"
    torch.cuda.set_device(self_device_idx)
    try:
        model, tokenizer, frontend_delimiters = secalign.maybe_load_secalign_defended_model(model_name, defence, device_map=f"cuda:{str(self_device_idx)}", torch_dtype=torch.float16, attn_implementation="eager")
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        traceback.print_exc()
    
    for example_num, example_target in enumerate(example_targets):
        now_str = str(datetime.datetime.now()).replace("-", "").replace(" ", "").replace(":", "").replace(".", "")
        expt_id = f"run_{now_str}"
        logger = experiment_logger.ExperimentLogger(f"{expt_folder}/{expt_id}")
        logger.log(model_name, example_num=example_num)
        example_target = {
            "input_conv": example_target,
            "target": secalign.SECALIGN_HARD_TARGETS[0]
        }
        logger.log(example_target)
        attack_secalign_model(example_target, model, tokenizer, frontend_delimiters, logger, convert_to_secalign_format=False)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Script with GPU device selection.')
    # parser.add_argument(
    #     '--device', 
    #     type=int, 
    #     default=0, 
    #     help='GPU device index to use (default: 0)'
    # )

    # parser.set_defaults(multiprocess=False)

    # args = parser.parse_args()


    # distinct_ppl_examples_indices = [0, 3, 7] # 1, 2, 4 excluded due to large length
    # with open(f"data/purplellama_indirect.json", "r") as purplellama_data:
    #     purplellama_data = json.load(purplellama_data)
    
    # distinct_ppl_examples = [purplellama_data[index] for index in distinct_ppl_examples_indices]

    # LLAMA_3_PATH = "secalign_refactored/secalign_models/meta-llama/Meta-Llama-3-8B-Instruct"
    # MISTRAL_PATH = "secalign_refactored/secalign_models/mistralai/Mistral-7B-Instruct-v0.1"
    

    # MODEL_PATH = MISTRAL_PATH
    # EXPT_FOLDER_PREFIX = "logs/purplellama_gcg_comparison_mistral"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # os.makedirs(EXPT_FOLDER_PREFIX, exist_ok=True)
    # model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=f"cuda:0")
    # tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=f"cuda:0")
    
    model_name = "mistralai-instruct"
    defence = "secalign"
    
    try:
        model, tokenizer, frontend_delimiters = secalign.maybe_load_secalign_defended_model(model_name, defence, device_map=f"cuda:1", torch_dtype=torch.float16, attn_implementation="eager")
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        traceback.print_exc()

    EXPT_FOLDER_PREFIX = "logs/secalign_gcg_comp_mistral"

    with open(secalign.ALPACAFARM_DATASET_PATH, "r") as alpacaeval_file:
        alpacaeval = json.load(alpacaeval_file)
    alpacaeval = [x for x in alpacaeval if ((x["input"] != "") and (x["datasplit"] == "eval"))]
    alpacaeval_convs_raw = [
        [
            {
                "role": "system",
                "content": x["instruction"]
            },
            {
                "role": "user",
                "content": x["input"]
            }
        ]
        for x in alpacaeval
    ]

    distinct_alpacafarm_examples_indices = random.sample(alpacaeval_convs_raw, 3)
    for example_num, example_target in enumerate(distinct_alpacafarm_examples_indices):
        for num_iter in range(3):
            now_str = str(datetime.datetime.now()).replace("-", "").replace(" ", "").replace(":", "").replace(".", "")
            expt_id = f"run_{now_str}"
            logger = experiment_logger.ExperimentLogger(f"{EXPT_FOLDER_PREFIX}/{expt_id}_{example_num}_{num_iter}")
            logger.log(example_num)
        
            example_target_conv = {
                "input_conv": example_target,
                "target": secalign.SECALIGN_HARD_TARGETS[0]
            }
            logger.log(example_target_conv)
            attack_secalign_model(example_target_conv, model, tokenizer, frontend_delimiters, logger, convert_to_secalign_format=False)
            gc.collect()
            torch.cuda.empty_cache()
