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

    for topk in [1, 4, 16, 64]:    
        initial_config_1 = {
            "strategy_type": "random",
            "prefix_length": 0,
            "suffix_length": 25,
            "seed": int(time.time())
        }

        custom_gcg_hyperparameters = {
            "max_steps": 150,
            "topk": topk,
            "forward_eval_candidates": "all"
        }    

        adversarial_parameters_dict = {
            "init_config": initial_config_1,
            "attack_algorithm": "custom_gcg",
            "attack_hyperparameters": custom_gcg_hyperparameters,
            "early_stop": False
        }

        logger.log(adversarial_parameters_dict, example_num=example_num)
        loss_sequences, best_outputs_sequences = adversarial_opt.adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict, logger)
        logger.log(loss_sequences, example_num=example_num)
        logger.log(best_outputs_sequences, example_num=example_num)

    initial_config_1 = {
        "strategy_type": "random",
        "prefix_length": 25,
        "suffix_length": 25,
        "seed": int(time.time())
    }

    custom_gcg_hyperparameters = {
        "max_steps": 150,
        "topk": 256,
        "forward_eval_candidates": 512
    }    

    adversarial_parameters_dict = {
        "init_config": initial_config_1,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": custom_gcg_hyperparameters,
        "early_stop": False
    }

    logger.log(adversarial_parameters_dict, example_num=example_num)
    loss_sequences, best_outputs_sequences = adversarial_opt.adversarial_opt(model, tokenizer, input_conversation, target_string, adversarial_parameters_dict, logger)
    logger.log(loss_sequences, example_num=example_num)
    logger.log(best_outputs_sequences, example_num=example_num)

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
        "prefix_length": 2,
        "suffix_length": 20,
        "prefix_filter": secalign.secalign_filter,
        "suffix_filter": secalign.secalign_filter,
        "filter_metadata": {
            "tokenizer": tokenizer
        }
    }

    input_tokenized_data, true_init_config = attack_utility.generate_valid_input_tokenized_data(tokenizer, input_conv, target, initial_config, logger)
    logger.log(true_init_config)

    weighted_attention_hyperparams = {
        "signal_function": losses_experimental.attention_metricized_signal_v2,
        "signal_kwargs": {
            "prob_dist_metric": losses_experimental.pointwise_sum_of_differences_payload_only,
            "layer_weight_strategy": losses_experimental.cached_abs_grad_dolly_layer_weights,
            "ideal_attentions": losses_experimental.uniform_ideal_attentions,
            "ideal_attentions_kwargs": {
                "attention_mask_strategy": "payload_only"
            }
        },
        "true_loss_function": losses_experimental.attention_metricized_v2_true_loss,
        "true_loss_kwargs": {
            "prob_dist_metric": losses_experimental.pointwise_sum_of_differences_payload_only,
            "layer_weight_strategy": losses_experimental.cached_abs_grad_dolly_layer_weights,
            "ideal_attentions": losses_experimental.uniform_ideal_attentions,
            "ideal_attentions_kwargs": {
                "attention_mask_strategy": "payload_only"
            }
        },
        "max_steps": 350,
        "forward_eval_candidates": 512,
        "topk": 256,
        "substitution_validity_function": secalign.secalign_filter
    }
    weighted_attention_step = {
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": weighted_attention_hyperparams
    }

    gcg_hyperparams = {
        "max_steps": 150,
        "topk": 256,
        "forward_eval_candidates": 512,
        "substitution_validity_function": secalign.secalign_filter
    }
    gcg_step = {
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": gcg_hyperparams
    }

    attack_config = {
        "input_tokenized_data": input_tokenized_data,
        "attack_algorithm": "sequential",
        "attack_hyperparameters": [
            weighted_attention_step,
            gcg_step
        ],
        "early_stop": False,
        "eval_every_step": False,
        "to_cache_logits": True,
        "to_cache_attentions": True
    }

    logger.log(attack_config)
    loss_sequences_attack, best_output_sequences_attack = adversarial_opt.adversarial_opt(model, tokenizer, input_conv, target, attack_config, logger)
    logger.log(loss_sequences_attack)
    logger.log(best_output_sequences_attack)
    final_inputs_strings_attack = tokenizer.batch_decode(best_output_sequences_attack, clean_up_tokenization_spaces=False)
    logger.log(final_inputs_strings_attack)

    del loss_sequences_attack, best_output_sequences_attack, final_inputs_strings_attack
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    gcg_baseline_params = {
        "signal_function": gcg.og_gcg_signal,
        "max_steps": 500,
        "topk": 256,
        "forward_eval_candidates": 512,
        "substitution_validity_function": secalign.secalign_filter
    }
    adversarial_parameters_dict_baseline = {
        "input_tokenized_data": input_tokenized_data,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": gcg_baseline_params,
        "early_stop": False,
        "eval_every_step": False,
        "to_cache_logits": True,
        "to_cache_attentions": True,
    }

    logger.log(adversarial_parameters_dict_baseline)
    loss_sequences_baseline, best_output_sequences_baseline = adversarial_opt.adversarial_opt(model, tokenizer, input_conv, target, adversarial_parameters_dict_baseline, logger)
    logger.log(loss_sequences_baseline)
    logger.log(best_output_sequences_baseline)
    final_inputs_strings_baseline = tokenizer.batch_decode(best_output_sequences_baseline, clean_up_tokenization_spaces=False)
    logger.log(final_inputs_strings_baseline)

    del loss_sequences_baseline, best_output_sequences_baseline, final_inputs_strings_baseline
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    

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

    parser = argparse.ArgumentParser(description='Script with GPU device selection.')
    parser.add_argument(
        '--device', 
        type=int, 
        default=0, 
        help='GPU device index to use (default: 0)'
    )
    multiprocess_group = parser.add_mutually_exclusive_group(required=False)
    multiprocess_group.add_argument(
        "--multiprocess",
        dest="multiprocess",
        action="store_true",
        help="Use multiple processes (default)"
    )
    multiprocess_group.add_argument(
        "--no-multiprocess",
        dest="multiprocess",
        action="store_false",
        help="Do not use multiple processes"
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=50,
        help="Num examples to attack"
    )

    parser.set_defaults(multiprocess=True)

    args = parser.parse_args()


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

    print(f"num_examples={args.num_examples}")
    indices_to_sample = random.sample(list(range(len(alpacaeval_convs_raw))), args.num_examples)
    print(f"indices_to_sample={indices_to_sample}")
    alpacaeval_convs_raw = [alpacaeval_convs_raw[i] for i in indices_to_sample]

    if args.multiprocess:
        EXPT_FOLDER_PREFIX = "logs/llama_3_full_2_20"
        os.makedirs(EXPT_FOLDER_PREFIX, exist_ok=True)
        gpu_ids = list(range(torch.cuda.device_count()))
        NUM_EXPERIMENTS_ON_GPU = len(alpacaeval_convs_raw) // len(gpu_ids)
        alpacaeval_batched = [alpacaeval_convs_raw[(NUM_EXPERIMENTS_ON_GPU) * x: (NUM_EXPERIMENTS_ON_GPU)* (x + 1)] for x in gpu_ids]
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(len(gpu_ids)) as process_pool:
            final_results = process_pool.starmap(run_secalign_eval_on_single_gpu, [(EXPT_FOLDER_PREFIX, i, alpacaeval_batched[i]) for i in gpu_ids])
    else:
        EXPT_FOLDER_PREFIX = "logs/llama_3_full_2_20"
        final_results = run_secalign_eval_on_single_gpu(EXPT_FOLDER_PREFIX, args.device, alpacaeval_convs_raw)
