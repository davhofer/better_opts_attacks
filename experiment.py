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


def secalign_filter(token_ids, **kwargs):

    masks_data = kwargs.get("masks_data", None)
    tokenizer = kwargs.get("tokenizer", None)

    if tokenizer is None:
        raise ValueError(f"SecAlign filter function needs a tokenizer to be sent through")

    is_invertible = attack_utility.invertibility_filter(token_ids, tokenizer=tokenizer)

    if masks_data is None:
        decoded_string = tokenizer.decode(token_ids)
        return not any([spec_token_id in decoded_string for spec_token_id in tokenizer.get_added_vocab()])
    prefix_mask = masks_data["prefix_mask"]
    suffix_mask = masks_data["suffix_mask"]
    decoded_prefix = tokenizer.decode(token_ids[prefix_mask])
    decoded_suffix = tokenizer.decode(token_ids[suffix_mask])
    prefix_contains_specs = any([spec_token_id in decoded_prefix for spec_token_id in tokenizer.get_added_vocab()])
    suffix_contains_specs = any([spec_token_id in decoded_suffix for spec_token_id in tokenizer.get_added_vocab()])
    
    
    return (not (prefix_contains_specs or suffix_contains_specs)) and is_invertible

def secalign_ideal_attention_v1(
    model,
    tokenizer,
    input_points,
    masks_data,
    *,
    attention_mask_strategy,
    **kwargs
):
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    
    payload_mask: torch.Tensor = masks_data["payload_mask"]
    control_mask: torch.Tensor = masks_data["control_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]

    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
        attention_mask_formatted = torch.zeros(input_points.shape)
        attention_mask_formatted[:, attention_mask] = 1
        attentions = model(input_ids=input_points.to(model.device), attention_mask=attention_mask_formatted.to(model.device), output_attentions=True).attentions
    elif attention_mask_strategy == "payload_and_control":
        attention_mask = torch.cat((payload_mask, control_mask))
        attention_mask_formatted = torch.zeros(input_points.shape)
        attention_mask_formatted[:, attention_mask] = 1
        attentions = model(input_ids=input_points.to(model.device), attention_mask=attention_mask_formatted.to(model.device), output_attentions=True).attentions
    # Removing these two, because it might be that we can emulate the squeezed
    # versions with just modifying position_ids???
    # 
    # elif attention_mask_strategy == "payload_control_squeezed":
    #     attention_mask = torch.cat((payload_mask, control_mask))
    #     attentions = model(input_ids=input_points[:, sorted(attention_mask)], output_attentions=True).attentions
    # elif attention_mask_strategy == "payload_squeezed":
    #     attention_mask = payload_mask
    #     attentions = model(input_ids=input_points[:, sorted(attention_mask)], output_attentions=True).attentions
    else:
        raise ValueError(f"attention_mask_strategy {attention_mask_strategy} is not implemented yet.")
    return torch.stack(attentions)[:, :, :, target_mask - 1, :]

def uniform_ideal_attentions(
    model,
    tokenizer,
    input_points,
    masks_data,
    *,
    attention_mask_strategy,
):
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    payload_mask: torch.Tensor = masks_data["payload_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]
    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
    elif attention_mask_strategy == "payload_and_control":
        control_mask: torch.Tensor = masks_data["control_mask"]    
        attention_mask = torch.cat((payload_mask, control_mask))
    else:
        raise ValueError(f"attention_mask_strategy {attention_mask_strategy} is not implemented yet.")
    dummy_attentions = torch.stack(model(input_ids=torch.unsqueeze(input_points[0], dim=0).to(model.device), output_attentions=True).attentions)
    ideal_shape = dummy_attentions.shape
    ideal_shape = (ideal_shape[0], input_points.shape[0], ideal_shape[2], ideal_shape[3], ideal_shape[4])
    attentions = torch.zeros(ideal_shape)
    attentions[:, :, :, :, attention_mask] = 1 / len(attention_mask)
    return attentions[:, :, :, -(len(target_mask) + 1):-1, :]


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
        assert isinstance(input_conv, list) and all([isinstance(conv_part, dict) for conv_part in input_conv])
        inst_str = copy.deepcopy(input_conv[0]["content"])
        data_str = copy.deepcopy(input_conv[1]["content"])
        if data_str[-1] != '.' and data_str[-1] != '!' and data_str[-1] != '?': data_str += '.'
        data_str += ' '     
        data_str += attack_utility.ADV_PREFIX_INDICATOR + secalign.SECALIGN_COMMON_INSTRUCTION + " " + attack_utility.ADV_SUFFIX_INDICATOR
        static_string = prompt_template.format_map({"instruction": inst_str, "input": data_str})
        input_conv = tokenizer.batch_decode(tokenizer([static_string])["input_ids"], clean_up_tokenization_spaces=False)[0]
    else:
        input_conv = [
            {
                "role": input_conv[0]["role"],
                "content": input_conv[0]["content"]
            },
            {
                "role": input_conv[1]["role"],
                "content": input_conv[1]["content"] + " " + attack_utility.ADV_PREFIX_INDICATOR + " " +  secalign.SECALIGN_COMMON_INSTRUCTION  + attack_utility.ADV_SUFFIX_INDICATOR
            }
        ]

    initial_config = {
        "strategy_type": "fixed_string",
        "adv_prefix_init": "",
        "adv_suffix_init": " ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        "prefix_filter": secalign_filter,
        "suffix_filter": secalign_filter,
        "filter_metadata": {
            "tokenizer": tokenizer
        }
    }

    input_tokenized_data, true_init_config = attack_utility.generate_valid_input_tokenized_data(tokenizer, input_conv, target, initial_config, logger)
    logger.log(true_init_config)

    custom_gcg_hyperparameters_1 = {
        "signal_function": gcg.og_gcg_signal,
        "true_loss_function": attack_utility.target_logprobs,
        "max_steps": 500,
        "topk": 256,
        "forward_eval_candidates": 512,
        "substitution_validity_function": secalign_filter
    }
    adversarial_parameters_dict_1 = {
        "input_tokenized_data": input_tokenized_data,
        "attack_algorithm": "custom_gcg",
        "attack_hyperparameters": custom_gcg_hyperparameters_1,
        "early_stop": False,
        "eval_every_step": False
    }

    logger.log(adversarial_parameters_dict_1)
    loss_sequences_control, best_output_sequences_control = adversarial_opt.adversarial_opt(model, tokenizer, input_conv, target, adversarial_parameters_dict_1, logger)
    logger.log(loss_sequences_control)
    logger.log(best_output_sequences_control)

    final_inputs_strings_control = tokenizer.batch_decode(best_output_sequences_control, clean_up_tokenization_spaces=False)
    final_inputs_dicts_control = [
        {
            "instruction": x.split(f"\n\n{data_delm}\n")[0].split(f"{inst_delm}\n")[-1],
            "input": x.split(f"\n\n{data_delm}\n")[-1].split(f"\n\n{resp_delm}\n")[0]
        }
        for x in final_inputs_strings_control
    ]
    final_outputs_control = secalign.test_model_output(
        [
            prompt_template.format_map(x)
            for x in final_inputs_dicts_control
        ],
        model,
        tokenizer
    )
    logger.log(final_outputs_control)

    # adversarial_parameters_dict_2 = {
    #     "input_tokenized_data": input_tokenized_data,
    #     "attack_algorithm": "sequential",
    #     "attack_hyperparameters": [
    #         {
    #             "attack_algorithm": "custom_gcg",
    #             "attack_hyperparameters": {
    #                 "signal_function": losses_experimental.attention_metricized_signal_v2,
    #                 "signal_kwargs": {
    #                     "prob_dist_metric": losses_experimental.pointwise_sum_of_differences_payload_only,
    #                     "layer_weight_strategy": "uniform",
    #                     "attention_mask_strategy": "payload_only",
    #                     "ideal_attentions": uniform_ideal_attentions,
    #                     "ideal_attentions_kwargs": {
    #                         "attention_mask_strategy": "payload_only"
    #                     }
    #                 },
    #                 "true_loss_function": losses_experimental.attention_metricized_v2_true_loss,
    #                 "true_loss_kwargs": {
    #                     "prob_dist_metric": losses_experimental.pointwise_sum_of_differences_payload_only,
    #                     "layer_weight_strategy": "uniform",
    #                     "attention_mask_strategy": "payload_only",
    #                     "ideal_attentions": uniform_ideal_attentions,
    #                     "ideal_attentions_kwargs": {
    #                         "attention_mask_strategy": "payload_only"
    #                     }
    #                 },
    #                 "max_steps": 350,
    #                 "topk": 256,
    #                 "forward_eval_candidates": 512,
    #                 "substitution_validity_function": secalign_filter,
    #             }
    #         },
    #         {
    #             "attack_algorithm": "custom_gcg",
    #             "attack_hyperparameters": {
    #                 "signal_function": gcg.og_gcg_signal,
    #                 "true_loss_function": attack_utility.target_logprobs,
    #                 "max_steps": 150,
    #                 "topk": 256,
    #                 "forward_eval_candidates": 512,
    #                 "substitution_validity_function": secalign_filter
    #             }
    #         },
    #     ],
    #     "early_stop": False,
    #     "eval_every_step": False   
    # }
    # logger.log(adversarial_parameters_dict_2)
    # loss_sequences_real, best_output_sequences_real = adversarial_opt.adversarial_opt(model, tokenizer, input_conv, target, adversarial_parameters_dict_2, logger)
    # logger.log(loss_sequences_real)
    # logger.log(best_output_sequences_real)

    # final_inputs_strings_real = tokenizer.batch_decode(best_output_sequences_real, clean_up_tokenization_spaces=False)
    # final_inputs_dicts_real = [
    #     {
    #         "instruction": x.split(f"\n\n{data_delm}\n")[0].split(f"{inst_delm}\n")[-1],
    #         "input": x.split(f"\n\n{data_delm}\n")[-1].split(f"\n\n{resp_delm}\n")[0]
    #     }
    #     for x in final_inputs_strings_real
    # ]
    # final_outputs_real = secalign.test_model_output(
    #     [
    #         prompt_template.format_map(x)
    #         for x in final_inputs_dicts_real
    #     ],
    #     model,
    #     tokenizer
    # )
    # logger.log(final_outputs_real)


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
        attack_secalign_model(example_target, model, tokenizer, frontend_delimiters, logger, convert_to_secalign_format=False)
        gc.collect()
        torch.cuda.empty_cache()

def run_secalign_eval_auto(expt_folder_prefix, example_targets, **kwargs):
    expt_folder = f"{expt_folder_prefix}"
    if not os.path.exists(expt_folder):
        os.mkdir(expt_folder)
    shutil.copy(__file__, expt_folder)
    model_name = "meta-llama-instruct"
    defence = "secalign"
    try:
        model, tokenizer, frontend_delimiters = secalign.maybe_load_secalign_defended_model(model_name, defence, device_map=f"cuda:1")
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
        attack_secalign_model(example_target, model, tokenizer, frontend_delimiters, logger, convert_to_secalign_format=True)
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
    ][104:]

    if args.multiprocess:
        EXPT_FOLDER_PREFIX = "logs/serious_attack_5"
        os.makedirs(EXPT_FOLDER_PREFIX, exist_ok=True)
        gpu_ids = list(range(torch.cuda.device_count()))
        NUM_EXPERIMENTS_ON_GPU = len(alpacaeval_convs_raw) // len(gpu_ids)
        alpacaeval_batched = [alpacaeval_convs_raw[(NUM_EXPERIMENTS_ON_GPU) * x: (NUM_EXPERIMENTS_ON_GPU)* (x + 1)] for x in gpu_ids]
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(len(gpu_ids)) as process_pool:
            final_results = process_pool.starmap(run_secalign_eval_on_single_gpu, [(EXPT_FOLDER_PREFIX, i, alpacaeval_batched[i]) for i in gpu_ids])
    else:
        EXPT_FOLDER_PREFIX = "logs/gcg_authoritative"
        final_results = run_secalign_eval_on_single_gpu(EXPT_FOLDER_PREFIX, args.device, alpacaeval_convs_raw)
