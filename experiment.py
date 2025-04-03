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

import utils.experiment_logger as experiment_logger
import algorithms.gcg as gcg
from secalign_refactored import secalign
import adversarial_opt


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
        "prefix_length": 0,
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
    malicious_inst = "<ADV_PREFIX>" + adv_bench_example["goal"] + "<ADV_SUFFIX>"
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


SECALIGN_TARGET_STRING = "Hacked!"
def attack_secalign_model(
    
    model,
    tokenizer,
    frontend_delimiters,
    logger
):
    target_string


if __name__ == "__main__":

    EXPT_FOLDER = "logs/runs30"
    shutil.copy(__file__, EXPT_FOLDER)

    for (model_name, defence) in secalign.MODEL_REL_PATHS:
        if "instruct" not in model_name.lower():
            continue
        if "defence" == "undefended":
            continue
        try:
            model, tokenizer, frontend_delimiters = secalign.load_defended_model(model_name, defence)
            attack_secalign_model()
        except Exception:
            traceback.print_exc()




    # MODEL_PATH = "/data/models/hf/Meta-Llama-3-8B-Instruct"
    # model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    # tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    # with open(f"data/purplellama_indirect.json", "r") as purplellama_indirect_file:
    #     purplellama_data = json.load(purplellama_indirect_file)

    # ADVBENCH_PATH = "data/advbench_harmful_behaviors.csv"    
    # EXPT_FOLDER = "logs/runs24"
    # if not os.path.exists(EXPT_FOLDER):
    #     os.mkdir(EXPT_FOLDER)
    # shutil.copy(__file__, EXPT_FOLDER)
    
    # for i in range(3, 5):
    #     for rand_restart in range(3):
    #         expt_id = f"run_{str(datetime.datetime.now()).replace("-","").replace(" ","").replace(":","").replace(".","")}"
    #         logger = experiment_logger.ExperimentLogger(f"{EXPT_FOLDER}/{expt_id}")
    #         logger.log(model.__repr__(), example_num=i, rand_restart=rand_restart)
    #         attack_advbench(ADVBENCH_PATH, i, model, tokenizer, logger=logger)
