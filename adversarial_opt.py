import transformers
import torch
import typing
import datetime
import json
import time
import bitsandbytes
import itertools
import pandas as pd

import algorithms.gcg as gcg
import algorithms.autodan as autodan
import algorithms.embed_snap as embed_snap
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
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)

        logprobs_sequences, best_output_sequences = gcg.custom_gcg(model,
            tokenizer,
            input_tokenized_data,
            adversarial_parameters_dict["attack_hyperparameters"],
            logger,
            early_stop=early_stop,
            eval_every_step=eval_every_step,
            identical_outputs_before_stop=identical_outputs_before_stop,
            generation_config=generation_config,
            eval_initial=eval_initial
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
    elif attack_algorithm == "embed_opt":
        early_stop = adversarial_parameters_dict.get("early_stop", False)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", True)
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)

        logprobs_sequences, embeds_sequences = embed_snap.embed_opt(model,
            tokenizer,
            input_tokenized_data,
            adversarial_parameters_dict["attack_hyperparameters"],
            logger,
            early_stop=early_stop,
            eval_every_step=eval_every_step,
            eval_initial=eval_initial,
            generation_config=generation_config,
            identical_outputs_before_stop=identical_outputs_before_stop
        )
        logger.log(logprobs_sequences)
        logger.log(embeds_sequences)
        return logprobs_sequences, embeds_sequences