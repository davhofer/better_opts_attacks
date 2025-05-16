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
    logger: experiment_logger.ExperimentLogger,    
):
    if "input_tokenized_data" in adversarial_parameters_dict:
        input_tokenized_data = adversarial_parameters_dict["input_tokenized_data"]
    else:
        if "init_config" in adversarial_parameters_dict and input_template is not None:
            init_config = adversarial_parameters_dict["init_config"]
            num_init_tries = 0
            while num_init_tries < 100:
                try:
                    adv_prefix_init, adv_suffix_init = attack_utility.initialize_adversarial_strings(tokenizer, init_config)
                    if isinstance(input_template, str):
                        input_tokenized_data = attack_utility.string_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                    elif isinstance(input_template, list):
                        input_tokenized_data = attack_utility.conversation_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                except Exception as e:
                    INIT_TOKENIZATION_FAILED = f"The given initialization failed due to the following reasons - {str(e)}"
                    logger.log(INIT_TOKENIZATION_FAILED)
                    if init_config["strategy"] != "random":
                        raise ValueError(f"{INIT_TOKENIZATION_FAILED}")
                    new_seed = int(time.time())
                    RETRYING_STRING = f"Retrying with another random seed: {str(new_seed)}"
                    init_config["seed"] = new_seed
                    logger.log(RETRYING_STRING)
                else:
                    break
                num_init_tries += 1
            
            if num_init_tries >= 99:
                SEVERAL_INITS_FAILED = "ALL INITS FAILED. RAISING EXCEPTION."
                logger.log(SEVERAL_INITS_FAILED)
                raise ValueError(SEVERAL_INITS_FAILED)
        else:
            raise ValueError(f"At least give something to build off off")

    attack_algorithm = adversarial_parameters_dict["attack_algorithm"]
    if attack_algorithm == "gcg":
        loss_sequences, best_output_sequences = gcg.gcg(model, tokenizer, input_tokenized_data, adversarial_parameters_dict["attack_hyperparameters"], logger)
        logger.log(loss_sequences)
        logger.log(best_output_sequences)
        return loss_sequences, best_output_sequences, None
    elif attack_algorithm == "custom_gcg":
        early_stop = adversarial_parameters_dict.get("early_stop", True)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", False)
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)
        to_cache_logits = adversarial_parameters_dict.get("to_cache_logits", True)
        to_cache_attentions = adversarial_parameters_dict.get("to_cache_attentions", True)

        logprobs_sequences, best_output_sequences = gcg.custom_gcg(model,
            tokenizer,
            input_tokenized_data,
            adversarial_parameters_dict["attack_hyperparameters"],
            logger,
            early_stop=early_stop,
            eval_every_step=eval_every_step,
            identical_outputs_before_stop=identical_outputs_before_stop,
            generation_config=generation_config,
            eval_initial=eval_initial,
            to_cache_logits=to_cache_logits,
            to_cache_attentions=to_cache_attentions
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

    elif attack_algorithm == "sequential":

        attack_steps = adversarial_parameters_dict["attack_hyperparameters"]
        assert isinstance(attack_steps, list)
        assert all([isinstance(x, dict) for x in attack_steps])


        early_stop = adversarial_parameters_dict.get("early_stop", False)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", False)
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)
        best_choice_function = adversarial_parameters_dict.get("best_choice_function", attack_utility.default_best_choice_function)

        all_logprobs_sequences, all_best_tokens_sequences = [], []
        for attack_block, attack_config in enumerate(attack_steps):
            
            if attack_block != 0:
                eval_initial = False

            logger.log(attack_block)
            logger.log(attack_config)

            attack_config.update({
                "input_tokenized_data": input_tokenized_data,
                "early_stop": early_stop,
                "eval_every_step": eval_every_step,
                "identical_outputs_before_stop": identical_outputs_before_stop,
                "generation_config": generation_config,
                "eval_initial": eval_initial,
            })

            logprobs_sequences, best_tokens_sequences = adversarial_opt(model,
                tokenizer,
                None,
                target_output_str,
                attack_config,
                logger,
            )
            

            logger.log(logprobs_sequences)
            logger.log(best_tokens_sequences)

            all_logprobs_sequences.extend(logprobs_sequences)
            all_best_tokens_sequences.extend(best_tokens_sequences)

            input_tokenized_data = best_choice_function(model, tokenizer, input_tokenized_data, all_best_tokens_sequences, logger, logprobs_sequences=all_logprobs_sequences)

        return all_logprobs_sequences, all_best_tokens_sequences