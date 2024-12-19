import transformers
import torch
import typing
import random
import gc
import experiment_logger

ADV_SUFFIX_INDICATOR = "<ADV_SUFFIX>"
ADV_PREFIX_INDICATOR = "<ADV_PREFIX>"
def string_masks(
    tokenizer: transformers.AutoTokenizer,
    input_string_template: str,
    adv_pre_init: str,
    adv_suf_init: str,
    target_string: str
):
    pre_prefix_string = input_string_template.split(ADV_PREFIX_INDICATOR)[0]
    suf_suffix_string = input_string_template.split(ADV_SUFFIX_INDICATOR)[-1]
    
    payload_string_left = input_string_template.split(ADV_PREFIX_INDICATOR)[-1].split(ADV_SUFFIX_INDICATOR)[0]
    payload_string_right = input_string_template.split(ADV_SUFFIX_INDICATOR)[0].split(ADV_PREFIX_INDICATOR)[-1]
    try:
        assert payload_string_left == payload_string_right
        payload_string = payload_string_left
    except AssertionError:
        raise AssertionError("Payload string mismatch when extracted.")

    pre_prefix_tokens = tokenizer(pre_prefix_string, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    suf_suffix_tokens = tokenizer(suf_suffix_string, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    payload_tokens = tokenizer(payload_string, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    adv_pre_init_tokens = tokenizer(adv_pre_init, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    adv_suf_init_tokens = tokenizer(adv_suf_init, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    
    target_tokens = tokenizer(target_string, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    pre_payload_tokens = torch.cat((pre_prefix_tokens, adv_pre_init_tokens)).to(pre_prefix_tokens.dtype)
    retokenized_pre_payload_tokens = tokenizer(tokenizer.decode(pre_payload_tokens), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(retokenized_pre_payload_tokens) == len(pre_payload_tokens):
        optimizable_prefix_start_index = len(pre_prefix_tokens)
    elif len(retokenized_pre_payload_tokens) == len(pre_payload_tokens) - 1:
        optimizable_prefix_start_index = len(pre_prefix_tokens) + 1
    else:
        raise ValueError("Retokenization of break point between pre-prefix and pre-opt-string leads to more than 2 error")

    prefix_plus_payload_tokens = torch.cat((retokenized_pre_payload_tokens, payload_tokens)).to(retokenized_pre_payload_tokens.dtype)
    retokenized_prefix_plus_payload_tokens = tokenizer(tokenizer.decode(prefix_plus_payload_tokens), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(retokenized_prefix_plus_payload_tokens) == len(prefix_plus_payload_tokens):
        optimizable_prefix_end_index = len(retokenized_pre_payload_tokens)
    elif len(retokenized_prefix_plus_payload_tokens) == len(prefix_plus_payload_tokens) - 1:
        optimizable_prefix_end_index = len(retokenized_pre_payload_tokens) - 1
    else:
        raise ValueError("Retokenization of break point between pre-opt-string and payload leads to more than 2 error")
    payload_start_index = optimizable_prefix_end_index + 1

    upto_adv_suf_tokens = torch.cat((retokenized_prefix_plus_payload_tokens, adv_suf_init_tokens)).to(retokenized_prefix_plus_payload_tokens.dtype)
    retokenized_upto_adv_suf_tokens = tokenizer(tokenizer.decode(upto_adv_suf_tokens), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(retokenized_upto_adv_suf_tokens) == len(upto_adv_suf_tokens):
        optimizable_suffix_start_index = len(prefix_plus_payload_tokens)
    elif len(retokenized_upto_adv_suf_tokens) == len(upto_adv_suf_tokens) - 1:
        optimizable_suffix_start_index = len(prefix_plus_payload_tokens) + 1
    else:
        raise ValueError("Retokenization of break point between payload and suf-opt-string leads to more than 2 error")
    payload_end_index = optimizable_suffix_start_index - 1

    upto_suf_suffix_tokens = torch.cat((retokenized_upto_adv_suf_tokens, suf_suffix_tokens)).to(retokenized_upto_adv_suf_tokens.dtype)
    retokenized_upto_suf_suffix_tokens = tokenizer(tokenizer.decode(upto_suf_suffix_tokens), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(retokenized_upto_suf_suffix_tokens) == len(upto_suf_suffix_tokens):
        optimizable_suffix_end_index = len(retokenized_upto_adv_suf_tokens)
    elif len(retokenized_upto_suf_suffix_tokens) == len(upto_suf_suffix_tokens) - 1:
        optimizable_suffix_end_index = len(retokenized_upto_adv_suf_tokens) - 1
    else:
        raise ValueError("Retokenization of break point between suf-opt-string and suf-suffix leads to more than 2 error")

    eval_input_mask = torch.arange(0, len(retokenized_upto_suf_suffix_tokens))

    target_string_included_tokens = torch.cat((retokenized_upto_suf_suffix_tokens, target_tokens)).to(retokenized_upto_suf_suffix_tokens.dtype)
    retokenized_including_target = tokenizer(tokenizer.decode(target_string_included_tokens), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(retokenized_including_target) == len(target_string_included_tokens):
        target_string_start_index = len(retokenized_upto_suf_suffix_tokens)
        target_string_end_index = len(target_string_included_tokens)
    else:
        raise ValueError("Retokenization of string with target is not happening cleanly. Change something please.")

    final_tokens = target_string_included_tokens
    try:
        prefix_mask = torch.arange(optimizable_prefix_start_index, optimizable_prefix_end_index)
    except RuntimeError:
        prefix_mask = torch.tensor([]).to(final_tokens.dtype)
    
    try:
        suffix_mask = torch.arange(optimizable_suffix_start_index, optimizable_suffix_end_index)
    except RuntimeError:
        suffix_mask = torch.tensor([]).to(final_tokens.dtype)
    
    final_opt_mask = torch.cat((prefix_mask, suffix_mask)).to(prefix_mask.dtype)
    target_string_mask = torch.arange(target_string_start_index, target_string_end_index)
    payload_mask = torch.arange(payload_start_index, payload_end_index + 1)

    return {
        "tokens": final_tokens,
        "masks": {
            "optim_mask": final_opt_mask,
            "prefix_mask": prefix_mask,
            "suffix_mask": suffix_mask,
            "target_mask": target_string_mask,
            "input_mask": eval_input_mask,
            "payload_mask": payload_mask
        }
    }

def DEFAULT_FILTER_FUNCTION(tokens: torch.tensor, **kwargs):
    return True

INITIAL_PREFIX_LENGTH = 40
INITIAL_SUFFIX_LENGTH = 40
DEFAULT_INIT_TOKEN = " And"
def initialize_adversarial_strings(tokenizer: transformers.AutoTokenizer, init_config: typing.Dict):

    adv_prefix_init: str
    adv_suffix_init: str

    try:
        init_strategy_type = init_config["strategy_type"]
    except KeyError:
        raise ValueError("strategy_type needs to be in initialization strategy.")

    if init_strategy_type == "random":
        try:
            random_seed = init_config["seed"]
            random.seed(random_seed)
        except KeyError:
            pass

        try:
            prefix_filter: typing.Callable[..., bool] = init_config["prefix_filter"]
        except KeyError:
            prefix_filter = DEFAULT_FILTER_FUNCTION
        
        try:
            suffix_filter: typing.Callable[..., bool] = init_config["suffix_filter"]
        except KeyError:
            suffix_filter = DEFAULT_FILTER_FUNCTION
        
        try:
            filter_metadata = init_config["filter_metadata"]
        except KeyError:
            filter_metadata = None

        try:
            prefix_length = init_config["prefix_length"]
        except KeyError:
            prefix_length = INITIAL_PREFIX_LENGTH
        
        try:
            suffix_length = init_config["suffix_length"]
        except KeyError:
            suffix_length = INITIAL_SUFFIX_LENGTH

        while True:
            prefix_random_tokens = []
            for _ in range(prefix_length):
                rand_token = random.randint(0, tokenizer.vocab_size)
                prefix_random_tokens.append(rand_token)
            prefix_random_tokens = torch.tensor(prefix_random_tokens)
            if prefix_filter(prefix_random_tokens, filter_metadata=filter_metadata):
                break
        
        while True:
            suffix_random_tokens = []
            for _ in range(suffix_length):
                rand_token = random.randint(0, tokenizer.vocab_size)
                suffix_random_tokens.append(rand_token)
            suffix_random_tokens = torch.tensor(suffix_random_tokens)
            if suffix_filter(suffix_random_tokens, filter_metadata=filter_metadata):
                break
        
        adv_prefix_init = tokenizer.decode(prefix_random_tokens)
        adv_suffix_init = tokenizer.decode(suffix_random_tokens)

    elif init_strategy_type == "fixed_string":
        try:
            adv_prefix_init = init_config["adv_prefix_init"]
            adv_suffix_init = init_config["adv_suffix_init"]
        except KeyError:
            raise ValueError("Both adv_prefix_init and adv_suffix_init need to be if your initialization strategy is fixed_string")

    elif init_strategy_type == "fixed_length_const_init":
        try:
            prefix_length = init_config["prefix_length"]
        except KeyError:
            prefix_length = INITIAL_PREFIX_LENGTH
        
        try:
            prefix_token = init_config["prefix_token"]
        except KeyError:
            prefix_token = DEFAULT_INIT_TOKEN

        try:
            suffix_length = init_config["suffix_length"]
        except KeyError:
            suffix_length = INITIAL_SUFFIX_LENGTH
        
        try:
            suffix_token = init_config["suffix_token"]
        except KeyError:
            suffix_token = DEFAULT_INIT_TOKEN
        
        adv_prefix_init = prefix_token * prefix_length
        adv_suffix_init = suffix_token * suffix_length
    
    else:
        raise ValueError("Initialization Strategy not recognized")
    
    return adv_prefix_init, adv_suffix_init

BULK_FORWARD_DEFAULT_BSZ = 64
DEFAULT_GENERATION_PARAMS = {
    "logprobs": True,
}
def bulk_logits(
    model: transformers.AutoModelForCausalLM,
    data: torch.tensor,
    batch_size=BULK_FORWARD_DEFAULT_BSZ,
    generation_params=DEFAULT_GENERATION_PARAMS
):
    with torch.no_grad():
        data_split = torch.split(data, BULK_FORWARD_DEFAULT_BSZ)
        list_of_results = []
        for data_piece in data_split:
            try:
                data_piece_result = model.forward(input_ids=data_piece).logits
            except torch.cuda.OutOfMemoryError:
                data_piece_result = bulk_logits(model, data_piece, batch_size // 2, generation_params)
            list_of_results.append(data_piece)
            del data_piece_result
            gc.collect()
            torch.cuda.empty_cache()
    return torch.cat(list_of_results, dim=0)

def bulk_logits_iter(
    model: transformers.AutoModelForCausalLM,
    data: torch.tensor,
    batch_size=BULK_FORWARD_DEFAULT_BSZ,
    generation_params=DEFAULT_GENERATION_PARAMS
):
    """
    Iterator version of bulk_logits that yields results one batch at a time
    to reduce memory usage.
    """
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_piece = data[i:i + batch_size]
            try:
                logits = model.forward(input_ids=data_piece).logits
                cpu_logits = logits.cpu()
                del logits
                gc.collect()
                torch.cuda.empty_cache()
                yield cpu_logits
            except torch.cuda.OutOfMemoryError:
                # If OOM occurs, recursively process with smaller batch size
                sub_iterator = bulk_logits_iter(
                    model, 
                    data_piece, 
                    batch_size // 2, 
                    generation_params
                )
                for sub_result in sub_iterator:
                    yield sub_result
            
            # Clean up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()

            torch.cuda.synchronize()

def bulk_logits_from_embeds(
    model: transformers.AutoModelForCausalLM,
    data: torch.tensor,
    batch_size=BULK_FORWARD_DEFAULT_BSZ,
    generation_params=DEFAULT_GENERATION_PARAMS
):
    with torch.no_grad():
        data_split = torch.split(data, BULK_FORWARD_DEFAULT_BSZ)
        list_of_results = []
        for data_piece in data_split:
            try:
                data_piece_result = model.forward(inputs_embeds=data_piece).logits
            except torch.cuda.OutOfMemoryError:
                data_piece_result = bulk_logits_from_embeds(model, data_piece, batch_size // 2, generation_params)
            list_of_results.append(data_piece_result.detach())
            del data_piece_result
            gc.collect()
            torch.cuda.empty_cache()
    return torch.cat(list_of_results, dim=0)

def bulk_forward_iter(
    model: transformers.AutoModelForCausalLM,
    data: torch.tensor,
    batch_size=BULK_FORWARD_DEFAULT_BSZ,
    generation_params=DEFAULT_GENERATION_PARAMS
) -> typing.Iterator[typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, ...]]]:
    """
    Iterator that yields both logits and attentions one batch at a time.
    
    Returns:
        Iterator yielding tuples of (logits, attentions) where:
        - logits: tensor of shape (batch_size, sequence_length, vocab_size)
        - attentions: tuple of attention tensors for each layer
    """
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_piece = data[i:i + batch_size]
            try:
                output = model.forward(
                    input_ids=data_piece,
                    output_attentions=True
                )
                yield output.logits, output.attentions
                
            except torch.cuda.OutOfMemoryError:
                # If OOM occurs, recursively process with smaller batch size
                sub_iterator = bulk_forward_iter(
                    model, 
                    data_piece, 
                    batch_size // 2, 
                    generation_params
                )
                for sub_logits, sub_attentions in sub_iterator:
                    yield sub_logits, sub_attentions
            
            # Clean up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()

UNREDUCED_CE_LOSS = torch.nn.CrossEntropyLoss(reduction="none")
def target_logprobs(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    target_tokens: torch.tensor,
    logger: experiment_logger.ExperimentLogger,
):
    target_mask = masks_data["target_mask"]
    losses_list = []
    for logit_piece in bulk_logits_iter(model, input_points):
        loss_tensor = UNREDUCED_CE_LOSS(torch.transpose(logit_piece[:, target_mask - 1, :], 1, 2), target_tokens.repeat((logit_piece.shape[0], 1)).to(logit_piece.device)).sum(dim=1)
        losses_list.append(loss_tensor)
    loss_tensor = torch.cat(losses_list)
    return loss_tensor
