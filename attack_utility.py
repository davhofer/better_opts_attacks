import transformers
import torch
import typing
import random
import gc
import experiment_logger


def analyze_conversation_tokens(conversation, tokenizer):
    """
    Analyzes tokenization of a conversation using the tokenizer's built-in chat template,
    separating content tokens from control tokens. Includes generation prompt in the analysis.
    
    Args:
        conversation: List of dictionaries with 'role' and 'content' keys
        tokenizer: HuggingFace tokenizer instance with chat_template
    
    Returns:
        dict: Contains lists of content token indices and control token indices,
              along with the full token list and their string representations
    """
    # Get the formatted text using the tokenizer's chat template
    formatted_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Track the original content positions
    content_char_ranges = []
    for msg in conversation:
        # Find each content occurrence in the formatted text
        content = msg["content"]
        content_start = formatted_text.find(content)
        while content_start != -1:
            content_char_ranges.append(
                (content_start, content_start + len(content))
            )
            # Look for any additional occurrences
            content_start = formatted_text.find(
                content,
                content_start + 1
            )
    
    # Tokenize the full text
    tokens = tokenizer(formatted_text, return_offsets_mapping=True)
    
    # Separate content tokens from control tokens
    content_token_indices = []
    control_token_indices = []
    generation_prompt_indices = []
    
    # Get the offset mapping
    offset_mapping = tokens["offset_mapping"]
    
    # Find generation prompt location
    generation_start = len(formatted_text)
    # Any tokens that start after or at the generation prompt position
    # should be considered part of the generation prompt
    
    # Analyze each token
    for i, (start, end) in enumerate(offset_mapping):
        # Special tokens have (0,0) offset
        if start == end == 0:
            control_token_indices.append(i)
            continue
            
        # Check if token is part of the generation prompt
        if start >= generation_start:
            generation_prompt_indices.append(i)
            continue
            
        # Check if token falls within any content range
        is_content = False
        for content_start, content_end in content_char_ranges:
            # Token is content if it overlaps with content range
            if not (end <= content_start or start >= content_end):
                content_token_indices.append(i)
                is_content = True
                break
        
        if not is_content:
            control_token_indices.append(i)
    
    # Get the actual tokens for reference
    
    return {
        "content_token_indices": content_token_indices,
        "control_token_indices": control_token_indices,
        "generation_prompt_indices": generation_prompt_indices,
        "formatted_text": formatted_text
    }

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

def find_clean_token_span(tokenizer: transformers.PreTrainedTokenizer, 
                         full_text: str,
                         target_text: str,
                         full_tokens: typing.List[int]) -> None | typing.Dict[str, typing.Any]:
    """
    Find the largest contiguous sequence of tokens that cleanly maps to a substring of target_text.
    """
    # Get the token ids and offsets for the full text
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    char_spans = encoding.offset_mapping
    
    # Find the character positions of target_text in full_text
    start_pos = full_text.find(target_text)
    if start_pos == -1:
        return None
    end_pos = start_pos + len(target_text)
    
    # Find token spans that fall within these character positions
    token_start = None
    token_end = None
    
    for i, (start, end) in enumerate(char_spans):
        # Token completely within target_text
        if start >= start_pos and end <= end_pos:
            if token_start is None:
                token_start = i
            token_end = i + 1
    
    if token_start is None:
        return None
        
    # Verify the decoded tokens match a substring of target_text
    span_tokens = full_tokens[token_start:token_end]
    decoded = tokenizer.decode(span_tokens)
    
    assert decoded in target_text, "Decoded string not a subset of target_text"
    return {
        "start": token_start,
        "end": token_end,
        "text": decoded
    }

def conversation_masks(
    tokenizer: transformers.PreTrainedTokenizer,
    conversation: typing.List[typing.Dict[str, str]],
    adv_prefix_init: str,
    adv_suffix_init: str,
    target_string: str,
    prefix_placeholder: str = "<ADV_PREFIX>",
    suffix_placeholder: str = "<ADV_SUFFIX>"
) -> typing.Dict[str, typing.Any]:
    """
    Create masks for different parts of a tokenized conversation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        conversation: List of dictionaries with 'role' and 'content' keys
        adv_prefix_init: String to replace prefix_placeholder
        adv_suffix_init: String to replace suffix_placeholder
        target_string: Target string to be appended after the conversation
        prefix_placeholder: Placeholder for prefix in the content
        suffix_placeholder: Placeholder for suffix in the content
    
    Returns:
        Dictionary containing tokens and various masks
    """
    # First, process the conversation by replacing placeholders
    processed_conversation = []
    for turn in conversation:
        content = turn['content']
        
        # Replace placeholders in content
        if prefix_placeholder in content and suffix_placeholder in content:
            prefix_pos = content.find(prefix_placeholder)
            suffix_pos = content.find(suffix_placeholder)
            
            content = (
                content[:prefix_pos] + 
                adv_prefix_init + 
                content[prefix_pos + len(prefix_placeholder):suffix_pos] + 
                adv_suffix_init + 
                content[suffix_pos + len(suffix_placeholder):]
            )
        
        processed_conversation.append({
            "role": turn['role'],
            "content": content
        })
    
    full_text = tokenizer.apply_chat_template(
        processed_conversation,  # Exclude the target string message
        tokenize=False,
        add_generation_prompt=True,    # Let the tokenizer handle the generation prompt
    ) + target_string  # Add target string separately to maintain control over its position
    
    # Tokenize the conversation without the target string first
    conversation_tokens = tokenizer(
        tokenizer.apply_chat_template(
            processed_conversation,
            tokenize=False,
            add_generation_prompt=True
        ),
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    
    # Tokenize the target string separately
    target_tokens = tokenizer(
        target_string,
        return_offsets_mapping=True,
        add_special_tokens=False  # No special tokens for target as they're already in the conversation
    )
    
    # Combine tokens
    final_tokens = conversation_tokens['input_ids'] + target_tokens['input_ids']
    final_tokens = torch.tensor(final_tokens)

    # Combine offset mappings, adjusting target offsets
    last_offset = len(full_text) - len(target_string)
    target_offsets = [(start + last_offset, end + last_offset) 
                     for start, end in target_tokens.offset_mapping]
    char_spans = conversation_tokens.offset_mapping + target_offsets
    
    # Initialize masks
    seq_length = len(final_tokens)
    prefix_mask = torch.zeros(seq_length, dtype=torch.bool)
    suffix_mask = torch.zeros(seq_length, dtype=torch.bool)
    content_mask = torch.zeros(seq_length, dtype=torch.bool)
    target_mask = torch.zeros(seq_length, dtype=torch.bool)
    
    # Find clean token spans for prefix and suffix
    prefix_span = find_clean_token_span(tokenizer, full_text, adv_prefix_init, final_tokens)
    suffix_span = find_clean_token_span(tokenizer, full_text, adv_suffix_init, final_tokens)
    
    if prefix_span:
        prefix_mask[prefix_span["start"]:prefix_span["end"]] = True
    if suffix_span:
        suffix_mask[suffix_span["start"]:suffix_span["end"]] = True
    
    # Create content mask - we'll identify content by finding non-template parts in each message
    for turn in processed_conversation:  # Exclude target message
        turn_text = turn['content']
        # Find this content in the full text and create mask for its tokens
        start_pos = full_text.find(turn_text)
        if start_pos != -1:
            end_pos = start_pos + len(turn_text)
            for i, (start, end) in enumerate(char_spans):
                if start >= start_pos and end <= end_pos:
                    content_mask[i] = True
    
    # Create payload mask (between prefix and suffix)
    if prefix_span and suffix_span:
        payload_mask = torch.zeros(seq_length, dtype=torch.bool)
        payload_mask[prefix_span["end"]:suffix_span["start"]] = True
    else:
        payload_mask = torch.zeros(seq_length, dtype=torch.bool)
    
    # Create target mask
    target_start = len(full_text) - len(target_string)
    for i, (start, end) in enumerate(char_spans):
        if start >= target_start:
            target_mask[i] = True
    
    # Create input mask (everything before target)
    input_mask = torch.ones(seq_length, dtype=torch.bool)
    input_mask[target_mask.nonzero()] = False
    
    control_mask = ~(content_mask | target_mask)

    prefix_indices = torch.where(prefix_mask)[0]
    suffix_indices = torch.where(suffix_mask)[0]
    payload_indices = torch.where(payload_mask)[0]
    target_indices = torch.where(target_mask)[0]
    input_indices = torch.where(input_mask)[0]
    content_indices = torch.where(content_mask)[0]
    control_indices = torch.where(control_mask)[0]

    return {
        "tokens": final_tokens,
        "masks": {
            "optim_mask": torch.cat([prefix_indices, suffix_indices]),
            "prefix_mask": prefix_indices,
            "suffix_mask": suffix_indices,
            "payload_mask": payload_indices,
            "target_mask": target_indices,
            "input_mask": input_indices,
            "content_mask": content_indices,
            "control_mask": control_indices
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
                    del sub_result
                    gc.collect()
                    torch.cuda.empty_cache()
            
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
                    del sub_logits, sub_attentions
                    gc.collect()
                    torch.cuda.empty_cache()
            
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
