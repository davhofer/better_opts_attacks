import transformers
import torch
import typing
import random
import gc
import utils.experiment_logger as experiment_logger

def invertibility_filter(token_ids, **kwargs):
    tokenizer = kwargs.get("tokenizer", None)
    if tokenizer is None:
        raise ValueError(f"Tokenizer required for evaluating invertibility, you complete dingus.")
    
    try:
        bool_val = all(tokenizer.encode(tokenizer.decode(token_ids, clean_up_tokenization_spaces=False), add_special_tokens=False, return_tensors="pt")[0] == token_ids)
        return bool_val
    except Exception:
        return False

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
    tokenizer: "transformers.AutoTokenizer",
    input_string_template: str,
    adv_pre_init: str,
    adv_suf_init: str,
    target_string: str,
    prefix_placeholder: str = "<ADV_PREFIX>",
    suffix_placeholder: str = "<ADV_SUFFIX>",
):
    """
    Create masks for different parts of a tokenized string.
    
    Args:
        tokenizer: HuggingFace tokenizer
        input_string_template: Template string with placeholders for adversarial prefix and suffix
        adv_pre_init: String to replace prefix_placeholder
        adv_suf_init: String to replace suffix_placeholder
        target_string: Target string to be appended after the input string
        prefix_placeholder: Placeholder for prefix in the template
        suffix_placeholder: Placeholder for suffix in the template
    
    Returns:
        Dictionary containing tokens and various masks
    """
    # Replace placeholders in the template with the initial strings
    if prefix_placeholder in input_string_template and suffix_placeholder in input_string_template:
        prefix_pos = input_string_template.find(prefix_placeholder)
        suffix_pos = input_string_template.find(suffix_placeholder)
        
        # Extract the payload string between placeholders
        payload_string = input_string_template[prefix_pos + len(prefix_placeholder):suffix_pos]
        
        # Create the full text by replacing placeholders
        full_text = (
            input_string_template[:prefix_pos] + 
            adv_pre_init + 
            payload_string + 
            adv_suf_init + 
            input_string_template[suffix_pos + len(suffix_placeholder):]
        ) + target_string
    else:
        raise ValueError(f"Template must contain both {prefix_placeholder} and {suffix_placeholder}")
    
    # Tokenize the full text
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding.input_ids
    char_spans = encoding.offset_mapping
    
    # Convert to torch tensor for consistency with the existing code
    final_tokens = torch.tensor(tokens)
    seq_length = len(final_tokens)
    
    # Find spans for different components
    prefix_span = find_clean_token_span(tokenizer, full_text, adv_pre_init, final_tokens)
    suffix_span = find_clean_token_span(tokenizer, full_text, adv_suf_init, final_tokens)
    
    # Create masks using clean token spans
    prefix_mask = torch.zeros(seq_length, dtype=torch.bool)
    suffix_mask = torch.zeros(seq_length, dtype=torch.bool)
    target_mask = torch.zeros(seq_length, dtype=torch.bool)
    
    if prefix_span:
        prefix_mask[prefix_span["start"]:prefix_span["end"]] = True
    if suffix_span:
        suffix_mask[suffix_span["start"]:suffix_span["end"]] = True
    
    # Create payload mask (between prefix and suffix)
    payload_mask = torch.zeros(seq_length, dtype=torch.bool)
    if prefix_span and suffix_span:
        payload_mask[prefix_span["end"]:suffix_span["start"]] = True
    
    # Create target mask
    target_start = len(full_text) - len(target_string)
    for i, (start, end) in enumerate(char_spans):
        if start >= target_start:
            target_mask[i] = True
    
    # Create input mask (everything before target)
    input_mask = torch.ones(seq_length, dtype=torch.bool)
    input_mask[target_mask.nonzero()] = False
    
    # Convert boolean masks to index tensors for compatibility with existing code
    prefix_indices = torch.where(prefix_mask)[0]
    suffix_indices = torch.where(suffix_mask)[0]
    payload_indices = torch.where(payload_mask)[0]
    target_indices = torch.where(target_mask)[0]
    input_indices = torch.where(input_mask)[0]
    
    # Create the optim_mask as the combination of prefix and suffix indices
    optim_mask = torch.cat([prefix_indices, suffix_indices])
    
    return {
        "tokens": final_tokens,
        "masks": {
            "optim_mask": optim_mask,
            "prefix_mask": prefix_indices,
            "suffix_mask": suffix_indices,
            "target_mask": target_indices,
            "input_mask": input_indices,
            "payload_mask": payload_indices
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
    decoded = tokenizer.decode(span_tokens, clean_up_tokenization_spaces=False)
    
    assert decoded in target_text, f"Decoded string: {decoded} not a subset of target_text: {target_text}"
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
            if prefix_filter(prefix_random_tokens, **filter_metadata):
                break
        
        while True:
            suffix_random_tokens = []
            for _ in range(suffix_length):
                rand_token = random.randint(0, tokenizer.vocab_size)
                suffix_random_tokens.append(rand_token)
            suffix_random_tokens = torch.tensor(suffix_random_tokens)
            if suffix_filter(suffix_random_tokens, **filter_metadata):
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

BULK_FORWARD_DEFAULT_BSZ = 512
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
                data_piece_result = model(input_ids=data_piece.to(model.device)).logits
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
                logits = model(input_ids=data_piece.to(model.device)).logits
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
                data_piece_result = model(inputs_embeds=data_piece.to(model.device)).logits
            except torch.cuda.OutOfMemoryError:
                data_piece_result = bulk_logits_from_embeds(model, data_piece, batch_size // 2, generation_params)
            list_of_results.append(data_piece_result.detach())
            del data_piece_result
            gc.collect()
            torch.cuda.empty_cache()
    return torch.cat(list_of_results, dim=0)

def bulk_logits_from_embeds_iter(
    model: transformers.AutoModelForCausalLM,  # Changed to AutoModelForCausalLM to match logits usage
    data: torch.tensor,
    batch_size=BULK_FORWARD_DEFAULT_BSZ,
    generation_params=DEFAULT_GENERATION_PARAMS
):
    """
    Iterator version of bulk_embeds that yields logits one batch at a time
    to reduce memory usage. Returns logits from the model.
    
    Args:
        model: HuggingFace transformer model
        data: Input tensor of token IDs
        batch_size: Initial batch size (will be reduced if OOM occurs)
        generation_params: Additional parameters for generation
        
    Yields:
        torch.Tensor: Logit tensors for each batch (on CPU)
    """
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_piece = data[i:i + batch_size]
            try:
                # Get logits from model
                logits = model(inputs_embeds=data_piece.to(model.device)).logits
                
                # Move to CPU and clear GPU memory
                cpu_logits = logits.cpu()
                del logits
                gc.collect()
                torch.cuda.empty_cache()
                
                yield cpu_logits
                
            except torch.cuda.OutOfMemoryError:
                # If OOM occurs, recursively process with smaller batch size
                sub_iterator = bulk_logits_from_embeds_iter(
                    model,
                    data_piece,
                    max(1, batch_size // 2),
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

BULK_ATT_FORWARD_DEFAULT_SIZE=128
def bulk_forward_iter(
    model: transformers.AutoModelForCausalLM,
    data: torch.tensor,
    batch_size=BULK_ATT_FORWARD_DEFAULT_SIZE,
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
                output = model(
                    input_ids=data_piece.to(model.device),
                    output_attentions=True
                )
                yield output.logits, output.attentions
                
            except torch.cuda.OutOfMemoryError:
                # If OOM occurs, recursively process with smaller batch size
                gc.collect()
                torch.cuda.empty_cache()
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
    logger: experiment_logger.ExperimentLogger = None,
):
    target_mask = masks_data["target_mask"]
    losses_list = []
    for logit_piece in bulk_logits_iter(model, input_points):
        loss_tensor = UNREDUCED_CE_LOSS(torch.transpose(logit_piece[:, target_mask - 1, :], 1, 2), target_tokens.repeat((logit_piece.shape[0], 1)).to(logit_piece.device)).sum(dim=1)
        losses_list.append(loss_tensor)
    loss_tensor = torch.cat(losses_list)
    return loss_tensor

def target_logprobs_from_embeds(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    inputs_embeds: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    target_tokens: torch.tensor,
    logger: experiment_logger.ExperimentLogger = None
):
    target_mask = masks_data["target_mask"]
    losses_list = []
    for logit_piece in bulk_logits_from_embeds_iter(model, inputs_embeds):
        loss_tensor = UNREDUCED_CE_LOSS(torch.transpose(logit_piece[:, target_mask - 1, :], 1, 2), target_tokens.repeat((logit_piece.shape[0], 1)).to(logit_piece.device)).sum(dim=1)
        losses_list.append(loss_tensor)
    loss_tensor = torch.cat(losses_list)
    return loss_tensor

DEFAULT_TEXT_GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 200
}

def default_best_choice_function(model, tokenizer, input_tokenized_data, best_tokens_sequences, logger, **kwargs):
    masks_data = input_tokenized_data["masks"]
    best_index = torch.argmin(target_logprobs(model, tokenizer, torch.stack(best_tokens_sequences), masks_data, input_tokenized_data["tokens"][masks_data["target_mask"]], logger))
    return {
        "tokens": best_tokens_sequences[best_index],
        "masks": masks_data
    }
