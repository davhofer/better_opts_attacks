import torch
import transformers
import typing
import json

import experiment_logger
import attack_utility

MODEL_PATH = "/data/models/hf/Llama-3.2-1B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
model.generation_config.pad_token_id = tokenizer.pad_token_id



def _v1_true(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    target_tokens: torch.tensor,
    logger: experiment_logger.ExperimentLogger,
):
    pass

def _v1_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    attention_layer = 0
):
    assert input_points.shape[0] == 1, "Only supports one-d gradients for now"
    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)

    output = model.forward(inputs_embeds=inputs_embeds, output_attentions=True)
    logits = output.logits
    attentions_tuple = output.attentions
    last_attention = attentions_tuple[attention_layer][0, :, :, input_points[masks_data["payload_mask"]]].sum()
    last_attention.backward()
    grad_optims = - (one_hot_tensor.grad[masks_data["optim_mask"], :])
    best_tokens_indices = grad_optims.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices

example_num = 2

with open(f"data/purplellama_indirect.json", "r") as purplellama_indirect_file:
    purplellama_data = json.load(purplellama_indirect_file)

purplellama_example = purplellama_data[example_num]
input_string_template = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content":  purplellama_example["test_case_prompt"]
        },
        {
            "role": "user",
            "content": purplellama_example["user_input_wrapper"]
        }
    ],
    add_generation_prompt=True,
    tokenize=False
)
target_string = purplellama_example["target"]
target_string = target_string + "<|eot_id|>"

initial_config_1 = {
    "strategy_type": "random",
    "prefix_length": 25,
    "suffix_length": 25,
    "seed": 0
}

adv_prefix_init, adv_suffix_init = attack_utility.initialize_adversarial_strings(tokenizer, initial_config_1)
input_tokenized_data = attack_utility.string_masks(tokenizer, input_string_template, adv_prefix_init, adv_suffix_init, target_string)

model.forward(input_ids=torch.unsqueeze(input_tokenized_data["tokens"], dim=0))

pass