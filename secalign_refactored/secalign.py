from .secalign_raw_code import config as config
from .secalign_raw_code import test as test
from .secalign_raw_code import struq as struq

import fastchat.conversation
from copy import deepcopy
import torch
import transformers
from peft import PeftModel
import os
import dataclasses
from enum import Enum
from typing import Union, List, Dict


SECALIGN_DATA_PATH = "other_repos"

DEFENDED_MODEL_COMMON_PATH = "secalign_refactored/secalign_models"
MODEL_REL_PATHS = {
    
    ("mistralai", "undefended"): 'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11',
    ("mistralai", "struq"): "mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2024-07-20-05-46-17",
    ("mistralai", "secalign"): "mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2024-07-20-01-59-11_dpo_NaiveCompletion_2024-08-13-17-46-51",
    
    ("meta-llama", "undefended"): "meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02",
    ("meta-llama", "struq"): "meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2024-08-09-12-55-56",
    ("meta-llama", "secalign"): "meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2024-08-09-17-02-02_dpo_NaiveCompletion_2024-08-09-21-28-53",
    
    ("huggyllama", "undefended"): "huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00",
    ("huggyllama", "struq"): "huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2024-02-02-00-00-00",
    ("huggyllama", "secalign"): "huggyllama/llama-7b_SpclSpclSpcl_None_2024-06-02-00-00-00_dpo_NaiveCompletion_2024-07-06-07-42-23",
    
    ("mistralai-instruct", "secalign"): "mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2024-11-12-17-59-37",
    ("meta-llama-instruct", "secalign"): "meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06"
}


class Role(Enum):
    USER = 1
    ASSISTANT = 2
    SYSTEM = 3

@dataclasses.dataclass
class Message:
    role: Role
    content: str

    def __str__(self):
        return f"[{self.role.name.title()}]: {self.content}"

    @staticmethod
    def serialize(messages, user_only=False):
        if not isinstance(messages, list):
            messages = [messages]
        if user_only:
            messages = [
                {"role": m.role.name, "content": m.content} for m in messages if m.role == Role.USER
            ]
        else:
            messages = [{"role": m.role.name, "content": m.content} for m in messages]
        return messages

    @staticmethod
    def unserialize(messages: Union[dict, List[dict]]):
        if not isinstance(messages, list):
            messages = [messages]
        messages = [Message(Role[m["role"]], m["content"]) for m in messages]
        return messages

def load_defended_model(model_name, defence, model_path = None):
    global MODEL_REL_PATHS, DEFENDED_MODEL_COMMON_PATH
    if ((model_name, defence) not in MODEL_REL_PATHS) and (model_path is None):
        raise ValueError("Please give either a defended model path or pre-trained model/defence config")
    
    try:
        model_path = os.path.join(DEFENDED_MODEL_COMMON_PATH, MODEL_REL_PATHS[(model_name, defence)])
    except KeyError:
        pass
    
    return load_lora_model(model_path)

def recursive_filter(s):
    filtered = False
    while not filtered:
        for f in config.FILTERED_TOKENS:
            if f in s: s = s.replace(f, '')
        filtered = True
        for f in config.FILTERED_TOKENS:
            if f in s: filtered = False
    return s

def load_model_and_tokenizer(model_path, tokenizer_path=None, **kwargs):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, **kwargs)
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)

    if "oasst-sft-6-llama-30b" in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if "guanaco" in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if "llama-2" in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "falcon" in tokenizer_path:
        tokenizer.padding_side = "left"
    if "mistral" in tokenizer_path:
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_embedding_indices(tokenizer):
    init_values = [tokenizer.encode(v, add_special_tokens=False)[0] for v in config.TEXTUAL_DELM_TOKENS]
    ignore_values = [i for i in range(len(tokenizer)) if tokenizer.decode(i) == "#"]
    return init_values, ignore_values


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    REAL_DELIMITERS_INIT_EMBD_IND, _ = get_embedding_indices(tokenizer)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens] = input_embeddings_avg
        output_embeddings[-num_new_tokens] = output_embeddings_avg

        for i in range(len(config.SPECIAL_DELM_TOKENS)): ### initialize real delimiter's embedding by the existing ones
            input_embeddings[-num_new_tokens+i+1] = input_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]
            output_embeddings[-num_new_tokens+i+1] = output_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]

def load_lora_model(model_name_or_path, **kwargs):
    configs = model_name_or_path.split('/')[-1].split('_') + ['Frontend-Delimiter-Placeholder', 'None']
    for alignment in ['dpo', 'kto', 'orpo']:
        base_model_index = model_name_or_path.find(alignment) - 1
        if base_model_index > 0:
            break
        else:
            base_model_index = False

    base_model_path = model_name_or_path[:base_model_index] if base_model_index else model_name_or_path
    frontend_delimiters = configs[1] if configs[1] in config.DELIMITERS else base_model_path.split('/')[-1]
    model, tokenizer = load_model_and_tokenizer(base_model_path, **kwargs)
    
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = config.DEFAULT_TOKENS['pad_token']
    special_tokens_dict["eos_token"] = config.DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = config.DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = config.DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = config.SPECIAL_DELM_TOKENS

    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)
    tokenizer.model_max_length = 512 ### the default value is too large for model.generation_config.max_new_tokens
    if base_model_index:
        model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=False)
    return model, tokenizer, frontend_delimiters



def test_model_output(llm_input, model, tokenizer):
    model.generation_config.max_new_tokens = tokenizer.model_max_length
    model.generation_config.do_sample = False
    model.generation_config.temperature = 0.0

    llm_output = []
    for i, inpt in enumerate(llm_input):
        input_ids = _tokenize_fn([inpt], tokenizer)['input_ids'][0].unsqueeze(0)
        outp = tokenizer.decode(
            model.generate(
                input_ids.to(model.device),
                attention_mask=torch.ones_like(input_ids).to(model.device),
                generation_config=model.generation_config,
                pad_token_id=tokenizer.pad_token_id,
            )[0][input_ids.shape[1]:]
        )
        start = 0 
        while outp[start] == ' ':
            start += 1
        outp = outp[start:outp.find(tokenizer.eos_token)]

        llm_output.append(outp)
    return llm_output



@dataclasses.dataclass
class StruqConversation(fastchat.conversation.Conversation):
    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        seps = [self.sep, self.sep2]
        ret = system_prompt + self.sep
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + "\n" + message + seps[i % 2]
            else:
                ret += role + "\n"
        return ret
    
    def copy(self):
        return StruqConversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


ROLE_STR_MAP = {
    "system": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT
}
def get_conversation_tokens(
    model: transformers.PreTrainedModel,
    tokenizer,
    defence,
    frontend_delimiters,
    conversation
) -> str:
    if "instruct" in model.name.lower():
        return tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)["input_ids"]
    else:
        _ = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
        inst_delm = config.DELIMITERS[frontend_delimiters][0]
        _ = config.DELIMITERS[frontend_delimiters][1]
        resp_delm = config.DELIMITERS[frontend_delimiters][2]
        try:
            fastchat.conversation.register_conv_template(
                StruqConversation(
                    name=f"struq_{model.name}_{defence}",
                    system_message=config.SYS_INPUT,
                    roles=(inst_delm, resp_delm),
                    sep="\n\n",
                    sep2="</s>",
                ),
            )
        except AssertionError:
            pass        
        conv_template = fastchat.conversation.get_conv_template(f"struq_{model.name}_{defence}")
        conversation = [Message(role=ROLE_STR_MAP[x["role"]], content=x["content"]) for x in conversation]
        if conversation[0].role == Role.SYSTEM:
            conv_template.set_system_message(conversation[0].content)
        conv_template.messages = []
        assert conversation[1].role == Role.USER
        conv_template.append_message(conv_template.roles[0], conversation[1].content)
        conv_template.append_message(conv_template.roles[1], "") # equivalent of add_generation_prompt
        sep = deepcopy(conv_template.sep); conv_template.sep = ""
        prompt = conv_template.get_prompt()
        conv_template.sep = sep
        tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            # tokenizer(" ", add_special_tokens=False).input_ids + \
            # tokenizer(conv_template.sep, add_special_tokens=False).input_ids
        return tokens


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokens = tokenizer(
        strings,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids_lens = [tokenized["input_ids"].input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokens]
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "input_ids_lens": input_ids_lens
    }



def get_static_single_turn_message_according_to_template(messages: List[Message], conv_template: fastchat.conversation.Conversation):
    assert len(messages) == 2, "Only single turn messages with system and user input supported"
    conv_template.messages = []
    conv_template.set_system_message(messages[0].content)
    conv_template.append_message(conv_template.roles[0], messages[1].content)
    
    if conv_template.name == "struq":
        sep = deepcopy(conv_template.sep); conv_template.sep = ''
        final_string = conv_template.get_prompt() + " "
        conv_template.sep = sep
        return final_string
    else:
        return conv_template.get_prompt()

if __name__ == "__main__":

    import gc
    import traceback


    for (model_name, defence) in MODEL_REL_PATHS:
        # if ((defence != "struq") and (defence != "secalign")) or ("instruct" not in model_name):
            # continue
        try:
            model, tokenizer, frontend_delimiters = load_defended_model(model_name, defence)
            conv_template = get_struq_conv_template(model_name, defence, frontend_delimiters)            
            test_conversation = [
                {
                    "role": Role.SYSTEM,
                    "content": "You are a helpful assistant"
                },
                {
                    "role": Role.USER,
                    "content": "What is 2+2?"
                },
                {
                    "role": Role.ASSISTANT,
                    "content": None
                }
            ]
            test_conversation = [Message(message["role"], message["content"]) for message in test_conversation]

            input_toks = gcg_suffix_manager.get_input_ids(test_messages, static_only=True)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()


























