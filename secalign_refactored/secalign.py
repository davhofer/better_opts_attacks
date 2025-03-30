import secalign_raw_code.config as config
import secalign_raw_code.test as test
import secalign_raw_code.struq as struq

import fastchat.conversation
from copy import deepcopy
import torch
import transformers
from peft import PeftModel
import os
import dataclasses
from enum import Enum
from typing import Union, List, Dict

@dataclasses.dataclass
class CustomConversation(fastchat.conversation.Conversation):
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
        return CustomConversation(
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

class SuffixManager:
    """Suffix manager for adversarial suffix generation."""

    valid_templates = (
        "llama-3",
        "llama-2",
        "vicuna_v1.1",
        "mistral",
        "chatgpt",
        "completion",
        "raw",
        "tinyllama",
        "struq",
        "bipia"
    )

    def __init__(self, *, tokenizer, use_system_instructions, conv_template):
        """Initialize suffix manager.

        Args:
            tokenizer: Tokenizer for model.
            use_system_instructions: Whether to use system instructions.
            conv_template: Conversation template.
        """
        self.tokenizer = tokenizer
        self.use_system_instructions = use_system_instructions
        self.conv_template = conv_template
        self.sep_tokens = self.tokenizer(self.conv_template.sep, add_special_tokens=False).input_ids
        self.num_tok_sep = len(self.sep_tokens)
        if self.conv_template.sep2 not in (None, ""):
            self.num_tok_sep2 = len(self.tokenizer(self.conv_template.sep2, add_special_tokens=False).input_ids)
        else:
            self.num_tok_sep2 = 0
        
        if self.conv_template.stop_str not in (None, ""):
            self.num_tok_sep2 += 1
        
    @torch.no_grad()
    def get_input_ids(
        self,
        messages,
        adv_suffix: str | None = None,
        target: str | None = None,
        static_only: bool = False,
    ) -> tuple[torch.Tensor, slice, slice, slice]:
        """Turn messages into token ids. Run once for attack step.

        Compute token ids for given messages and target, along with slices
        tracking positions of important tokens.

        Args:
            messages: Messages in the conversation.
            adv_suffix: Current adversarial suffix.
            target: Current target output for model.
            static_only: If True, only return token ids for static tokens.

        Returns:
            input_ids: Token ids for messages and target.
            optim_slice: Slice of input_ids corresponding to tokens to optimize.
            target_slice: Slice of input_ids corresponding to target.
            loss_slice: Slice of input_ids corresponding to loss.
        """
        # This code was tested with llama-2 and vicuna_v1.1 templates but remove
        # this check to experiment with others.

        self.conv_template.messages = []

        if messages[0].content:
            self.conv_template.set_system_message(messages[0].content)

        user_msg = messages[1].content
        if len(messages) <= 2:
            # going here!
            # Toxicity scenario
            self.conv_template.append_message(
                self.conv_template.roles[0], messages[1].content
            )  # user rules
        else:
            if not self.use_system_instructions:
                self.conv_template.append_message(
                    self.conv_template.roles[0], messages[1].content
                )  # user rules
                self.conv_template.append_message(
                    self.conv_template.roles[1], messages[2].content
                )  # asst response
                user_msg = messages[3].content
            # user msg
            self.conv_template.append_message(self.conv_template.roles[0], user_msg)
        
        sep = deepcopy(self.conv_template.sep); self.conv_template.sep = ''
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.sep_tokens
               
        ### toks: ...Print exactly 'hacked!' \n\n
        num_static_tokens = len(toks)
        
        if user_msg:
            num_static_tokens -= self.num_tok_sep
        elif self.conv_template.name == "vicuna_v1.1":
            pass
        else:
            num_static_tokens -= self.num_tok_sep2

        static_input_ids = torch.tensor(toks[:num_static_tokens])
        if static_only:
            self.conv_template.sep = sep
            return static_input_ids

        """
        # user msg + adv suffix
        if user_msg:
            #print(user_msg, self.is_tiktoken); exit() #... Print exactly 'hacked!', True
            if adv_suffix.startswith(" ") and self.is_tiktoken:
                # NOTE: space is part of token in tiktoken, i.e., " !" != "!".
                self.conv_template.update_last_message(f"{user_msg}{adv_suffix}")
            else:
                self.conv_template.update_last_message(f"{user_msg} {adv_suffix}")
        else:
            self.conv_template.update_last_message(adv_suffix)
        """

        # It seems that we do not need toks and self.conv_template after this function
        # Thus, we can calculate toks by adding (user_msg, adv_suffix, '\n\n', self.conv_template.roles[1]) tokens directly
        # instead of asking self.tokenizer to do self.tokenizer(self.conv_template.get_prompt()).input_ids
        
        #print('0', toks, self.tokenizer.decode(toks))
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.tokenizer(adv_suffix, add_special_tokens=False).input_ids + \
               self.sep_tokens
        #print('1', toks, self.tokenizer.decode(toks))#; exit() # the last ! is combined with \n or \n\n
        optim_slice = slice(num_static_tokens, len(toks) - self.num_tok_sep)
        
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.tokenizer(adv_suffix, add_special_tokens=False).input_ids + \
               self.sep_tokens + \
               self.tokenizer(self.conv_template.roles[1], add_special_tokens=False).input_ids + \
               self.tokenizer('\n', add_special_tokens=False).input_ids
        #print('2', toks, self.tokenizer.decode(toks))
        #self.conv_template.append_message(self.conv_template.roles[1], None)
        #toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        assistant_role_slice = slice(optim_slice.stop, len(toks))

        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.tokenizer(adv_suffix, add_special_tokens=False).input_ids + \
               self.sep_tokens + \
               self.tokenizer(self.conv_template.roles[1], add_special_tokens=False).input_ids + \
               self.tokenizer('\n' + target, add_special_tokens=False).input_ids + \
               self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False).input_ids
        # self.tokenizer('\n', add_special_tokens=False).input_ids + \              
        # self.tokenizer(target, add_special_tokens=False).input_ids
        
        #print('3', toks, self.tokenizer.decode(toks))#; exit()
        #self.conv_template.update_last_message(target)  # asst target
        #toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        target_slice = slice(assistant_role_slice.stop, len(toks) - self.num_tok_sep2)
        loss_slice = slice(assistant_role_slice.stop - 1, len(toks) - self.num_tok_sep2 - 1)

        # DEBUG
        #print('\'' + self.tokenizer.decode(toks[optim_slice]) + '\'')
        #print('\'' + self.tokenizer.decode(toks[assistant_role_slice]) + '\'')
        #print('\'' + self.tokenizer.decode(toks[target_slice]) + '\'')
        #print('\'' + self.tokenizer.decode(toks[loss_slice]) + '\'')
        #exit()
        # import pdb; pdb.set_trace()

        # Don't need final sep tokens
        input_ids = torch.tensor(toks[: target_slice.stop])
        self.conv_template.sep = sep
        return input_ids, optim_slice, target_slice, loss_slice

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


class GCGSuffixManager:
    """Suffix manager for adversarial suffix generation."""

    valid_templates = (
        "llama-3",
        "llama-2",
        "vicuna_v1.1",
        "mistral",
        "chatgpt",
        "completion",
        "raw",
        "tinyllama",
        "struq",
        "bipia"
    )

    def __init__(self, *, tokenizer, use_system_instructions, conv_template):
        """Initialize suffix manager.

        Args:
            tokenizer: Tokenizer for model.
            use_system_instructions: Whether to use system instructions.
            conv_template: Conversation template.
        """
        self.tokenizer = tokenizer
        self.use_system_instructions = use_system_instructions
        self.conv_template = conv_template
        self.is_tiktoken = not isinstance(tokenizer, transformers.AutoTokenizer)

        self.sep_tokens = self.tokenizer(self.conv_template.sep, add_special_tokens=False).input_ids
        non_empty = [(True if self.tokenizer.decode([token]) != '' else False) for token in self.sep_tokens]
        self.num_tok_sep = len(self.sep_tokens)
        #self.num_tok_sep = sum(non_empty)
        #sep_tokens = []
        #for i, token in enumerate(self.sep_tokens):
            #if non_empty[i]: sep_tokens.append(token)
        #self.sep_tokens = sep_tokens
        
        #self.sep_tokens = self.sep_tokens[non_empty]
        # if self.num_tok_sep is wrong, low ASR is observed with no error in running and debugging
        #print(self.tokenizer(self.conv_template.sep, add_special_tokens=False).input_ids)#; exit()
        if self.conv_template.name == "chatgpt":
            # Space is subsumed by following token in GPT tokenizer
            assert self.conv_template.sep == " ", self.conv_template.sep
            self.num_tok_sep = 0
        elif self.conv_template.name == "llama-3":
            # FastChat adds <|eot_id|> after each message, but it's not sep.
            # Not exactly sure why, but not we need to manually set
            # self.num_tok_sep because sep is just "".
            # https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L167
            self.num_tok_sep = 1
        #elif self.conv_template.name == "struq":
            # Somehow "\n\n" sep in Alpaca is tokenized to 3 tokens instead of 2.
            # This requires a manual fix here.
        #    self.num_tok_sep = 2
        elif self.conv_template.name == "bipia":
            # Somehow "\n\n" sep in Alpaca is tokenized to 3 tokens instead of 2.
            # This requires a manual fix here.
            # \n\n sep is tokenized to 2 in llama3
            self.num_tok_sep = 2 # change to 1 for llama3
        self.num_tok_sep2 = 0
        if self.conv_template.sep2 not in (None, ""):
            self.num_tok_sep2 = len(
                self.tokenizer(self.conv_template.sep2, add_special_tokens=False).input_ids
            )
        if self.conv_template.stop_str not in (None, ""):
            self.num_tok_sep2 += 1
        
        print('num_tok_sep:', self.num_tok_sep)
        print('num_tok_sep2:', self.num_tok_sep2)

    @torch.no_grad()
    def get_input_ids(
        self,
        messages: list,
        adv_suffix: str | None = None,
        target: str | None = None,
        static_only: bool = False,
    ) -> tuple[torch.Tensor, slice, slice, slice]:
        """Turn messages into token ids. Run once for attack step.

        Compute token ids for given messages and target, along with slices
        tracking positions of important tokens.

        Args:
            messages: Messages in the conversation.
            adv_suffix: Current adversarial suffix.
            target: Current target output for model.
            static_only: If True, only return token ids for static tokens.

        Returns:
            input_ids: Token ids for messages and target.
            optim_slice: Slice of input_ids corresponding to tokens to optimize.
            target_slice: Slice of input_ids corresponding to target.
            loss_slice: Slice of input_ids corresponding to loss.
        """
        # This code was tested with llama-2 and vicuna_v1.1 templates but remove
        # this check to experiment with others.

        if self.conv_template.name not in self.valid_templates:
            raise NotImplementedError(
                f"{self.conv_template.name} is not implemented! Please use one "
                f"of {self.valid_templates}"
            )
        self.conv_template.messages = []

        if messages[0].content:
            self.conv_template.set_system_message(messages[0].content)

        user_msg = messages[1].content
        if len(messages) <= 2:
            # going here!
            # Toxicity scenario
            self.conv_template.append_message(
                self.conv_template.roles[0], messages[1].content
            )  # user rules
        else:
            if not self.use_system_instructions:
                self.conv_template.append_message(
                    self.conv_template.roles[0], messages[1].content
                )  # user rules
                self.conv_template.append_message(
                    self.conv_template.roles[1], messages[2].content
                )  # asst response
                user_msg = messages[3].content
            # user msg
            self.conv_template.append_message(self.conv_template.roles[0], user_msg)
        
        sep = deepcopy(self.conv_template.sep); self.conv_template.sep = ''
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.sep_tokens
               
        ### toks: ...Print exactly 'hacked!' \n\n
        num_static_tokens = len(toks)
        
        if user_msg:
            num_static_tokens -= self.num_tok_sep
        elif self.conv_template.name == "vicuna_v1.1":
            pass
        else:
            num_static_tokens -= self.num_tok_sep2

        static_input_ids = torch.tensor(toks[:num_static_tokens])
        if static_only:
            self.conv_template.sep = sep
            return static_input_ids

        """
        # user msg + adv suffix
        if user_msg:
            #print(user_msg, self.is_tiktoken); exit() #... Print exactly 'hacked!', True
            if adv_suffix.startswith(" ") and self.is_tiktoken:
                # NOTE: space is part of token in tiktoken, i.e., " !" != "!".
                self.conv_template.update_last_message(f"{user_msg}{adv_suffix}")
            else:
                self.conv_template.update_last_message(f"{user_msg} {adv_suffix}")
        else:
            self.conv_template.update_last_message(adv_suffix)
        """

        # It seems that we do not need toks and self.conv_template after this function
        # Thus, we can calculate toks by adding (user_msg, adv_suffix, '\n\n', self.conv_template.roles[1]) tokens directly
        # instead of asking self.tokenizer to do self.tokenizer(self.conv_template.get_prompt()).input_ids
        
        #print('0', toks, self.tokenizer.decode(toks))
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.tokenizer(adv_suffix, add_special_tokens=False).input_ids + \
               self.sep_tokens
        #print('1', toks, self.tokenizer.decode(toks))#; exit() # the last ! is combined with \n or \n\n
        optim_slice = slice(num_static_tokens, len(toks) - self.num_tok_sep)
        
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.tokenizer(adv_suffix, add_special_tokens=False).input_ids + \
               self.sep_tokens + \
               self.tokenizer(self.conv_template.roles[1], add_special_tokens=False).input_ids + \
               self.tokenizer('\n', add_special_tokens=False).input_ids
        #print('2', toks, self.tokenizer.decode(toks))
        #self.conv_template.append_message(self.conv_template.roles[1], None)
        #toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        assistant_role_slice = slice(optim_slice.stop, len(toks))

        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids + \
               self.tokenizer(' ', add_special_tokens=False).input_ids + \
               self.tokenizer(adv_suffix, add_special_tokens=False).input_ids + \
               self.sep_tokens + \
               self.tokenizer(self.conv_template.roles[1], add_special_tokens=False).input_ids + \
               self.tokenizer('\n' + target, add_special_tokens=False).input_ids + \
               self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False).input_ids
        # self.tokenizer('\n', add_special_tokens=False).input_ids + \              
        # self.tokenizer(target, add_special_tokens=False).input_ids
        
        #print('3', toks, self.tokenizer.decode(toks))#; exit()
        #self.conv_template.update_last_message(target)  # asst target
        #toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        target_slice = slice(assistant_role_slice.stop, len(toks) - self.num_tok_sep2)
        loss_slice = slice(assistant_role_slice.stop - 1, len(toks) - self.num_tok_sep2 - 1)

        # DEBUG
        #print('\'' + self.tokenizer.decode(toks[optim_slice]) + '\'')
        #print('\'' + self.tokenizer.decode(toks[assistant_role_slice]) + '\'')
        #print('\'' + self.tokenizer.decode(toks[target_slice]) + '\'')
        #print('\'' + self.tokenizer.decode(toks[loss_slice]) + '\'')
        #exit()
        # import pdb; pdb.set_trace()

        # Don't need final sep tokens
        input_ids = torch.tensor(toks[: target_slice.stop])
        self.conv_template.sep = sep
        return input_ids, optim_slice, target_slice, loss_slice

    @torch.no_grad()
    def gen_eval_inputs(
        self,
        messages: list,
        suffix: str,
        target: str,
        num_fixed_tokens: int = 0,
        max_target_len: int | None = None,
    ):
        """Generate inputs for evaluation. Run once for each sample

        Returns:
            eval_inputs: Inputs for evaluation.
        """
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids
        suffix_ids.requires_grad_(False)
        suffix_ids.squeeze_(0)

        out = self.get_input_ids(messages, suffix, target)
        orig_input_ids, optim_slice, target_slice, loss_slice = out
        #print(optim_slice, optim_slice.start, optim_slice.stop)
        if max_target_len is not None:
            # Adjust target slice to be at most max_target_len
            end = min(target_slice.stop, target_slice.start + max_target_len)
            target_slice = slice(target_slice.start, end)
            loss_slice = slice(loss_slice.start, end - 1)

        # Offset everything to ignore static tokens which are processed separately
        orig_input_ids = orig_input_ids[num_fixed_tokens:]
        optim_slice = slice(
            optim_slice.start - num_fixed_tokens,
            optim_slice.stop - num_fixed_tokens,
        )
        #print(optim_slice, optim_slice.start, optim_slice.stop); exit()
        target_slice = slice(
            target_slice.start - num_fixed_tokens,
            target_slice.stop - num_fixed_tokens,
        )
        loss_slice = slice(
            loss_slice.start - num_fixed_tokens,
            loss_slice.stop - num_fixed_tokens,
        )
        target_ids = orig_input_ids[target_slice]
        assert target_ids.ndim == 1
        target_ids.requires_grad_(False)

        eval_input = EvalInput(
            suffix_ids=suffix_ids,
            dynamic_input_ids=orig_input_ids,
            target_ids=target_ids,
            optim_slice=optim_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
        )
        return eval_input

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
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs)
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

def form_secalign_input(data, prompt_format, apply_defensive_filter = True):
    llm_input = []
    for i, d in enumerate(data):        
        d_item = deepcopy(d)
        # if d_item['input'][-1] != '.' and d_item['input'][-1] != '!' and d_item['input'][-1] != '?':
        #     d_item['input'] += '.'
        # d_item['input'] += ' '
        
        if apply_defensive_filter:
            d_item['input'] = recursive_filter(d_item['input'])  

        llm_input_i = prompt_format['prompt_input'].format_map(d_item)
        llm_input.append(llm_input_i)
    return llm_input


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenizer(
        strings,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
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

# def secalign_init()

def get_model_input_strings(input_convs, frontend_delimiters):
    model, tokenizer, frontend_delimiters, training_attacks = test.load_lora_model(args.model_name_or_path, args.device)

    # cfg = config_dict.ConfigDict()
    # cfg.name = "gcg"  # Attack name
    # cfg.seed = 0  # Random seed
    # cfg.log_freq = 20
    # cfg.adv_suffix_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    # # Init suffix length (auto-generated from adv_suffix_init)
    # cfg.init_suffix_len = -1
    # cfg.num_steps = 500
    # cfg.fixed_params = True  # Used fixed scenario params in each iteration
    # cfg.allow_non_ascii = False
    # cfg.batch_size = 512  # Number of candidates to evaluate in each step
    # # NOTE: Reduce mini_batch_size if OOM
    # cfg.mini_batch_size = 64#32 #128 #256  # -1 for full batch (config.batch_size)
    # cfg.seq_len = 50  # Max sequence length for computing loss
    # cfg.loss_temperature = 1.0  # Temperature for computing loss
    # cfg.max_queries = -1  # Max number of queries (default: -1 for no limit)
    # cfg.skip_mode = "none"  # "none", "visited", "seen"
    # cfg.add_space = False  # Add metaspace in front of target
    # cfg.topk = 256
    # cfg.num_coords = (1, 1)  # Number of coordinates to change in one step
    # cfg.mu = 0.0  # Momentum parameter
    # cfg.custom_name = ""
    # cfg.log_dir = args.model_name_or_path if os.path.exists(args.model_name_or_path) else (args.model_name_or_path+'-log')
    # cfg.sample_id = -1 # to be initialized in every run of the sample

    prompt_template = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
    inst_delm = config.DELIMITERS[frontend_delimiters][0]
    data_delm = config.DELIMITERS[frontend_delimiters][1]
    resp_delm = config.DELIMITERS[frontend_delimiters][2]

    fastchat.conversation.register_conv_template(
        test.CustomConversation(
            name="struq",
            system_message=config.SYS_INPUT,
            roles=(inst_delm, resp_delm),
            sep="\n\n",
            sep2="</s>",
        )
    )

    # def eval_func(adv_suffix, messages):
    #     inst, data = messages[1].content.split(f'\n\n{data_delm}\n')
    #     return test.test_model_output([
    #         prompt_template.format_map({
    #             "instruction": inst,
    #             "input": data + ' ' + adv_suffix
    #         })
    #     ], model, tokenizer)

    suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            use_system_instructions=False,
            conv_template=fastchat.conversation.get_conv_template("struq"),
        )

    # attack = GCGAttack(
    #     config=cfg,
    #     model=model,
    #     tokenizer=tokenizer,
    #     eval_func=eval_func,
    #     suffix_manager=suffix_manager,
    #     not_allowed_tokens=None if cfg.allow_non_ascii else get_nonascii_toks(tokenizer),
    # )

    data = [d for d in jload(args.data_path) if d["input"] != ""]
    sample_ids = list(range(len(data))) if args.sample_ids is None else args.sample_ids
    data = [data[i] for i in sample_ids]
    test.form_llm_input(
        data,
        lambda x: gcg(x, attack, cfg, data_delm),
        config.PROMPT_FORMAT[frontend_delimiters],
        apply_defensive_filter=not (training_attacks == 'None' and len(args.model_name_or_path.split('/')[-1].split('_')) == 4),
        defense=args.defense,
        sample_ids=sample_ids,
    )

def get_template(model_name, defence, frontend_delimiters):
    if "instruct" in model_name:
        if "mistral" in model_name:
            return fastchat.conversation.get_conv_template("mistral")
        if "Llama-3" in model_name:
            return fastchat.conversation.get_conv_template("")







if __name__ == "__main__":

    import gc
    import traceback

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

    for (model_name, defence) in MODEL_REL_PATHS:
        if ((defence != "struq") and (defence != "secalign")) or ("instruct" not in model_name):
            continue
        try:
            model, tokenizer, frontend_delimiters = load_defended_model(model_name, defence)
            print(f"model_name: {model_name}, defence: {defence}")
            test_str = "Hello, how are you?"
            tokens = tokenizer(test_str, return_tensors="pt")
            output = model.generate(input_ids=tokens["input_ids"].to(model.device), attention_mask=tokens["attention_mask"].to(model.device), max_new_tokens=1000)
            print(f"OUTPUT: {tokenizer.batch_decode(output)[0]}")
            pass
            prompt_template = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
            inst_delm = config.DELIMITERS[frontend_delimiters][0]
            data_delm = config.DELIMITERS[frontend_delimiters][1]
            resp_delm = config.DELIMITERS[frontend_delimiters][2]

            if defence in ["secalign", "struq"]:
                fastchat.conversation.register_conv_template(
                    CustomConversation(
                        name="struq",
                        system_message=config.SYS_INPUT,
                        roles=(inst_delm, resp_delm),
                        sep="\n\n",
                        sep2="</s>",
                    ),
                    override=True
                )
                gcg_suffix_manager = GCGSuffixManager(tokenizer=tokenizer, use_system_instructions=True, conv_template=fastchat.conversation.get_conv_template("struq"))
            else:
                gcg_suffix_manager = GCGSuffixManager(tokenizer=tokenizer, use_system_instructions=True, conv_template=fastchat.conversation.get_conv_template())
            
            test_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": "What is 2+2?"
                }
            ]

            test_messages = [Message(message["role"], message["content"]) for message in test_messages]

            input_toks = gcg_suffix_manager.get_input_ids(test_messages, static_only=True)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()


























