import torch
import transformers
import random
import os
import pickle as pkl

from utils.experiment_logger import ExperimentLogger, load_experiment_logs, params_and_trace_ids_by_function
from utils.attack_utility import UNREDUCED_CE_LOSS, bulk_logits_from_embeds_iter

def generate_batched_input_tensor(src: torch.tensor, dest: torch.tensor, num_points = 64):
    assert len(src.shape) == 2 == len(dest.shape)
    difference_vector: torch.tensor = dest - src
    step_sizes = torch.linspace(0, 1, num_points).view(-1, 1, 1)
    return torch.stack([src] * num_points)  + (step_sizes.to(difference_vector.device) * (difference_vector)).to(src.device)

def path_logits(model, tokenizer, src: torch.tensor, dest: torch.tensor, num_points: int = 64):
    path_points = generate_batched_input_tensor(src, dest, num_points).half()
    path_points = path_points.to()
    logits = bulk_logits_from_embeds_iter(model, path_points)
    return logits

def target_loss_over_path(model, tokenizer, src, dest, masks_data, target_tokens, num_points: int = 64):
    logprobs_over_path = []
    for logit_batch in path_logits(model, tokenizer, src, dest, num_points):
        logprobs_over_path.extend(UNREDUCED_CE_LOSS(torch.transpose(logit_batch[:, masks_data["target_mask"] - 1], -1, -2), torch.stack([target_tokens] * logit_batch.shape[0])).sum(dim=-1).tolist())
    return logprobs_over_path

def compute_paths(
    model,
    tokenizer,
    log_folder,
    steps_fraction = 15,
    num_points_along_path = 64,
):
    result_df = load_experiment_logs(f"{log_folder}/metadata.jsonl")
    params_list, trace_ids_list = params_and_trace_ids_by_function(log_folder, result_df, "custom_gcg")
    logger = ExperimentLogger(log_folder)
    embedding = model.get_input_embeddings()
    for param_dict, trace_id in zip(params_list, trace_ids_list):
        masks_data = param_dict["input_tokenized_data"]["masks"]
        target_tokens = param_dict["input_tokenized_data"]["tokens"][masks_data["target_mask"]]
        num_steps = param_dict["custom_gcg_hyperparams"]["max_steps"]
        steps_to_check_at = list(range(0, num_steps, int(num_steps / steps_fraction)))
        topk = param_dict["custom_gcg_hyperparams"]["topk"]

        results_dict = {}
        for step_to_check_at in steps_to_check_at:
            source_emb = embedding(next(logger.query({"variable_name": "previous_best_tokens", "step_num": step_to_check_at, "trace_id": trace_id})))
            substitution_data = next(logger.query({"variable_name": "substitution_data", "step_num": step_to_check_at, "trace_id": trace_id}))
            
            path_results_for_point = []
            for sub_pt in substitution_data:
                dest_emb = embedding(sub_pt)
                logprobs_over_path = target_loss_over_path(model, tokenizer, source_emb, dest_emb, masks_data, target_tokens, num_points_along_path)
                path_results_for_point.append(logprobs_over_path)
            results_dict[step_to_check_at] = path_results_for_point

        if os.path.exists(f"{log_folder}/path_losses.pkl"):
            with open(f"{log_folder}/path_losses.pkl", "rb") as gradient_ranks_pkl:
                final_results_list = pkl.load(gradient_ranks_pkl)
        else:
            final_results_list = []
        
        final_results_list.append(results_dict)
        with open(f"{log_folder}/path_losses.pkl", "wb") as gradient_ranks_pkl:
            pkl.dump(final_results_list, gradient_ranks_pkl)
    return final_results_list

if __name__ == "__main__":
    MODEL_PATH = "/data/models/hf/Meta-Llama-3-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    LOG_FOLDER = "logs/runs28/run_20250221184322935036"
    final_results = compute_paths(model, tokenizer, LOG_FOLDER, 15, 64)
    