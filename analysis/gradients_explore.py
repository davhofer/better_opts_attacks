# %%
import torch
import transformers
import matplotlib.pyplot as plt
import random
import utils.attack_utility as attack_utility
import pickle as pkl
import os

import utils.experiment_logger as experiment_logger

# %% [markdown]
# The code in the next section attempts to explore the gradients observed at each step of the optimization problem.

# %%
def set_of_points(model, tokenizer, grad_optim, topk, factor_to_sample):
    assert len(grad_optim.shape) == 1, "Only does one-by-one"
    topk_indices = torch.topk(grad_optim, topk).indices
    other_sample_set = set(list(range(len(tokenizer)))).difference(set(topk_indices.tolist()))
    num_others_to_sample = factor_to_sample * topk
    other_indices = random.sample(list(other_sample_set), num_others_to_sample)
    return topk_indices, torch.tensor(other_indices)

def evaluate_ranking(
    model,
    tokenizer,
    current_best_tokens: torch.tensor,
    masks_data,
    grad_optims,
    idx_to_change,
    topk,
    factor_to_sample,
    num_trials
):
    target_tokens = current_best_tokens[masks_data["target_mask"]]
    list_rankings = []
    for i in range(num_trials):
        topk_indices, other_indices = set_of_points(model, tokenizer, grad_optims[idx_to_change], topk, factor_to_sample)
        scattered_tokens = current_best_tokens.expand(topk_indices.shape[0] + other_indices.shape[0], -1).clone()
        scattered_tokens[:, masks_data["optim_mask"][idx_to_change]] = torch.cat((topk_indices, other_indices))
        true_logprobs = attack_utility.target_logprobs(model, tokenizer, scattered_tokens, masks_data, target_tokens)
        ranking_observed = torch.argsort(true_logprobs)
        list_rankings.append(ranking_observed)
    return list_rankings

def plot_rankings(list_of_rankings, topk):
    total_length = len(list_of_rankings[0])
    final_list = [0] * total_length
    for ranking in list_of_rankings:
        for k_val in range(topk):
            final_list[ranking.index(k_val)] = final_list[ranking.index(k_val)] + 1
    plt.figure()
    plt.bar(list(range(total_length)), final_list)
    plt.close()

def get_rankings_distribution_across_run(
    model,
    tokenizer,
    logger,
    param_dict,
    trace_id,
    optimization_steps_to_check_at,
    positions_to_check_at,
    scale_factor,
    num_trials
):
    topk = param_dict["custom_gcg_hyperparams"]["topk"]
    masks_data = param_dict["input_tokenized_data"]["masks"]
    best_tokens_iterator = logger.query({"variable_name": "current_best_tokens", "trace_id": trace_id})
    _ = next(best_tokens_iterator)
    grad_optims_iterator = logger.query({"variable_name": "grad_optims", "trace_id": trace_id})

    rankings_dict = {}
    for step_idx, (grad_optims, current_best_tokens) in enumerate(zip(grad_optims_iterator, best_tokens_iterator)):
        if step_idx in optimization_steps_to_check_at:
            for position_idx in positions_to_check_at:
                list_of_rankings = [rank.tolist() for rank in evaluate_ranking(model, tokenizer, current_best_tokens, masks_data, grad_optims, position_idx, topk, scale_factor, num_trials)]
                rankings_dict[(position_idx, step_idx)] = list_of_rankings
    return rankings_dict


# %%
def gradient_rankings_run(model, tokenizer, log_folder, steps_fraction = 10, num_positions = 1, scale_factor = 4, num_samples = 50):
    result_df = experiment_logger.load_experiment_logs(f"{log_folder}/metadata.jsonl")
    params_list, trace_ids_list = experiment_logger.params_and_trace_ids_by_function(log_folder, result_df, "custom_gcg")
    logger = experiment_logger.ExperimentLogger(log_folder)
    for param_dict, trace_id in zip(params_list, trace_ids_list):
        num_steps = param_dict["custom_gcg_hyperparams"]["max_steps"]
        steps_to_check_at = list(range(0, num_steps, int(num_steps / steps_fraction)))
        positions_to_check_at = random.sample(list(range(len(param_dict["input_tokenized_data"]["masks"]["optim_mask"]))), num_positions)
        rankings_dict = get_rankings_distribution_across_run(model, tokenizer, logger, param_dict, trace_id, steps_to_check_at, positions_to_check_at, scale_factor, num_samples)

        if os.path.exists(f"{log_folder}/gradients_rankings.pkl"):
            with open(f"{log_folder}/gradients_rankings.pkl", "rb") as gradient_ranks_pkl:
                old_result_list = pkl.load(gradient_ranks_pkl)
        else:
            old_result_list = []
        
        old_result_list.append(rankings_dict)
        with open(f"{log_folder}/gradients_rankings.pkl", "wb") as gradient_ranks_pkl:
            pkl.dump(old_result_list, gradient_ranks_pkl)

MODEL_PATH = "/data/models/hf/Meta-Llama-3-8B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda:0")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
LOG_FOLDER = "logs/runs19/run_20250128120450222530"
gradient_rankings_run(model, tokenizer, LOG_FOLDER)



