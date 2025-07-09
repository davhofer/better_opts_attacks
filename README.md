# _May I have your attention?_ Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks

This repository contains code to run the ASTRA attacks that break SecAlign and StruQ.

## Setup

### Clone Repository

This will clone the repository as well as all the submodules in the right location required.

```bash
git clone --recurse-submodules https://github.com/nishitvp/better_opts_attacks.git
```

### Set up Python environment

The codebase doesn't use any special Python packages beyond the standard collection of LLM and Deep Learning related packages. To install the requirements, run

```bash
python3 -m pip install -r requirements.txt
```

and this should install the necessary packages.

### Download Models

1. First, download the base models from HuggingFace (some of these may be gated, requiring you to gain access from HuggingFace)

    ```bash
    huggingface-cli login --token ${HF_PASSWORD}
    huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir secalign_refactored/secalign_models/meta-llama/Meta-Llama-3-8B-Instruct
    huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --local-dir secalign_refactored/secalign_models/mistralai/Mistral-7B-Instruct-v0.1
    ```

2. Download the SecAlign adapters and StruQ models. There's a helpful script to download and extract in the right place.

   ```bash
   python3 setup_scripts/download_models.py
   ```

    This should download and extract the correct models in the correct place. To verify, check whether `secalign_refactored/secalign_models` has three folders `huggyllama`, `meta-llama` and `mistralai` which should each contain the model(s) themselves.

## Running the attack

The key file used to run the experiments is `experiment.py`.
