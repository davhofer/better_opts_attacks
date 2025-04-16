#!/bin/bash
# Simple RunPod setup script

# Exit on error
set -e

# Constants
WORKSPACE_DIR="./workspace"
VENV_PATH="${WORKSPACE_DIR}/venv"
REPO_PATH="${WORKSPACE_DIR}/repo"

# Fetch GitHub credentials from RunPod secrets
GITHUB_TOKEN=$(runpodctl get secret github_nishitvp_runpod_personal_token | jq -r '.value')
GITHUB_USERNAME=$(runpodctl get secret github_nishitvp_username | jq -r '.value')
HF_TOKEN=$(runpodctl get secret huggingface_nishitvp_access_token | jq -r '.value')
HF_USERNAME=$(runpodctl get secret huggingface_nishitvp_username | jq -r '.value')

# Clone the repository with submodules
git clone --recursive https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/better_opts_attacks.git ${REPO_PATH}

# Create and activate Python virtual environment
python3 -m venv ${VENV_PATH}
source ${VENV_PATH}/bin/activate

# Install required packages
pip3 install transformers accelerate peft pandas python-dotenv
pip3 install -r ${REPO_PATH}/requirements.txt
pip3 install -U "huggingface_hub[cli]"

huggingface-cli login --token $HF_TOKEN --add-to-git-credential
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ${REPO_PATH}/secalign_refactored/secalign_models/meta-llama
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --local-dir ${REPO_PATH}/secalign_refactored/secalign_models/mistralai


python3 ${REPO_PATH}/initial_setup.py

echo "Setup complete!"