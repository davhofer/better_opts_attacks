#!/bin/bash
# Simple RunPod setup script

# Exit on error
set -e

# Constants
WORKSPACE_DIR="/workspace"
VENV_PATH="${WORKSPACE_DIR}/venv"

# Clone the repository with submodules
cd ${WORKSPACE_DIR}
git clone --recursive https://${GITHUB_USERNAME}:${GITHUB_PASSWORD}@github.com/${GITHUB_USERNAME}/better_opts_attacks.git

# Create and activate Python virtual environment
python3 -m venv ${VENV_PATH}
source ${VENV_PATH}/bin/activate

cd "${WORKSPACE_DIR}/better_opts_attacks"

# Install required packages
pip3 install -r requirements.txt

hf auth login --token ${HF_PASSWORD}
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir secalign_refactored/secalign_models/meta-llama/Meta-Llama-3-8B-Instruct
hf download mistralai/Mistral-7B-Instruct-v0.1 --local-dir secalign_refactored/secalign_models/mistralai/Mistral-7B-Instruct-v0.1

python3 setup_scripts/download_models.py

echo "Setup complete!"
