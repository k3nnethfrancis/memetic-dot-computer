import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT_DIR = Path(__file__).parents[2]

# make sure to login to huggingface before running this script, which requires a token from huggingface.co
# huggingface-cli login

# Specify the repository ID and the local directory where you want to save the model
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
LOCAL_DIR = f"{ROOT_DIR}/weights/{MODEL_NAME}"
snapshot_download(
    repo_id=MODEL_NAME,  
    local_dir=LOCAL_DIR
    )