#!/usr/bin/env python3
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "davnozdubv/sage-ru-dictation-onnx"
MODEL_DIR = Path("/Users/davnozdu/Downloads/speechcorrector/models/sage_ru_corrector_int8")

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
print(f"Created/verified public HF repo: https://huggingface.co/{REPO_ID}", flush=True)

api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload sage-ru-dictation-onnx (INT8 encoder/decoder + FP32 lm_head)",
)
print(f"Uploaded all model files to https://huggingface.co/{REPO_ID}", flush=True)
