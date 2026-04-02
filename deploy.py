import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

if token:
    login(token=token)
else:
    print("❌ HF_TOKEN not found!")

api = HfApi()
space_id = "kaushalya7/mlops-visit-with-us-v2"

# ✅ Ensure Space exists (prevents 404)
api.create_repo(
    repo_id=space_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

files_to_upload = ["app.py", "Dockerfile", "requirements-training.txt", "README.md"]

for file in files_to_upload:
    if os.path.exists(file):
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=space_id,
            repo_type="space"
        )
        print(f"✅ Uploaded {file}!")
    else:
        print(f"⚠️ {file} not found, skipping...")
