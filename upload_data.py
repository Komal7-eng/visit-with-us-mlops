import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv


# LOAD The token from the .env file
load_dotenv()       
token = os.getenv("HF_TOKEN")

if not token:
    print("Error: .env file mein HF_TOKEN nahi mila!")
else:
    login(token=token)
    # Hugging Face Hub API initialization
    api = HfApi()

    
    repo_id = "kaushalya7/visit-with-us-mlops" 
    local_file = "C:\\visit_with_us_MLOps\\data\\tourism.csv" 

    
    # UPloading the file to Hugging Face Hub
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo="raw_data.csv",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Success! Link: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Detailed Error: {e}")
