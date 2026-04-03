import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi, login
from dotenv import load_dotenv


# LOAD The token from the .env file
load_dotenv()
token = os.getenv("HF_TOKEN")
login(token=token)

# Initialize the Hugging Face API to interact with the Hub
api = HfApi()

# Define the repository ID where the raw data is stored
repo_id = "kaushalya7/visit-with-us-mlops" 

# 1. Download raw data from HF
print("Downloading raw data from Hugging Face...")
file_path = hf_hub_download(repo_id=repo_id, filename="raw_data.csv", repo_type="dataset")
df = pd.read_csv(file_path)

# 2. Data Cleaning & Validation (Detailed)
print("\n--- Step 2: Data Cleaning ---")

# A. Initial Shape
print(f"Original Dataset Shape: {df.shape}")

# B. Check for Duplicates
duplicate_count = df.duplicated().sum()
print(f"Duplicates found: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# C. Check for Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Filling or dropping based on objective (Dropping here as per basic rubric)
df_cleaned = df.dropna()
print(f"Shape after dropping missing values: {df_cleaned.shape}")

# D. Drop Unnecessary Columns
# CustomerID is just an index, not a feature for prediction
if 'CustomerID' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(['CustomerID'], axis=1)
    print("Column 'CustomerID' dropped.")

# 3. Split the cleaned dataset into training and testing sets, and save them locally
print("\n--- Step 3: Splitting Data ---")
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42, stratify=df_cleaned['ProdTaken'])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

trtrain_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

folder_path = "data"
# 4. Upload the resulting train and test datasets back to the Hugging Face data space
print("\n--- Step 4: Uploading to HF ---")

folder_path = "C:\\visit_with_us_MLOps\\data"

for file_name in ["train.csv", "test.csv"]:
    file_path = os.path.join(folder_path, file_name)

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset"
    )

    print(f"Successfully uploaded {file_name} to HF!")
