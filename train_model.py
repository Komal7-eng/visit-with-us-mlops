import os
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi, login
from dotenv import load_dotenv

load_dotenv("E:\\GL\\clean-repo\\.env", override=True)
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

api = HfApi()
# Repo IDs
repo_id = "kaushalya7/visit-with-us-mlops" 
model_repo_id = "kaushalya7/mlops-visit-with-us-model"


mlflow.set_experiment("Visit_With_Us_Experiment")

# Load data
train_path = hf_hub_download(repo_id=repo_id, filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=repo_id, filename="test.csv", repo_type="dataset")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --- NEW: Feature Selection to match Streamlit App ---
selected_features = ['Age', 'MonthlyIncome', 'Passport', 'NumberOfTrips', 'PitchSatisfactionScore', 'Designation']
target = 'ProdTaken'
train_df = train_df[selected_features + [target]]
test_df = test_df[selected_features + [target]]
# -----------------------------------------------------

# Preprocessing
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

X_train = train_df.drop(target, axis=1)
y_train = train_df[target]
X_test = test_df.drop(target, axis=1)
y_test = test_df[target]

# MLflow Training Block
with mlflow.start_run():
    # class_weight='balanced' model ko Likely (1) classes par zyada dhyan dene ko bolega
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))
    
    
    # Log to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    precision = precision_score(y_test, best_model.predict(X_test))
    recall = recall_score(y_test, best_model.predict(X_test))
    f1 = f1_score(y_test, best_model.predict(X_test))
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
        
    # Save metrics locally
    with open("metrics.json", "w") as f:
        json.dump({"best_params": grid_search.best_params_, "accuracy": acc}, f)

# Save and Upload
joblib.dump(best_model, "model.joblib")
api.upload_file(path_or_fileobj="model.joblib", path_in_repo="model.joblib", repo_id=model_repo_id, repo_type="model")
api.upload_file(path_or_fileobj="metrics.json", path_in_repo="metrics.json", repo_id=model_repo_id, repo_type="model")

print("✅ Training Complete with Aligned Features!")
