import os
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from huggingface_hub import hf_hub_download, HfApi, login
from dotenv import load_dotenv

# =========================
# 🔐 Load Token
# =========================
load_dotenv("C:\\visit_with_us_MLOps\\.env", override=True)
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

api = HfApi()

# =========================
# 📦 Repo IDs
# =========================
repo_id = "kaushalya7/visit-with-us-mlops"
model_repo_id = "kaushalya7/mlops-visit-with-us-model"

# =========================
# 📊 MLflow Setup
# =========================
mlflow.set_experiment("Visit_With_Us_Experiment")

# 🔥 Prevent MLflow crash (IMPORTANT)
mlflow.end_run()

# =========================
# 📥 Load Data
# =========================
train_path = hf_hub_download(repo_id=repo_id, filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=repo_id, filename="test.csv", repo_type="dataset")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# =========================
# 🎯 Feature Selection
# =========================
selected_features = [
    'Age', 'MonthlyIncome', 'Passport',
    'NumberOfTrips', 'PitchSatisfactionScore', 'Designation'
]
target = 'ProdTaken'

train_df = train_df[selected_features + [target]]
test_df = test_df[selected_features + [target]]

# =========================
# 🔄 Encoding
# =========================
cat_cols = train_df.select_dtypes(include=['object']).columns

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    encoders[col] = le

# =========================
# 📊 Split
# =========================
X_train = train_df.drop(target, axis=1)
y_train = train_df[target]

X_test = test_df.drop(target, axis=1)
y_test = test_df[target]

# =========================
# 🤖 Models
# =========================
models = {
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

param_grids = {
    "RandomForest": {
        'n_estimators': [50, 100],
        'max_depth': [5, 10]
    },
    "XGBoost": {
        'n_estimators': [50, 100],
        'max_depth': [3, 6]
    }
}

best_model = None
best_score = 0
best_params = {}

# =========================
# 🔥 Training Loop
# =========================
for model_name in models:
    grid = GridSearchCV(
        models[model_name],
        param_grids[model_name],
        cv=3,
        n_jobs=-1   # 🔥 faster
    )
    grid.fit(X_train, y_train)

    for params in grid.cv_results_["params"]:
        
        mlflow.end_run()  # 🔥 ensure clean run

        with mlflow.start_run():

            model = models[model_name].set_params(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            # 🔥 Logging
            mlflow.log_param("model", model_name)
            mlflow.log_params(params)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Save best model
            if acc > best_score:
                best_score = acc
                best_model = model
                best_params = params

# =========================
# 📊 Confusion Matrix
# =========================
disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.savefig("confusion_matrix.png")

# =========================
# 📊 Feature Importance
# =========================
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
    features = X_train.columns

    plt.figure(figsize=(8, 5))
    plt.barh(features, importance)
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")

# =========================
# 🔥 Log Artifacts Safely
# =========================
mlflow.end_run()
with mlflow.start_run(run_name="final_artifacts"):
    mlflow.log_artifact("confusion_matrix.png")
    if os.path.exists("feature_importance.png"):
        mlflow.log_artifact("feature_importance.png")

# =========================
# 💾 Save Metrics
# =========================
metrics = {
    "best_params": best_params,
    "accuracy": best_score
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# =========================
# 💾 Save Model
# =========================
joblib.dump(best_model, "model.joblib")

# =========================
# 🔥 Log Model
# =========================
mlflow.sklearn.log_model(best_model, "model")

# =========================
# ☁️ Upload to Hugging Face
# =========================
api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=model_repo_id,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="metrics.json",
    path_in_repo="metrics.json",
    repo_id=model_repo_id,
    repo_type="model"
)

print("🚀 Training + Tracking + Deployment Complete!")
