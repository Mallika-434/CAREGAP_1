import os
from huggingface_hub import hf_hub_download

TOKEN = os.environ.get("HF_TOKEN", "")

def download_if_missing(filename, size_threshold_mb=10):
    if not os.path.exists(filename) or os.path.getsize(filename) < size_threshold_mb * 1024 * 1024:
        print(f"Downloading {filename} from HF Dataset repo...")
        hf_hub_download(
            repo_id="mc18102001/caregap-data",
            filename=filename,
            repo_type="dataset",
            local_dir=".",
            token=TOKEN,
        )
        print(f"{filename} download complete.")
    else:
        print(f"{filename} already present, skipping download.")

download_if_missing("db_demo.sqlite3", size_threshold_mb=10)
download_if_missing("synthea_california.duckdb", size_threshold_mb=5)

# Download model files
MODEL_FILES = [
    'lasso_logistic_regression.pkl',
    'random_forest.pkl',
    'xgboost.pkl',
    'htn_lasso.pkl',
    'htn_random_forest.pkl',
    'htn_gradient_boosting.pkl',
    'diabetes_lasso.pkl',
    'diabetes_random_forest.pkl',
    'diabetes_gradient_boosting.pkl',
    'scaler_htn.pkl',
    'scaler_t2d.pkl',
    'onset_features.json',
]

os.makedirs('models', exist_ok=True)
for model_file in MODEL_FILES:
    dest = f'models/{model_file}'
    if not os.path.exists(dest) or os.path.getsize(dest) < 100:
        print(f'Downloading {model_file}...', end=' ', flush=True)
        hf_hub_download(
            repo_id='mc18102001/caregap-data',
            filename=f'models/{model_file}',
            repo_type='dataset',
            local_dir='.',
            token=TOKEN
        )
        print('done')
    else:
        print(f'{model_file} already present, skipping.')
