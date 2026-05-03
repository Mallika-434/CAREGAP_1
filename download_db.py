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
