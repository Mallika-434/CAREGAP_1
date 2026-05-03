import os
from huggingface_hub import hf_hub_download

db_path = "db_demo.sqlite3"
if not os.path.exists(db_path) or os.path.getsize(db_path) < 10 * 1024 * 1024:
    print("Downloading db_demo.sqlite3 from HF Hub...")
    hf_hub_download(
        repo_id="mc18102001/CareGap",
        filename="db_demo.sqlite3",
        repo_type="space",
        local_dir=".",
        token=os.environ.get("HF_TOKEN", ""),
    )
    print("Download complete.")
else:
    print("db_demo.sqlite3 already present, skipping download.")
