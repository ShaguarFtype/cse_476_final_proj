from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_model():
    # Get token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not found. Model download may fail if authentication is required.")
    
    model_id = "meta-llama/Llama-3.2-3B"
    cache_dir = os.path.join(os.getcwd(), "models", "llama-3.2-3b-base")

    print(f"Downloading {model_id} to {cache_dir}...")
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=cache_dir,
        token=hf_token
    )
    print("Download complete!")

if __name__ == "__main__":
    download_model()
