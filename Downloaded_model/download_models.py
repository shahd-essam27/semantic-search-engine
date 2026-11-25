
from huggingface_hub import snapshot_download

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

for model_name in models:
    print(f"\nDownloading: {model_name}")

    location = snapshot_download(repo_id=model_name)

    print(f"Cached at: {location}")

print("\nAll model downloads completed!")
