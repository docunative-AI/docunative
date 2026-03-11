import os
from huggingface_hub import hf_hub_download

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# The models we need to download
MODELS = [
    {
        "repo_id": "CohereLabs/tiny-aya-global-GGUF",
        "filename": "tiny-aya-global-q4_k_m.gguf"
    },
    {
        "repo_id": "CohereLabs/tiny-aya-earth-GGUF",
        "filename": "tiny-aya-earth-q4_k_m.gguf"
    }
]

def main():
    print("Starting download of Tiny Aya GGUF models...\n")
    print("NOTE: You must have agreed to the model terms on HuggingFace!")
    print("Make sure you are logged in via `huggingface-cli login` or have HF_TOKEN set.\n")

    for model in MODELS:
        print(f"Downloading {model['filename']}...")
        try:
            # local_dir_use_symlinks=False forces the actual file to be saved in our folder 
            # instead of being hidden deep in a hidden cache folder.
            file_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                local_dir="models"
            )
            print(f"Successfully downloaded to: {file_path}\n")
        except Exception as e:
            print(f"Failed to download {model['filename']}. Error: {e}\n")
            print("Did you run `huggingface-cli login` first?")

    print("All downloads complete! You are ready to run llama-server.")

if __name__ == "__main__":
    main()