import os
from pathlib import Path
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = REPO_ROOT / "weights" / "clap"
DEFAULT_REPO_ID = "lukewys/laion_clap"
DEFAULT_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"


def download_clap_checkpoint():
    save_dir = Path(os.environ.get("GEN4REC_WEIGHTS_DIR", DEFAULT_SAVE_DIR)).expanduser().resolve()
    repo_id = os.environ.get("GEN4REC_CLAP_REPO_ID", DEFAULT_REPO_ID)
    filename = os.environ.get("GEN4REC_CLAP_BASE_CKPT_NAME", DEFAULT_FILENAME)

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{filename}' from '{repo_id}'...")
    print(f"Target directory: {save_dir}")

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(save_dir),
    )

    print("✅ Downloaded to:", file_path)


if __name__ == "__main__":
    download_clap_checkpoint()