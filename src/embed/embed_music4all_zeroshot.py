import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import laion_clap
from tqdm import tqdm

# Try to import faiss; if not present, we can skip index building
try:
    import faiss
except ImportError:
    faiss = None
    print("⚠ faiss is not installed. Zero-shot FAISS index will be skipped.")


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent


def resolve_path(env_var: str, default_path: Path) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))
    return str(default_path.resolve())


class Config:
    BASE_DIR = str(BASE_DIR)
    REPO_ROOT = str(REPO_ROOT)
    DATASET_PATH = resolve_path("GEN4REC_DATASET_PATH", REPO_ROOT / "music4all")
    WEIGHTS_DIR = resolve_path("GEN4REC_WEIGHTS_DIR", REPO_ROOT / "weights" / "clap")

    # Base CLAP checkpoint (same as you used for finetuning)
    CKPT_FILENAME = os.environ.get("GEN4REC_CLAP_BASE_CKPT_NAME", "music_audioset_epoch_15_esc_90.14.pt")
    CKPT_PATH = resolve_path("GEN4REC_CLAP_BASE_CKPT_PATH", Path(WEIGHTS_DIR) / CKPT_FILENAME)

    AUDIO_DIR = os.path.join(DATASET_PATH, "audios")
    SAMPLE_RATE = 48000

    CHUNK_DURATION = 10.0  # seconds
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
    NUM_CHUNKS = 4         # keep same as finetune for fair comparison

    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # adjust for your cluster


# ==========================================
# 2. DATASET FOR EMBEDDING (ZERO-SHOT)
# ==========================================

class Music4AllEmbedDataset(Dataset):
    """
    Same idea as your previous embed dataset, but agnostic to finetuning.
    Returns: (song_id, audio_chunks)
    """
    def __init__(
        self,
        root_dir,
        sample_rate=48000,
        num_chunks=4,
        chunk_samples=480000,
    ):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audios")
        self.sample_rate = sample_rate
        self.num_chunks = num_chunks
        self.chunk_samples = chunk_samples

        print(f"Loading metadata from {root_dir}...")
        try:
            genres_df = pd.read_csv(
                os.path.join(root_dir, "id_genres.csv"),
                sep="\t",
                header=None,
                names=["id", "genres"],
            )
            tags_df = pd.read_csv(
                os.path.join(root_dir, "id_tags.csv"),
                sep="\t",
                header=None,
                names=["id", "tags"],
            )
        except FileNotFoundError:
            print(f"❌ Error: CSV files not found in {root_dir}")
            raise

        meta_df = pd.merge(genres_df, tags_df, on="id", how="inner")
        self.valid_data = []

        print("Verifying audio files...")
        for _, row in meta_df.iterrows():
            song_id = row["id"]
            audio_path = os.path.join(self.audio_dir, f"{song_id}.mp3")
            if os.path.exists(audio_path):
                self.valid_data.append(
                    {
                        "id": str(song_id),
                        "path": audio_path,
                    }
                )
        print(f"Dataset loaded. Found {len(self.valid_data)} valid samples.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        song_id = item["id"]

        try:
            waveform, sr = torchaudio.load(item["path"])
        except Exception:
            # If file is broken, return silent audio
            chunks_tensor = torch.zeros(Config.NUM_CHUNKS, Config.CHUNK_SAMPLES)
            return song_id, chunks_tensor

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # [T]
        total_len = waveform.shape[0]
        chunks = []

        if total_len <= self.chunk_samples:
            # Pad short tracks then duplicate the same chunk NUM_CHUNKS times
            pad_len = self.chunk_samples - total_len
            padded = F.pad(waveform, (0, pad_len))
            for _ in range(self.num_chunks):
                chunks.append(padded)
        else:
            # Evenly sample NUM_CHUNKS chunks across the track
            max_start = total_len - self.chunk_samples
            start_points = np.linspace(0, max_start, self.num_chunks).astype(int)
            for start in start_points:
                chunk = waveform[start: start + self.chunk_samples]
                chunks.append(chunk)

        chunks_tensor = torch.stack(chunks)  # [num_chunks, chunk_samples]
        return song_id, chunks_tensor


def collate_fn(batch):
    """
    Custom collate because song_id is a string.
    batch: list of (song_id, chunks_tensor)
    """
    ids = [b[0] for b in batch]
    audio_tensors = torch.stack([b[1] for b in batch], dim=0)  # [B, C, T]
    return ids, audio_tensors


# ==========================================
# 3. LOAD ZERO-SHOT CLAP (NO FINETUNE)
# ==========================================

def load_zeroshot_clap(device: str):
    """
    Load original CLAP model with pretrained weights only.
    No finetuned checkpoint, no attention head.
    """
    if not os.path.exists(Config.CKPT_PATH):
        raise FileNotFoundError(
            f"Base CLAP weights not found at {Config.CKPT_PATH}. "
            f"Please place 'music_audioset_epoch_15_esc_90.14.pt' there."
        )

    print(f"Loading zero-shot CLAP from {Config.CKPT_PATH} ...")
    wrapper = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    wrapper.load_ckpt(Config.CKPT_PATH)
    model = wrapper.model.to(device)
    model.eval()
    return model


# ==========================================
# 4. EMBEDDING LOOP (MEAN POOLING BASELINE)
# ==========================================

def embed_music4all_zeroshot():
    device = Config.DEVICE
    print(f"Using device: {device}")

    model = load_zeroshot_clap(device)

    dataset = Music4AllEmbedDataset(
        Config.DATASET_PATH,
        sample_rate=Config.SAMPLE_RATE,
        num_chunks=Config.NUM_CHUNKS,
        chunk_samples=Config.CHUNK_SAMPLES,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    all_ids = []
    all_embs = []

    # Discover embedding dimension dynamically
    with torch.no_grad():
        dummy_wave = torch.zeros(1, Config.CHUNK_SAMPLES).to(device)
        dummy_dict = {"waveform": dummy_wave}
        dummy_out = model.audio_branch(dummy_dict)
        if isinstance(dummy_out, dict):
            dummy_emb = dummy_out.get("embedding", list(dummy_out.values())[0])
        else:
            dummy_emb = dummy_out
        if hasattr(model, "audio_projection"):
            dummy_emb = model.audio_projection(dummy_emb)
        embedding_dim = dummy_emb.shape[-1]
    print(f"Embedding dimension (zero-shot): {embedding_dim}")

    with torch.no_grad():
        for song_ids, audio_chunks in tqdm(dataloader, desc="Embedding (zero-shot) tracks"):
            # audio_chunks: [B, C, T]
            batch_size, num_chunks, samples = audio_chunks.shape

            flat_audio = audio_chunks.view(batch_size * num_chunks, samples).to(device)
            audio_dict = {"waveform": flat_audio}

            output_dict = model.audio_branch(audio_dict)
            if isinstance(output_dict, dict):
                if "embedding" in output_dict:
                    flat_audio_features = output_dict["embedding"]
                else:
                    flat_audio_features = list(output_dict.values())[0]
            else:
                flat_audio_features = output_dict

            if hasattr(model, "audio_projection"):
                flat_audio_features = model.audio_projection(flat_audio_features)

            # [B * C, D] -> [B, C, D]
            unflattened = flat_audio_features.view(batch_size, num_chunks, embedding_dim)

            # *** Mean pooling over chunks (simple baseline) ***
            pooled = unflattened.mean(dim=1)  # [B, D]

            # L2-normalize (CLIP-style)
            pooled = F.normalize(pooled, dim=-1)

            # Move to CPU numpy
            pooled_np = pooled.cpu().numpy()
            all_embs.append(pooled_np)
            all_ids.extend(song_ids)

    all_embs = np.vstack(all_embs)         # [N_tracks, D]
    all_ids = np.array(all_ids, dtype=str)

    print(f"[Zero-shot] Embeddings shape: {all_embs.shape}")
    print(f"[Zero-shot] Num IDs: {all_ids.shape[0]}")

    os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
    emb_path = os.path.join(Config.WEIGHTS_DIR, "music4all_embeddings_zeroshot.npy")
    ids_path = os.path.join(Config.WEIGHTS_DIR, "music4all_ids.npy")

    np.save(emb_path, all_embs)
    np.save(ids_path, all_ids)

    print(f"💾 Saved ZERO-SHOT embeddings to {emb_path}")
    print(f"💾 Saved ids (shared) to {ids_path}")

    return all_ids, all_embs, embedding_dim


# ==========================================
# 5. OPTIONAL: BUILD FAISS INDEX
# ==========================================

def build_faiss_index_zeroshot(embeddings: np.ndarray, dim: int, index_path: str):
    if faiss is None:
        print("⚠️ faiss is not installed. Skipping zero-shot index creation.")
        print("    Install with `pip install faiss-gpu` (or `faiss-cpu`) if you want this.")
        return

    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ≈ cosine
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, index_path)
    print(f"📚 Saved ZERO-SHOT FAISS index to {index_path}")


def main():
    ids, embs, dim = embed_music4all_zeroshot()

    # OPTIONAL: create FAISS index for zero-shot baseline
    index_path = os.path.join(Config.WEIGHTS_DIR, "music4all_faiss_zeroshot.index")
    build_faiss_index_zeroshot(embs, dim, index_path)


if __name__ == "__main__":
    main()
