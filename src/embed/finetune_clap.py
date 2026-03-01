import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import laion_clap
import numpy as np
from transformers import RobertaTokenizer
from tqdm import tqdm
import random

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

    CKPT_URL = (
        "https://huggingface.co/lukewys/laion_clap/resolve/main/"
        "music_audioset_epoch_15_esc_90.14.pt?download=true"
    )
    CKPT_FILENAME = os.environ.get("GEN4REC_CLAP_BASE_CKPT_NAME", "music_audioset_epoch_15_esc_90.14.pt")
    CKPT_PATH = resolve_path("GEN4REC_CLAP_BASE_CKPT_PATH", Path(WEIGHTS_DIR) / CKPT_FILENAME)

    AUDIO_DIR = os.path.join(DATASET_PATH, "audios")
    SAMPLE_RATE = 48000

    CHUNK_DURATION = 10.0  # seconds
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

    # ✅ Corrected: 3 chunks of 10 seconds each
    NUM_CHUNKS = 3

    BATCH_SIZE = 32
    GRAD_ACCUM_STEPS = 1

    LEARNING_RATE = 1e-5
    EPOCHS = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    SEED = 42

    NUM_WORKERS = 8

    # Semantic soft-target loss hyperparams (applied only to audio->text)
    SEMANTIC_SMOOTHING = 0.5        # lambda in [0, 1]
    SIMILARITY_TEMPERATURE = 10.0   # tau; larger = more peaked neighbor dist


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 2. UTILS & LAYERS
# ==========================================
def download_weights():
    print("Looking for weights at:", Config.CKPT_PATH)
    if os.path.exists(Config.CKPT_PATH):
        print(f"✅ Weights found at: {Config.CKPT_PATH}")
        return
    raise FileNotFoundError(
        f"❌ Weights not found at {Config.CKPT_PATH}. "
        f"Please download the checkpoint manually and place it there."
    )


class ContextAttention(nn.Module):
    """
    Learns to weigh chunks by importance rather than simple averaging.
    Input: (B, C, D)
    Output: (B, D)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attention_vector = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D]
        scores = self.attention_vector(x)      # [B, C, 1]
        weights = F.softmax(scores, dim=1)     # [B, C, 1]
        return (weights * x).sum(dim=1)        # [B, D]


# ==========================================
# 3. DATASET
# ==========================================
class Music4AllDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sample_rate=48000,
        num_chunks=3,
        chunk_samples=480000,
    ):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audios")
        self.sample_rate = sample_rate
        self.num_chunks = num_chunks
        self.chunk_samples = chunk_samples

        print(f"Loading metadata from {root_dir}...")
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

        self.meta_df = pd.merge(genres_df, tags_df, on="id", how="inner")
        self.valid_data = []

        print("Verifying audio files...")
        for _, row in self.meta_df.iterrows():
            song_id = row["id"]
            audio_path = os.path.join(self.audio_dir, f"{song_id}.mp3")
            if os.path.exists(audio_path):
                self.valid_data.append(
                    {
                        "path": audio_path,
                        "genres": str(row["genres"]),
                        "tags": str(row["tags"]),
                    }
                )
        print(f"Dataset loaded. Found {len(self.valid_data)} valid samples.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        text_prompt = f"genres: {item['genres']}, tags: {item['tags']}"

        try:
            waveform, sr = torchaudio.load(item["path"])
        except Exception:
            return (torch.zeros(self.num_chunks, self.chunk_samples), text_prompt)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # [T]
        total_len = waveform.shape[0]
        chunks = []

        if total_len <= self.chunk_samples:
            pad_len = self.chunk_samples - total_len
            padded = F.pad(waveform, (0, pad_len))
            for _ in range(self.num_chunks):
                chunks.append(padded)
        else:
            max_start = total_len - self.chunk_samples
            start_points = np.linspace(0, max_start, self.num_chunks).astype(int)
            for start in start_points:
                chunks.append(waveform[start : start + self.chunk_samples])

        return torch.stack(chunks), text_prompt  # [C, T], str


# ==========================================
# 4. ASYMMETRIC SOFT-TARGET LOSS (AUDIO->TEXT ONLY)
# ==========================================
class SemanticSoftClipLossA2TTextOnly(nn.Module):
    """
    Audio->Text: soft targets built from TEXT-TEXT similarity (genres/tags overlap).
    Text->Audio: standard one-hot CLIP/CLAP targets (no smoothing).
    """
    def __init__(self, semantic_smoothing: float = 0.5, similarity_temperature: float = 10.0):
        super().__init__()
        self.semantic_smoothing = semantic_smoothing  # lambda
        self.similarity_temperature = similarity_temperature  # tau

    @staticmethod
    def _one_hot_targets(B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        targets = torch.zeros((B, B), device=device, dtype=dtype)
        idx = torch.arange(B, device=device)
        targets[idx, idx] = 1.0
        return targets

    def _semantic_targets_from_textsim(self, text_sim: torch.Tensor) -> torch.Tensor:
        """
        Build y = (1-lambda)*I + lambda*softmax(tau*text_sim), using detached similarities.
        """
        B = text_sim.size(0)
        lam = self.semantic_smoothing
        tau = self.similarity_temperature

        I = self._one_hot_targets(B, text_sim.device, text_sim.dtype)
        if lam <= 0.0:
            return I

        with torch.no_grad():
            q = F.softmax((text_sim.detach() * tau), dim=-1)
            y = (1.0 - lam) * I + lam * q
        return y

    def forward(self, audio_features, text_features, logit_scale_a, logit_scale_t=None):
        # Normalize
        audio_features = F.normalize(audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Scales
        logit_scale_a = logit_scale_a.exp()
        logit_scale_t = logit_scale_t.exp() if logit_scale_t is not None else logit_scale_a

        # Cross-modal logits
        logits_a2t = logit_scale_a * (audio_features @ text_features.t())  # [B,B]
        logits_t2a = logit_scale_t * (text_features @ audio_features.t())  # [B,B]

        # Predicted distributions (log-probs)
        logp_a2t = F.log_softmax(logits_a2t, dim=-1)
        logp_t2a = F.log_softmax(logits_t2a, dim=-1)

        B = audio_features.size(0)

        # Targets:
        # ✅ audio->text softened by text-text similarity
        text_sim = text_features @ text_features.t()  # [B,B]
        y_a2t = self._semantic_targets_from_textsim(text_sim)

        # ✅ text->audio remains one-hot
        y_t2a = self._one_hot_targets(B, audio_features.device, audio_features.dtype)

        # Cross-entropy with soft/hard targets
        loss_a2t = -(y_a2t * logp_a2t).sum(dim=-1).mean()
        loss_t2a = -(y_t2a * logp_t2a).sum(dim=-1).mean()

        return 0.5 * (loss_a2t + loss_t2a)


# ==========================================
# 5. TRAIN / EVAL LOOPS
# ==========================================
def forward_batch(model, attention_pool, tokenizer, audio_chunks, texts, criterion, device):
    batch_size, num_chunks, samples = audio_chunks.shape

    flat_audio = audio_chunks.view(batch_size * num_chunks, samples).to(device)

    tokenized_text = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = tokenized_text["input_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)

    # Audio forward
    audio_dict = {"waveform": flat_audio}
    out = model.audio_branch(audio_dict)

    if isinstance(out, dict):
        flat_audio_features = out.get("embedding", list(out.values())[0])
    else:
        flat_audio_features = out

    if hasattr(model, "audio_projection"):
        flat_audio_features = model.audio_projection(flat_audio_features)

    D = flat_audio_features.shape[-1]
    chunk_features = flat_audio_features.view(batch_size, num_chunks, D)
    audio_features = attention_pool(chunk_features)  # [B,D]

    # Text forward
    text_out = model.text_branch(input_ids=input_ids, attention_mask=attention_mask)
    text_features = text_out[1]

    # ✅ Apply text projection if present
    if hasattr(model, "text_projection"):
        text_features = model.text_projection(text_features)

    logit_scale_a = model.logit_scale_a if hasattr(model, "logit_scale_a") else model.logit_scale
    logit_scale_t = model.logit_scale_t if hasattr(model, "logit_scale_t") else model.logit_scale

    return criterion(audio_features, text_features, logit_scale_a, logit_scale_t)


def train_one_epoch(model, attention_pool, tokenizer, dataloader, optimizer, criterion, device, epoch_idx):
    model.train()
    attention_pool.train()

    total_loss = 0.0
    num_steps = 0

    progress = tqdm(dataloader, desc=f"Train Epoch {epoch_idx}")
    optimizer.zero_grad()

    for step, (audio_chunks, texts) in enumerate(progress):
        loss = forward_batch(model, attention_pool, tokenizer, audio_chunks, texts, criterion, device)

        loss = loss / Config.GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * Config.GRAD_ACCUM_STEPS
        num_steps += 1
        progress.set_postfix(loss=(total_loss / num_steps))

    return total_loss / max(num_steps, 1)


def evaluate(model, attention_pool, tokenizer, dataloader, criterion, device, desc="Val"):
    model.eval()
    attention_pool.eval()

    total_loss = 0.0
    num_steps = 0

    with torch.no_grad():
        for audio_chunks, texts in tqdm(dataloader, desc=desc):
            loss = forward_batch(model, attention_pool, tokenizer, audio_chunks, texts, criterion, device)
            total_loss += loss.item()
            num_steps += 1

    return total_loss / max(num_steps, 1)


# ==========================================
# 6. MAIN
# ==========================================
def main():
    set_seed(Config.SEED)
    print(f"Initializing CLAP Finetuning on {Config.DEVICE}...")

    download_weights()

    full_dataset = Music4AllDataset(
        Config.DATASET_PATH,
        sample_rate=Config.SAMPLE_RATE,
        num_chunks=Config.NUM_CHUNKS,
        chunk_samples=Config.CHUNK_SAMPLES,
    )

    n = len(full_dataset)
    train_size = int(Config.TRAIN_SPLIT * n)
    val_size = int(Config.VAL_SPLIT * n)
    test_size = n - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )

    print(f"Dataset split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    print(f"Loading CLAP model from {Config.CKPT_PATH}...")
    wrapper = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    wrapper.load_ckpt(Config.CKPT_PATH)
    model = wrapper.model.to(Config.DEVICE)

    # Infer embedding dim
    with torch.no_grad():
        dummy = torch.zeros(1, Config.CHUNK_SAMPLES).to(Config.DEVICE)
        dummy_out = model.audio_branch({"waveform": dummy})
        dummy_emb = dummy_out.get("embedding", list(dummy_out.values())[0]) if isinstance(dummy_out, dict) else dummy_out
        if hasattr(model, "audio_projection"):
            dummy_emb = model.audio_projection(dummy_emb)
        D = dummy_emb.shape[-1]

    attention_pool = ContextAttention(dim=D).to(Config.DEVICE)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(attention_pool.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=0.01,
    )

    criterion = SemanticSoftClipLossA2TTextOnly(
        semantic_smoothing=Config.SEMANTIC_SMOOTHING,
        similarity_temperature=Config.SIMILARITY_TEMPERATURE,
    )

    print("🚀 Training with ATTENTION POOLING + A2T TEXT-ONLY SOFT TARGETS...")

    best_val = float("inf")
    best_ckpt = None
    os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)

    for epoch in range(1, Config.EPOCHS + 1):
        tr = train_one_epoch(model, attention_pool, tokenizer, train_loader, optimizer, criterion, Config.DEVICE, epoch)
        va = evaluate(model, attention_pool, tokenizer, val_loader, criterion, Config.DEVICE, desc=f"Val Epoch {epoch}")
        print(f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {tr:.4f}, Val Loss: {va:.4f}")

        ckpt_path = os.path.join(Config.WEIGHTS_DIR, f"clap_finetuned_epoch_{epoch}.pt")
        state = {"model": model.state_dict(), "attention_head": attention_pool.state_dict(),
                 "epoch": epoch, "train_loss": tr, "val_loss": va}
        torch.save(state, ckpt_path)
        print(f"💾 Saved checkpoint to {ckpt_path}")

        if va < best_val:
            best_val = va
            best_ckpt = os.path.join(Config.WEIGHTS_DIR, "clap_finetuned_best.pt")
            torch.save(state, best_ckpt)
            print(f"⭐ New best model saved to {best_ckpt}")

    if best_ckpt is not None:
        print(f"Loading best model from {best_ckpt} for test evaluation...")
        ckpt = torch.load(best_ckpt, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model"])
        attention_pool.load_state_dict(ckpt["attention_head"])

    te = evaluate(model, attention_pool, tokenizer, test_loader, criterion, Config.DEVICE, desc="Test")
    print(f"🧪 Final Test Loss: {te:.4f}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
