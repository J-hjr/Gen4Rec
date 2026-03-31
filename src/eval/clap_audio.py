from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from src.embed.embed_music4all import Config as FinetunedConfig
from src.embed.embed_music4all import load_finetuned_model_and_attention
from src.embed.embed_music4all_zeroshot import Config as ZeroShotConfig
from src.embed.embed_music4all_zeroshot import load_zeroshot_clap


EncoderBundle = tuple[torch.nn.Module, torch.nn.Module | None, int, dict[str, Any]]


@lru_cache(maxsize=3)
def load_audio_encoder(encoder: str = "auto") -> EncoderBundle:
    encoder_mode = encoder
    if encoder_mode == "auto":
        encoder_mode = "finetuned" if Path(FinetunedConfig.FINETUNED_CKPT).exists() else "zeroshot"

    if encoder_mode == "finetuned":
        model, attention_pool, embedding_dim = load_finetuned_model_and_attention(FinetunedConfig.DEVICE)
        return model, attention_pool, int(embedding_dim), {
            "encoder_name": "finetuned",
            "device": FinetunedConfig.DEVICE,
            "sample_rate": FinetunedConfig.SAMPLE_RATE,
            "num_chunks": FinetunedConfig.NUM_CHUNKS,
            "chunk_samples": FinetunedConfig.CHUNK_SAMPLES,
        }

    if encoder_mode == "zeroshot":
        model = load_zeroshot_clap(ZeroShotConfig.DEVICE)
        with torch.no_grad():
            dummy_wave = torch.zeros(1, ZeroShotConfig.CHUNK_SAMPLES).to(ZeroShotConfig.DEVICE)
            dummy_out = model.audio_branch({"waveform": dummy_wave})
            if isinstance(dummy_out, dict):
                dummy_emb = dummy_out.get("embedding", list(dummy_out.values())[0])
            else:
                dummy_emb = dummy_out
            if hasattr(model, "audio_projection"):
                dummy_emb = model.audio_projection(dummy_emb)
            embedding_dim = int(dummy_emb.shape[-1])
        return model, None, embedding_dim, {
            "encoder_name": "zeroshot",
            "device": ZeroShotConfig.DEVICE,
            "sample_rate": ZeroShotConfig.SAMPLE_RATE,
            "num_chunks": ZeroShotConfig.NUM_CHUNKS,
            "chunk_samples": ZeroShotConfig.CHUNK_SAMPLES,
        }

    raise ValueError(f"Unsupported encoder mode: {encoder}")


def prepare_audio_chunks(path: str | Path, sample_rate: int, num_chunks: int, chunk_samples: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    total_len = waveform.shape[0]
    chunks = []
    if total_len <= chunk_samples:
        pad_len = chunk_samples - total_len
        padded = F.pad(waveform, (0, pad_len))
        for _ in range(num_chunks):
            chunks.append(padded)
    else:
        max_start = total_len - chunk_samples
        start_points = np.linspace(0, max_start, num_chunks).astype(int)
        for start in start_points:
            chunks.append(waveform[start : start + chunk_samples])
    return torch.stack(chunks)


def embed_audio_file(
    path: str | Path,
    *,
    model: torch.nn.Module,
    attention_pool: torch.nn.Module | None,
    embedding_dim: int,
    device: str,
    sample_rate: int,
    num_chunks: int,
    chunk_samples: int,
) -> np.ndarray:
    chunks = prepare_audio_chunks(
        path=path,
        sample_rate=sample_rate,
        num_chunks=num_chunks,
        chunk_samples=chunk_samples,
    ).unsqueeze(0)
    _, observed_num_chunks, samples = chunks.shape

    with torch.no_grad():
        flat_audio = chunks.view(observed_num_chunks, samples).to(device)
        output_dict = model.audio_branch({"waveform": flat_audio})
        if isinstance(output_dict, dict):
            flat_audio_features = output_dict.get("embedding", list(output_dict.values())[0])
        else:
            flat_audio_features = output_dict

        if hasattr(model, "audio_projection"):
            flat_audio_features = model.audio_projection(flat_audio_features)

        unflattened = flat_audio_features.view(1, observed_num_chunks, embedding_dim)
        if attention_pool is None:
            pooled = unflattened.mean(dim=1)
        else:
            pooled = attention_pool(unflattened)
        pooled = F.normalize(pooled, dim=-1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def embed_audio_paths(paths: list[str | Path], *, encoder: str = "auto") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model, attention_pool, embedding_dim, encoder_cfg = load_audio_encoder(encoder)
    embeddings: dict[str, np.ndarray] = {}
    for path in paths:
        embeddings[str(path)] = embed_audio_file(
            path=path,
            model=model,
            attention_pool=attention_pool,
            embedding_dim=embedding_dim,
            device=str(encoder_cfg["device"]),
            sample_rate=int(encoder_cfg["sample_rate"]),
            num_chunks=int(encoder_cfg["num_chunks"]),
            chunk_samples=int(encoder_cfg["chunk_samples"]),
        )
    return embeddings, encoder_cfg
