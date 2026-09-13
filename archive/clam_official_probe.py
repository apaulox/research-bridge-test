#!/usr/bin/env python3
"""Smoke-test the official MoM-CLAM checkpoint on local audio files."""

import argparse
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class CLAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(13, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(13, 3, kernel_size=3, padding=1)
        self.crossAttention1 = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.crossAttention2 = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, embed1, embed2):
        embed1 = F.relu(self.conv1(embed1))
        embed2 = F.relu(self.conv2(embed2))
        stream1 = embed1[:, 0, :].unsqueeze(1)
        stream2 = embed2[:, 0, :].unsqueeze(1)
        stream1, _ = self.crossAttention1(stream1, stream1, stream1)
        stream2, _ = self.crossAttention2(stream2, stream2, stream2)
        stream1 = self.fc1(stream1.squeeze(1))
        stream2 = self.fc2(stream2.squeeze(1))
        return self.fc3(torch.cat((stream1, stream2), dim=1))


def load_wave(path: Path) -> np.ndarray:
    wave, _ = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
    return wave


def fixed_duration(wave: np.ndarray, sample_rate: int, duration: int = 90):
    tensor = torch.from_numpy(wave)
    if sample_rate != 16_000:
        tensor = torchaudio.functional.resample(tensor, 16_000, sample_rate)
    target = sample_rate * duration
    if tensor.numel() >= target:
        return tensor[:target]
    return F.pad(tensor, (0, target - tensor.numel()))


def extract_embedding(model, processor, wave, device, duration):
    sample_rate = int(processor.sampling_rate)
    fixed = fixed_duration(wave, sample_rate, duration)
    inputs = processor(
        fixed.numpy(), sampling_rate=sample_rate, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True)
    # Official extractor: stack 13 hidden states and average over time.
    return torch.stack(output.hidden_states).squeeze().mean(1).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--mert-dir", type=Path, required=True)
    parser.add_argument("--wav2vec-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--durations", default="90")
    args = parser.parse_args()
    durations = [int(value) for value in args.durations.split(",")]

    device = torch.device("cuda")
    files = sorted(
        path
        for path in args.audio_dir.iterdir()
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    )[: args.limit]
    if not files:
        raise FileNotFoundError(args.audio_dir)

    started = time.perf_counter()
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.mert_dir, local_files_only=True, trust_remote_code=True
    )
    mert = AutoModel.from_pretrained(
        args.mert_dir, local_files_only=True, trust_remote_code=True
    ).to(device).eval()
    mert_embeddings = {}
    for duration in durations:
        for path in files:
            wave = load_wave(path)
            tick = time.perf_counter()
            mert_embeddings[(path.stem, duration)] = extract_embedding(
                mert, mert_processor, wave, device, duration
            )
            print(
                f"MERT duration={duration} {path.name} "
                f"{time.perf_counter() - tick:.3f}s",
                flush=True,
            )
    del mert
    torch.cuda.empty_cache()

    wav_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.wav2vec_dir, local_files_only=True, trust_remote_code=True
    )
    wav2vec = AutoModel.from_pretrained(
        args.wav2vec_dir, local_files_only=True, trust_remote_code=True
    ).to(device).eval()
    clam = CLAM().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    clam.load_state_dict(state, strict=True)
    for duration in durations:
        for path in files:
            wave = load_wave(path)
            tick = time.perf_counter()
            wav_embedding = extract_embedding(
                wav2vec, wav_processor, wave, device, duration
            )
            with torch.inference_mode():
                logit = clam(
                    mert_embeddings[(path.stem, duration)].unsqueeze(0).to(device),
                    wav_embedding.unsqueeze(0).to(device),
                ).view(-1)[0]
                probability = torch.sigmoid(logit).item()
            print(
                f"CLAM duration={duration} {path.name} fake={probability:.8f} "
                f"logit={logit.item():.6f} "
                f"wav2vec+head={time.perf_counter() - tick:.3f}s",
                flush=True,
            )
    print(f"TOTAL files={len(files)} seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
