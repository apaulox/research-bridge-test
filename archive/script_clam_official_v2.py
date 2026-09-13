#!/usr/bin/env python3
"""Official pretrained MoM-CLAM batch-16 head with baseline voice and presence."""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

# 추론에는 model 폴더에 포함된 로컬 파일만 사용한다.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"
CLAM_DIR = MODEL_DIR / "clam"

DEFAULT_TEST_DIR = Path("data") / "test"
DEFAULT_SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "submission.csv"

# 오디오 처리 설정
AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5
CLAM_DURATION_SECONDS = 90
CLAM_BATCH_SIZE = 16
CLAM_SHUFFLE_SEED = 42
CLAM_MERT_SAMPLE_RATE = 24_000
CLAM_WAV2VEC_SAMPLE_RATE = 16_000
PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


# -----------------------------------------------------------------------------
# 1. 입력 파일 및 제출 양식 확인
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run official pretrained CLAM music with baseline voice and presence."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def select_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(device_name)


def find_audio_files(test_dir):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    audio_files = []
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(path)
    audio_files.sort(key=lambda path: path.stem)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")

    audio_ids = [path.stem for path in audio_files]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Audio IDs must be unique")
    return audio_files


def read_sample_submission(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)

    if column_names is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")

    required_columns = ["ID"] + PREDICTION_COLUMNS
    missing_columns = [name for name in required_columns if name not in column_names]
    if missing_columns:
        raise ValueError(f"Sample submission is missing columns: {missing_columns}")

    seen_ids = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Sample submission contains an empty ID")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate ID in sample submission: {audio_id}")
        seen_ids.add(audio_id)
        row["ID"] = audio_id

    return column_names, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in submission_rows]

    missing_ids = [audio_id for audio_id in submission_ids if audio_id not in audio_by_id]
    extra_ids = [audio_id for audio_id in audio_by_id if audio_id not in submission_ids]
    if missing_ids or extra_ids:
        raise ValueError(
            "Test audio and sample submission IDs do not match. "
            f"Missing: {missing_ids[:5]}, Extra: {extra_ids[:5]}"
        )

    return [audio_by_id[audio_id] for audio_id in submission_ids]


def load_audio(audio_path):
    audio, _ = librosa.load(
        audio_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {audio_path}")
    return audio


# -----------------------------------------------------------------------------
# 2. 오디오 구간 분할
# -----------------------------------------------------------------------------

def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]

    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)

    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# 3. PANNs를 이용한 음성·음악 존재 여부 추론
# -----------------------------------------------------------------------------

def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )

    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def predict_presence(model, voice_indices, music_indices, audio):
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_probability = float(predictions[:, voice_indices].max())
    music_probability = float(predictions[:, music_indices].max())
    return voice_probability, music_probability


def predict_presence_for_all_files(audio_files, device):
    model, voice_indices, music_indices = load_panns_model(device)
    presence_scores = {}

    for audio_path in tqdm(audio_files, desc="Presence"):
        audio = load_audio(audio_path)
        presence_scores[audio_path.stem] = predict_presence(
            model, voice_indices, music_indices, audio
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return presence_scores


# -----------------------------------------------------------------------------
# 4. HTDemucs를 이용한 음성·음악 분리
# -----------------------------------------------------------------------------

def load_htdemucs_model():
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        # PyTorch 2.6부터 바뀐 기본값에 맞춰 기존 체크포인트를 불러온다.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def separate_voice_and_music(audio_path, model, device):
    waveform = load_track(
        audio_path, model.audio_channels, model.samplerate
    ).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )


# -----------------------------------------------------------------------------
# 5. DF-Arena 1B를 이용한 성분별 Fake 추론
# -----------------------------------------------------------------------------

def load_df_arena_model(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous_directory = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous_directory)

    model = model.to(device).eval()
    fake_label_index = int(model.config.label2id["spoof"])
    return model, fake_label_index


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def predict_fake(model, fake_label_index, audio, device):
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    segment_scores = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment_tensor = torch.from_numpy(segment).to(device)

        with torch.inference_mode():
            logits = model(input_values=segment_tensor)["logits"]
            probabilities = torch.softmax(logits.float(), dim=-1)
        segment_scores.append(float(probabilities[0, fake_label_index]))

    return max(segment_scores)


def stable_sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
class CLAM(nn.Module):
    """Public MoM-CLAM detector head, preserved from the official repository."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(13, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(13, 3, kernel_size=3, padding=1)
        self.crossAttention1 = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.crossAttention2 = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, mert_embedding, wav2vec_embedding):
        stream1 = F.relu(self.conv1(mert_embedding))
        stream2 = F.relu(self.conv2(wav2vec_embedding))
        stream1 = stream1[:, 0, :].unsqueeze(1)
        stream2 = stream2[:, 0, :].unsqueeze(1)
        stream1, _ = self.crossAttention1(stream1, stream1, stream1)
        stream2, _ = self.crossAttention2(stream2, stream2, stream2)
        stream1 = self.fc1(stream1.squeeze(1))
        stream2 = self.fc2(stream2.squeeze(1))
        return self.fc3(torch.cat((stream1, stream2), dim=1))


def load_clam_head(device):
    checkpoint_path = CLAM_DIR / "best_model_triplet_loss_margin_0.2.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CLAM checkpoint missing: {checkpoint_path}")
    model = CLAM()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_clam_backbone(path, device):
    required = (path / "config.json", path / "preprocessor_config.json")
    for model_file in required:
        if not model_file.is_file():
            raise FileNotFoundError(f"CLAM backbone file missing: {model_file}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    ).to(device).eval()
    return model, processor


def make_clam_waveform(audio_path, sample_rate):
    # Match the official extractors: native audio -> backbone sample rate,
    # channel mean, then first 90 seconds / trailing zero padding.
    waveform, original_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid CLAM audio: {audio_path}")
    if original_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(original_rate, sample_rate)(waveform)
    waveform = waveform.mean(0)
    target_samples = sample_rate * CLAM_DURATION_SECONDS
    if waveform.numel() >= target_samples:
        return waveform[:target_samples]
    return F.pad(waveform, (0, target_samples - waveform.numel()))


def extract_clam_embedding(model, processor, audio_path, device):
    sample_rate = int(processor.sampling_rate)
    waveform = make_clam_waveform(audio_path, sample_rate)
    inputs = processor(
        waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True)
    return torch.stack(output.hidden_states).squeeze().mean(1).cpu()


# -----------------------------------------------------------------------------
# 6. 파일 단위 점수 계산 및 제출 파일 저장
# -----------------------------------------------------------------------------

def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):
    voice_score = voice_present * voice_fake
    music_score = music_present * music_fake
    return max(voice_score, music_score)


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_clam_embedding_batches(
    clam_head, audio_files, mert_embeddings, wav2vec_embeddings, device
):
    # The released checkpoint was trained/evaluated using batch_size=16 and
    # shuffle=True. Its sequence-first attention sees batch peers as tokens.
    # Preserve those learned operations; changing batch_first requires a
    # separately trained model and would otherwise reproduce v1 singleton scores.
    # Use a local seeded permutation so baseline RNG and output ID order stay intact.
    ordered_files = sorted(audio_files, key=lambda path: path.stem)
    generator = torch.Generator(device="cpu").manual_seed(CLAM_SHUFFLE_SEED)
    permutation = torch.randperm(len(ordered_files), generator=generator).tolist()
    shuffled_files = [ordered_files[index] for index in permutation]
    music_scores = {}
    for start in tqdm(range(0, len(shuffled_files), CLAM_BATCH_SIZE), desc="CLAM head batch16"):
        batch = shuffled_files[start:start + CLAM_BATCH_SIZE]
        mert_batch = torch.stack([mert_embeddings[path.stem] for path in batch]).to(device)
        wav2vec_batch = torch.stack([wav2vec_embeddings[path.stem] for path in batch]).to(device)
        with torch.inference_mode():
            logits = clam_head(mert_batch, wav2vec_batch).view(-1)
            # Match the official float32 sigmoid; real=0 and fake=1.
            probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        if len(probabilities) != len(batch):
            raise RuntimeError("CLAM output batch size mismatch")
        for path, probability in zip(batch, probabilities):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid CLAM probability for {path.name}")
            music_scores[path.stem] = float(probability)
    # Keep the final partial batch, matching DataLoader(drop_last=False).
    return music_scores

def predict_music_scores_for_all_files(audio_files, device):
    # Run the two official backbones sequentially to limit GPU memory.
    mert_path = CLAM_DIR / "MERT-v1-95M"
    mert_model, mert_processor = load_clam_backbone(mert_path, device)
    if int(mert_processor.sampling_rate) != CLAM_MERT_SAMPLE_RATE:
        raise ValueError("Unexpected MERT sample rate")
    mert_embeddings = {}
    for audio_path in tqdm(audio_files, desc="CLAM MERT"):
        mert_embeddings[audio_path.stem] = extract_clam_embedding(
            mert_model, mert_processor, audio_path, device
        )
    del mert_model, mert_processor
    clear_cuda()

    wav2vec_path = CLAM_DIR / "wav2vec2-base-100k-gtzan-music-genres"
    wav2vec_model, wav2vec_processor = load_clam_backbone(wav2vec_path, device)
    if int(wav2vec_processor.sampling_rate) != CLAM_WAV2VEC_SAMPLE_RATE:
        raise ValueError("Unexpected Wav2Vec2 sample rate")
    wav2vec_embeddings = {}
    for audio_path in tqdm(audio_files, desc="CLAM W2V embeddings"):
        wav2vec_embeddings[audio_path.stem] = extract_clam_embedding(
            wav2vec_model, wav2vec_processor, audio_path, device
        )
    del wav2vec_model, wav2vec_processor
    clear_cuda()
    clam_head = load_clam_head(device)
    music_scores = predict_clam_embedding_batches(
        clam_head, audio_files, mert_embeddings, wav2vec_embeddings, device
    )
    del clam_head, mert_embeddings, wav2vec_embeddings
    clear_cuda()
    return music_scores


def predict_fake_scores_for_all_files(
    audio_files, submission_rows, presence_scores, music_scores, device
):
    df_arena_model, fake_label_index = load_df_arena_model(device)
    htdemucs_model = load_htdemucs_model()

    for index, audio_path in enumerate(tqdm(audio_files, desc="Voice + fusion")):
        voice_audio, _ = separate_voice_and_music(
            audio_path, htdemucs_model, device
        )
        voice_fake = predict_fake(
            df_arena_model, fake_label_index, voice_audio, device
        )
        music_fake = music_scores[audio_path.stem]
        voice_present, music_present = presence_scores[audio_path.stem]
        file_fake = combine_file_fake_score(
            voice_fake, music_fake, voice_present, music_present
        )

        row = submission_rows[index]
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)

    return submission_rows


def save_submission(output_path, column_names, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    device = select_device(args.device)

    # 1. 테스트 파일을 제출 양식의 ID 순서에 맞춘다.
    audio_files = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files, submission_rows)

    # 2. 파일별 음성·음악 존재 확률을 계산한다.
    presence_scores = predict_presence_for_all_files(audio_files, device)

    # 3. 공식 CLAM으로 음악 Fake 확률을 계산한다.
    music_scores = predict_music_scores_for_all_files(audio_files, device)

    # 4. 보컬을 분리하고 baseline voice와 MAX fusion을 실행한다.
    submission_rows = predict_fake_scores_for_all_files(
        audio_files, submission_rows, presence_scores, music_scores, device
    )

    # 5. 5개 예측값을 제출 파일로 저장한다.
    save_submission(args.output, column_names, submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
