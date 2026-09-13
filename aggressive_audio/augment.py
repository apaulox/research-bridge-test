"""Label-preserving channel augmentation. Generation is a dataset operation, not a filter."""
from dataclasses import dataclass
import subprocess

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AugmentConfig:
    clean_probability: float = 0.25
    max_transforms: int = 3
    transforms: tuple = ("telephone", "codec", "replay_sim", "noise", "resample", "gain", "enhance")
    light: bool = False


def fit_length(x, length):
    return np.pad(x[:length], (0, max(0, length - len(x)))).astype(np.float32)


def codec_roundtrip(x, sr, codec):
    # Conventional codecs only. The output retains the source authenticity label.
    enc, container, bitrate = {"mp3": ("libmp3lame", "mp3", "48k"),
                               "opus": ("libopus", "ogg", "24k")}[codec]
    encoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(sr),
         "-ac", "1", "-i", "pipe:0", "-c:a", enc, "-b:a", bitrate, "-f", container, "pipe:1"],
        input=np.asarray(x, dtype="<f4").tobytes(), capture_output=True, timeout=20, check=True).stdout
    decoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le",
         "-ar", str(sr), "-ac", "1", "pipe:1"],
        input=encoded, capture_output=True, timeout=20, check=True).stdout
    y = np.frombuffer(decoded, dtype="<f4").copy()
    if not len(y):
        raise ValueError("Codec returned empty audio")
    return fit_length(y, len(x))


def transform(x, name, rng, sr=16000, light=False):
    x = np.asarray(x, dtype=np.float32)
    info = {"name": name, "label_policy": "preserve_source"}
    if name == "gain":
        db = float(rng.uniform(-6, 3) if light else rng.uniform(-12, 6))
        y = x * 10 ** (db / 20)
        info["db"] = db
    elif name == "noise":
        snr = float(rng.uniform(20, 30) if light else rng.uniform(8, 30))
        noise = rng.normal(size=len(x))
        if rng.random() < 0.5:
            noise = signal.lfilter([1], [1, -0.85], noise)
            info["color"] = "lowpass"
        else:
            info["color"] = "white"
        noise *= np.sqrt(np.mean(x.astype(float)**2) + 1e-12) / (np.std(noise) + 1e-12) / 10**(snr/20)
        y = x + noise
        info["snr_db"] = snr
    elif name == "telephone":
        # Narrowband + G.711-style mu-law quantization is a channel simulation, not TTS.
        y = signal.sosfilt(signal.butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos"), x)
        y = signal.resample_poly(y, 1, 2)
        y = np.clip(y, -1, 1)
        companded = np.sign(y) * np.log1p(255 * np.abs(y)) / np.log(256)
        quantized = np.round((companded + 1) * 127.5) / 127.5 - 1
        y = np.sign(quantized) * np.expm1(np.abs(quantized) * np.log(256)) / 255
        y = signal.resample_poly(y, 2, 1)
        info["simulation"] = "300-3400Hz / 8kHz / 8bit mu-law"
    elif name == "resample":
        target = 12000 if light else int(rng.choice([8000, 11025, 12000]))
        from math import gcd
        g = gcd(sr, target)
        y = signal.resample_poly(x, target // g, sr // g)
        y = signal.resample_poly(y, sr // g, target // g)
        info["intermediate_sr"] = target
    elif name == "codec":
        codec = str(rng.choice(["mp3", "opus"]))
        y = codec_roundtrip(x, sr, codec)
        info["codec"] = codec
    elif name == "replay_sim":
        # Synthetic impulse response; do not describe this as a physical replay recording.
        seconds = float(rng.uniform(0.08, 0.35))
        t = np.arange(int(sr * seconds)) / sr
        impulse = rng.normal(size=len(t)) * np.exp(-6.9 * t / seconds)
        impulse[0] = 0
        impulse *= float(rng.uniform(0.08, 0.25)) / (np.linalg.norm(impulse) + 1e-12)
        impulse[0] = 1
        y = signal.fftconvolve(x, impulse)[:len(x)]
        y = signal.sosfilt(signal.butter(2, [100, 7000], btype="bandpass", fs=sr, output="sos"), y)
        info["synthetic_rt60_seconds"] = seconds
    elif name == "enhance":
        # Conservative spectral attenuation; no generative/vocoder reconstruction.
        _, _, spectrum = signal.stft(x, fs=sr, nperseg=512, noverlap=384)
        magnitude = np.abs(spectrum)
        floor = np.quantile(magnitude, 0.10, axis=1, keepdims=True)
        mask = np.maximum(0.4, 1 - 0.6 * floor / (magnitude + 1e-8))
        _, y = signal.istft(spectrum * mask, fs=sr, nperseg=512, noverlap=384)
        info["algorithm"] = "non-generative spectral attenuation"
    else:
        raise ValueError(f"Unsupported augmentation: {name}")
    y = np.clip(fit_length(y, len(x)), -1, 1)
    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite augmented audio: {name}")
    return y, info


def augment(x, rng, config=AugmentConfig()):
    if rng.random() < config.clean_probability:
        return np.asarray(x, dtype=np.float32).copy(), [{"name": "clean"}]
    count = int(rng.integers(1, min(config.max_transforms, len(config.transforms)) + 1))
    names = rng.choice(config.transforms, size=count, replace=False)
    y = x
    trace = []
    for name in names:
        y, info = transform(y, str(name), rng, light=config.light)
        trace.append(info)
    return y, trace


def overlay_voice(music, voice, rng):
    """Overlay an independently labelled voice. MUSIC label must remain unchanged."""
    snr = float(rng.uniform(-12, 6))
    music = np.asarray(music, dtype=np.float32)
    voice = fit_length(voice, len(music))
    scale = np.sqrt(np.mean(music.astype(float)**2) + 1e-10) / np.sqrt(np.mean(voice.astype(float)**2) + 1e-10)
    mixed = music + voice * scale * 10**(snr/20)
    peak = max(1.0, float(np.max(np.abs(mixed))))
    return (mixed / peak).astype(np.float32), {"name": "voice_overlay", "voice_to_music_db": snr,
                                              "music_label_policy": "preserve_music_source",
                                              "voice_label_policy": "not_inferred_from_background"}
