"""Load author checkpoints with explicit label direction; train a probe or final layers."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_EAT = Path("/home/huskypaul/dacon/submissions/eat75_musicdet25_dependencyfix/model/eat_fmc")
DEFAULT_SPEECH = Path(__file__).resolve().parent / "assets/speech_xlsr1b"


def load_eat(path=DEFAULT_EAT):
    path = Path(path)
    sys.path.insert(0, str(path / "runtime"))
    from deepfense.models.frontends import eat  # noqa: F401
    from deepfense.models.backends import aasist  # noqa: F401
    from deepfense.models.losses import cross_entropy  # noqa: F401
    from deepfense.models.detector import ModularDetector
    model = ModularDetector({
        "frontend": {"type": "eat", "args": {"source": "huggingface", "ckpt_path": str(path / "eat_architecture"),
                                                "freeze": False, "trust_remote_code": False}},
        "backend": {"type": "AASIST", "args": {}},
        "loss": [{"type": "CrossEntropy", "embedding_dim": 160, "n_classes": 2, "weight": 1.0}]})
    model.load_state_dict(torch.load(path / "model_state.pt", map_location="cpu", weights_only=True), strict=True)
    return model


class SSLBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # The author-pinned 2022 fairseq uses NumPy aliases removed in 1.24.
        # Restore their identical builtin meanings without downgrading the shared NumPy.
        import numpy as np
        for name, value in {"float": float, "int": int, "bool": bool}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
        cfg = Wav2Vec2Config(
            quantize_targets=True, extractor_mode="layer_norm", layer_norm_first=True,
            final_dim=1024, latent_temp=(2.0, 0.1, 0.999995), encoder_layerdrop=0.0,
            dropout_input=0.0, dropout_features=0.0, dropout=0.0, attention_dropout=0.0,
            conv_bias=True, encoder_layers=48, encoder_embed_dim=1280,
            encoder_ffn_embed_dim=5120, encoder_attention_heads=16, feature_grad_mult=1.0)
        self.model = Wav2Vec2Model(cfg)

    def forward(self, x):
        return self.model(x, mask=False, features_only=True)["x"]


class AntiDeepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_ssl = SSLBackbone()
        self.adap_pool1d = nn.AdaptiveAvgPool1d(1)
        self.proj_fc = nn.Linear(1280, 2)

    def forward(self, x):
        # Normalize EACH fixed-length sample independently, following the author inference.
        x = F.layer_norm(x, (x.shape[-1],))
        features = self.m_ssl(x)
        return self.proj_fc(self.adap_pool1d(features.transpose(1, 2)).squeeze(-1))


def load_speech(path=DEFAULT_SPEECH):
    from safetensors.torch import load_file
    model = AntiDeepfake()
    model.load_state_dict(load_file(str(Path(path) / "model.safetensors")), strict=True)
    return model


def configure_training(model, branch, last_layers=0):
    for p in model.parameters():
        p.requires_grad_(False)
    modules = [model.backend, model.losses] if branch == "music" else [model.proj_fc]
    if last_layers:
        if branch == "music":
            blocks = model.frontend.model.model.blocks
        else:
            blocks = model.m_ssl.model.encoder.layers
        if last_layers > len(blocks):
            raise ValueError("Too many requested trainable layers")
        modules.extend(list(blocks[-last_layers:]))
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
    return modules


def train_mode(model, modules):
    # Frozen representation stays deterministic. Only selected modules train.
    model.eval()
    for module in modules:
        module.train()


def logits(model, waveform, branch):
    output = model(waveform)
    return output["logits"] if branch == "music" else output


def adaptation_state(model, modules):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    # Includes e.g. AASIST batch-normalization running statistics.
    for name, buffer in model.named_buffers():
        if any(any(buffer is b for b in module.buffers()) for module in modules):
            names.add(name)
    return {n: t.detach().cpu() for n, t in model.state_dict().items() if n in names}
