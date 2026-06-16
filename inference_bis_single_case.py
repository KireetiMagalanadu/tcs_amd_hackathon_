"""
BIS Model — Single-Case Inference
==================================
Loads a trained BISModel checkpoint and runs inference on one case.

INPUTS required:
  1. eeg          : np.ndarray  shape (3840, 2)   — 30 s of raw EEG at 128 Hz, both channels (µV)
  2. sef_sr       : np.ndarray  shape (300, 2)    — 5-min context of [SEF, SR] at 1 Hz
  3. missing_flags: np.ndarray  shape (2,)        — [sef_missing, sr_missing] binary flags
                                                    (1.0 if that column was entirely NaN for this case)

OUTPUTS (dict):
  - bis_t0   : float  — predicted BIS right now
  - bis_t30  : float  — predicted BIS in 30 s
  - bis_t60  : float  — predicted BIS in 60 s
  - bis_t180 : float  — predicted BIS in 3 min
  - bis_t300 : float  — predicted BIS in 5 min
  - embedding: np.ndarray (64,) — fusion embedding (for multimodal stack)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# ── CONFIG (must match training) ────────────────────────────────────────────
CONFIG: Dict = {
    "cnn_channels":  [2, 16, 32, 32],
    "cnn_kernels":   [64, 16, 8],
    "cnn_strides":   [4, 2, 2],
    "cnn_out_dim":   32,
    "mlp_in_dim":    6,
    "mlp_hidden":    32,
    "mlp_out_dim":   32,
    "fusion_in_dim": 64,
    "embedding_dim": 64,
    "checkpoint":    "ckpt_model_bis.pt",   # ← path to your saved checkpoint
    "eeg_clip_uv":   500.0,
}


# ── Model definition (copy from notebook) ───────────────────────────────────
class CNNBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config["cnn_channels"]
        kernels  = config["cnn_kernels"]
        strides  = config["cnn_strides"]
        layers: List[nn.Module] = []
        for i in range(len(kernels)):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1],
                          kernel_size=kernels[i], stride=strides[i],
                          padding=kernels[i] // 2, bias=False),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.conv_stack = nn.Sequential(*layers)
        self.pool       = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = self.conv_stack(x)
        h = self.pool(h)
        return h.squeeze(-1)


class MLPBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["mlp_in_dim"],  config["mlp_hidden"]),
            nn.ReLU(inplace=True),
            nn.Linear(config["mlp_hidden"],  config["mlp_out_dim"]),
            nn.ReLU(inplace=True),
        )

    def forward(self, sef_sr, missing_flags):
        last = sef_sr[:, -1, :]
        mean = sef_sr.mean(dim=1)
        x    = torch.cat([last, mean, missing_flags], dim=-1)
        return self.net(x)


class BISModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn_branch = CNNBranch(config)
        self.mlp_branch = MLPBranch(config)
        self.fusion = nn.Sequential(
            nn.Linear(config["fusion_in_dim"], config["embedding_dim"]),
            nn.ReLU(inplace=True),
        )
        emb = config["embedding_dim"]
        self.head_t0   = nn.Linear(emb, 1)
        self.head_t30  = nn.Linear(emb, 1)
        self.head_t60  = nn.Linear(emb, 1)
        self.head_t180 = nn.Linear(emb, 1)
        self.head_t300 = nn.Linear(emb, 1)

    def forward(self, eeg, sef_sr, missing_flags):
        cnn_feat  = self.cnn_branch(eeg)
        mlp_feat  = self.mlp_branch(sef_sr, missing_flags)
        concat    = torch.cat([cnn_feat, mlp_feat], dim=-1)
        embedding = self.fusion(concat)
        return {
            "embedding": embedding,
            "bis_t0":    self.head_t0(embedding).clamp(0, 100),
            "bis_t30":   self.head_t30(embedding).clamp(0, 100),
            "bis_t60":   self.head_t60(embedding).clamp(0, 100),
            "bis_t180":  self.head_t180(embedding).clamp(0, 100),
            "bis_t300":  self.head_t300(embedding).clamp(0, 100),
        }


# ── Load model ───────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, config: Dict, device: torch.device) -> BISModel:
    model = BISModel(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Model loaded from: {checkpoint_path}")
    return model


# ── Preprocess raw inputs ────────────────────────────────────────────────────
def preprocess_eeg(eeg_raw: np.ndarray, clip_uv: float = 500.0) -> np.ndarray:
    """
    Args:
        eeg_raw : (3840, 2) float32 — raw dual-channel EEG in µV
        clip_uv : clip amplitude (default ±500 µV, same as training)
    Returns:
        (3840, 2) float32 — NaN→0, clipped
    """
    eeg = np.nan_to_num(eeg_raw, nan=0.0)
    eeg = np.clip(eeg, -clip_uv, clip_uv).astype(np.float32)
    return eeg


def preprocess_sef_sr(sef_sr_raw: np.ndarray) -> np.ndarray:
    """
    Args:
        sef_sr_raw : (300, 2) float32 — [SEF (Hz), SR (%)] context window
    Returns:
        (300, 2) float32 — NaN→0
    """
    return np.nan_to_num(sef_sr_raw, nan=0.0).astype(np.float32)


# ── Single-case inference ────────────────────────────────────────────────────
def predict_one(
    model:         BISModel,
    eeg_raw:       np.ndarray,    # (3840, 2)  raw EEG µV
    sef_sr_raw:    np.ndarray,    # (300, 2)   [SEF, SR] at 1 Hz
    missing_flags: np.ndarray,    # (2,)       [sef_missing, sr_missing]
    device:        torch.device,
) -> Dict[str, float]:
    """
    Run the model on a single window.

    Returns
    -------
    dict with keys: bis_t0, bis_t30, bis_t60, bis_t180, bis_t300, embedding
    """
    # Preprocess
    eeg    = preprocess_eeg(eeg_raw, CONFIG["eeg_clip_uv"])
    sef_sr = preprocess_sef_sr(sef_sr_raw)

    # Add batch dimension and move to device
    eeg_t    = torch.from_numpy(eeg).unsqueeze(0).permute(0, 2, 1).to(device)  # (1, 2, 3840)
    sef_sr_t = torch.from_numpy(sef_sr).unsqueeze(0).to(device)                 # (1, 300, 2)
    flags_t  = torch.from_numpy(missing_flags).unsqueeze(0).to(device)          # (1, 2)

    with torch.no_grad():
        out = model(eeg_t, sef_sr_t, flags_t)

    return {
        "bis_t0":    float(out["bis_t0"].item()),
        "bis_t30":   float(out["bis_t30"].item()),
        "bis_t60":   float(out["bis_t60"].item()),
        "bis_t180":  float(out["bis_t180"].item()),
        "bis_t300":  float(out["bis_t300"].item()),
        "embedding": out["embedding"].cpu().numpy().flatten(),  # (64,)
    }


# ── Demo / usage example ─────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load trained model ───────────────────────────────────────────────────
    model = load_model(CONFIG["checkpoint"], CONFIG, device)

    # ── Build a synthetic input (replace with your real data) ───────────────
    # EEG: 30 seconds × 128 Hz = 3840 samples, 2 channels
    eeg_raw = np.random.randn(3840, 2).astype(np.float32) * 50.0  # µV amplitudes

    # SEF/SR: 5-minute context (300 seconds at 1 Hz) for 2 channels
    # SEF ranges 0–50 Hz; SR ranges 0–100 %
    sef_sr_raw = np.column_stack([
        np.random.uniform(8, 25, 300),   # SEF in Hz
        np.random.uniform(0, 10, 300),   # SR in %
    ]).astype(np.float32)

    # Missing flags: 0.0 = channel present, 1.0 = channel entirely absent
    missing_flags = np.array([0.0, 0.0], dtype=np.float32)  # both channels present

    # ── Run inference ────────────────────────────────────────────────────────
    result = predict_one(model, eeg_raw, sef_sr_raw, missing_flags, device)

    # ── Print outputs ────────────────────────────────────────────────────────
    print("\n=== BIS Predictions ===")
    print(f"  BIS now  (t+0s)  : {result['bis_t0']:.1f}")
    print(f"  BIS +30s         : {result['bis_t30']:.1f}")
    print(f"  BIS +60s         : {result['bis_t60']:.1f}")
    print(f"  BIS +3min        : {result['bis_t180']:.1f}")
    print(f"  BIS +5min        : {result['bis_t300']:.1f}")
    print(f"  Embedding shape  : {result['embedding'].shape}")
    print("\nBIS interpretation: 0–40 deep anesthesia | 40–60 general anesthesia | 60–100 conscious")
