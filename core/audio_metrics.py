from __future__ import annotations

from typing import Dict

import numpy as np

from app.config import (
    AUDIO_FRAME_SEC,
    AUDIO_HOP_SEC,
    AUDIO_LOUDNESS_HIGH,
    AUDIO_LOUDNESS_LOW,
    AUDIO_NOISE_HIGH,
    AUDIO_NOISE_LOW,
)


def _rms_db(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return float(20.0 * np.log10(rms + 1e-8))


def _frame_rms(samples: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if samples.size == 0:
        return np.array([0.0], dtype=float)
    if samples.size <= frame_len:
        return np.array([float(np.sqrt(np.mean(np.square(samples))))], dtype=float)

    total = 1 + (samples.size - frame_len) // hop_len
    rms = np.zeros(total, dtype=float)
    for i in range(total):
        start = i * hop_len
        window = samples[start : start + frame_len]
        rms[i] = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
    return rms


def _loudness_label(db: float) -> str:
    if db < AUDIO_LOUDNESS_LOW:
        return "quiet"
    if db > AUDIO_LOUDNESS_HIGH:
        return "loud"
    return "normal"


def _noise_label(db: float) -> str:
    if db > AUDIO_NOISE_HIGH:
        return "noisy"
    if db < AUDIO_NOISE_LOW:
        return "clean"
    return "moderate"


def compute_audio_metrics(waveform: np.ndarray, sr: int) -> Dict[str, object]:
    if waveform is None or sr <= 0:
        return {
            "audio_loudness_db": 0.0,
            "audio_loudness": "unknown",
            "audio_noise_db": 0.0,
            "audio_noise": "unknown",
            "audio_note": "Audio quality could not be assessed.",
        }

    samples = np.asarray(waveform, dtype=float).squeeze()
    if samples.ndim != 1:
        samples = samples.reshape(-1)

    loud_db = _rms_db(samples)
    loud_label = _loudness_label(loud_db)

    frame_len = max(1, int(sr * AUDIO_FRAME_SEC))
    hop_len = max(1, int(sr * AUDIO_HOP_SEC))
    frame_rms = _frame_rms(samples, frame_len, hop_len)
    noise_rms = float(np.percentile(frame_rms, 10)) if frame_rms.size else 0.0
    noise_db = float(20.0 * np.log10(noise_rms + 1e-8))
    noise_label = _noise_label(noise_db)

    note = f"Audio loudness is {loud_label}; background noise is {noise_label}."

    return {
        "audio_loudness_db": round(loud_db, 2),
        "audio_loudness": loud_label,
        "audio_noise_db": round(noise_db, 2),
        "audio_noise": noise_label,
        "audio_note": note,
    }
