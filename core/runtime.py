# core/runtime.py
from __future__ import annotations
import os
import numpy as np
import torch
import torchaudio
try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency
    sf = None
from transformers import pipeline

from core.modalities.video.feature_extractor import PretrainedImageEmbeddingExtractor
from core.modalities.audio.feature_extractor import PretrainedAudioEmbeddingExtractor
from core.modalities.text.feature_extractor import PretrainedTextEmbeddingExtractor
from core.models.models import MultiModalFusionModelWithAblation
from core.modalities.video.video_preprocessor import get_metadata

from core.media_utils import (
    ensure_dir, get_oscilloscope, convert_video_to_audio
)
from core.audio_metrics import compute_audio_metrics

import numpy as np
import torch.nn.functional as F

_DEVICE: torch.device | None = None
_ASR = None
_IMG = None
_AUD = None
_TXT = None
_FUSION = None


def _resolve_device(device: str | None) -> torch.device:
    if device in ("cuda", "cpu"):
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_everything(
    checkpoint_path: str = "best_ep9_emo0.6390_pkl0.8269.pt",
    device: str | None = None
):
    global _ASR, _IMG, _AUD, _TXT, _FUSION, _DEVICE
    if _DEVICE is None:
        _DEVICE = _resolve_device(device)

    # ASR (Whisper) — device=0 для cuda, -1 для cpu
    if _ASR is None:
        _ASR = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16 if _DEVICE.type == "cuda" else torch.float32,
            device=0 if _DEVICE.type == "cuda" else -1,
            return_timestamps=False,
        )

    if _IMG is None:
        _IMG = PretrainedImageEmbeddingExtractor(device=_DEVICE.type)
    if _AUD is None:
        _AUD = PretrainedAudioEmbeddingExtractor(device=_DEVICE.type)
    if _TXT is None:
        _TXT = PretrainedTextEmbeddingExtractor(device=_DEVICE.type)

    if _FUSION is None:
        _FUSION = MultiModalFusionModelWithAblation(
            hidden_dim=256,
            num_heads=8,
            dropout=0.2,
            emo_out_dim=7,
            pkl_out_dim=5,
            device=_DEVICE.type,
            # оставлено как было; добавлен enabled_modalities=None для совместимости, если ты уже перешёл на новую схему
            ablation_config={"disabled_modalities": [], "enabled_modalities": None, "disable_guide_pkl": True},
        ).to(device=_DEVICE)
        state = torch.load(checkpoint_path, map_location=_DEVICE)
        _FUSION.load_state_dict(state)
        _FUSION.eval()


def _load_audio_mono(path: str, target_sr: int = 16000):
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        if sf is None:
            raise RuntimeError(
                "torchaudio backend unavailable. Install 'soundfile' or enable "
                "an ffmpeg backend for torchaudio."
            )
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform.squeeze(0).cpu().numpy(), sr


def analyze_video_basic(
    video_path: str,
    checkpoint_path: str = "best_ep9_emo0.6390_pkl0.8269.pt",
    segment_length: int = 30,
    device: str | None = None,
    save_dir: str = "outputs",
) -> dict:
    init_everything(checkpoint_path=checkpoint_path, device=device)

    base = os.path.splitext(os.path.basename(video_path))[0]
    clip_dir = ensure_dir(save_dir)

    print(video_path)
    audio_path = convert_video_to_audio(file_path=video_path, file_save=os.path.join(clip_dir, base))
    print(audio_path)
    waveform, sr = _load_audio_mono(audio_path, target_sr=16000)
    duration_sec = float(len(waveform) / sr) if sr else 0.0
    audio_metrics = compute_audio_metrics(waveform, sr)
    get_oscilloscope(waveform, sr, clip_dir)
    osc_path = os.path.join(clip_dir, "oscilloscope.jpg")

    result_asr = _ASR(waveform)
    if isinstance(result_asr, dict) and "chunks" in result_asr:
        text = "".join(chunk["text"] for chunk in result_asr["chunks"])
    else:
        text = result_asr["text"]

    clip_processor = _IMG.processor
    _, body_tensor, face_tensor, scene_tensor = get_metadata(video_path, segment_length, clip_processor)

    video_results = _IMG.extract(body_tensor=body_tensor, face_tensor=face_tensor, scene_tensor=scene_tensor)
    audio_results = _AUD.extract(audio_path=audio_path)
    text_results = _TXT.extract(texts=text)

    def _as_logits_if_prob(t: torch.Tensor) -> torch.Tensor:
        """If tensor looks like probabilities (0..1), convert to logits to avoid double-sigmoid saturation."""
        if torch.is_floating_point(t):
            mx, mn = float(t.max()), float(t.min())
            if 0.0 <= mn and mx <= 1.0:
                eps = 1e-4
                return torch.logit(t.clamp(eps, 1 - eps))
        return t

    # NEW: безопасная упаковка (если body/face/scene отсутствуют — не падаем)
    def _pack_video(mod: str):
        r = video_results.get(mod)
        if r is None:
            return None
        feat = torch.cat(
            (r["last_emo_encoder_features"].mean(dim=1),
             r["last_per_encoder_features"].mean(dim=1)),
            dim=1
        )
        emo = r["emotion_logits"]
        per = _as_logits_if_prob(r["personality_scores"])
        return feat, emo, per

    features: dict = {}
    emo_logits: dict = {}
    per_scores: dict = {}

    for mod in ["body", "face", "scene"]:
        packed = _pack_video(mod)
        if packed is None:
            continue
        feat, emo, per = packed
        features[mod] = feat
        emo_logits[mod] = emo
        per_scores[mod] = per

    # audio/text обычно есть, но всё равно защитимся
    if audio_results is not None:
        features["audio"] = torch.cat(
            (audio_results["last_emo_encoder_features"].mean(dim=1),
             audio_results["last_per_encoder_features"].mean(dim=1)),
            dim=1
        )
        emo_logits["audio"] = audio_results["emotion_logits"]
        per_scores["audio"] = _as_logits_if_prob(audio_results["personality_scores"])

    if text_results is not None:
        features["text"] = torch.cat(
            (text_results["last_emo_encoder_features"].mean(dim=1),
             text_results["last_per_encoder_features"].mean(dim=1)),
            dim=1
        )
        emo_logits["text"] = text_results["emotion_logits"]
        per_scores["text"] = _as_logits_if_prob(text_results["personality_scores"])

    if not features:
        raise RuntimeError("No modalities extracted: features dict is empty")

    with torch.no_grad():
        outputs = _FUSION({
            "features": features,
            "emotion_logits": emo_logits,
            "personality_scores": per_scores,
        })

        emo_prob = torch.softmax(outputs["emotion_logits"], dim=-1).cpu().numpy()[0]
        per_sig = outputs["personality_scores"].cpu().numpy()[0]

    return {
        "transcript": text,
        "emotion_logits": emo_prob,
        "personality_scores": per_sig,
        "oscilloscope_path": osc_path,
        "video_duration_sec": duration_sec,
        "audio_metrics": audio_metrics,
    }


def get_multitask_pred_with_attribution(
    video_path: str,
    checkpoint_path: str = "best_ep9_emo0.6390_pkl0.8269.pt",
    segment_length: int = 30,
    device: str | None = None,
    save_dir: str | None = None,
):
    init_everything(checkpoint_path=checkpoint_path, device=device)

    # FIX: video_path[:-4] ломается на .webm и вообще на любые расширения != 3 символа
    base = os.path.splitext(os.path.basename(video_path))[0]
    if save_dir:
        ensure_dir(save_dir)
        file_save = os.path.join(save_dir, base)
    else:
        file_save = os.path.splitext(video_path)[0]
    audio_path = convert_video_to_audio(file_path=video_path, file_save=file_save)

    clip_processor = _IMG.processor
    waveform, sr = _load_audio_mono(audio_path, target_sr=16000)

    result = _ASR(waveform)
    if isinstance(result, dict) and "chunks" in result:
        text = "".join(chunk["text"] for chunk in result["chunks"])
    else:
        text = result["text"]

    _, body_tensor, face_tensor, scene_tensor = get_metadata(video_path, segment_length, clip_processor)
    video_results = _IMG.extract(body_tensor=body_tensor, face_tensor=face_tensor, scene_tensor=scene_tensor)
    audio_results = _AUD.extract(audio_path=audio_path)
    text_results = _TXT.extract(texts=text)

    def _as_logits_if_prob(t: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(t):
            mx, mn = float(t.max()), float(t.min())
            if 0.0 <= mn and mx <= 1.0:
                eps = 1e-4
                return torch.logit(t.clamp(eps, 1 - eps))
        return t

    modality_features = {}
    modality_emo_logits = {}
    modality_per_scores = {}

    def _requires(x: torch.Tensor) -> torch.Tensor:
        y = x.clone().detach().requires_grad_(True)
        return y

    # NEW: не падаем, если body/face/scene отсутствуют
    for mod in ["body", "face", "scene", "audio", "text"]:
        if mod in ["body", "face", "scene"]:
            res = video_results.get(mod)
            if res is None:
                continue
        else:
            res = audio_results if mod == "audio" else text_results
            if res is None:
                continue

        feat = torch.cat(
            (res["last_emo_encoder_features"].mean(dim=1),
             res["last_per_encoder_features"].mean(dim=1)),
            dim=1
        )
        modality_features[mod] = _requires(feat)
        modality_emo_logits[mod] = _requires(res["emotion_logits"])
        modality_per_scores[mod] = _requires(_as_logits_if_prob(res["personality_scores"]))

    if not modality_features:
        raise RuntimeError("No modalities extracted for attribution")

    fusion_input = {
        "features": modality_features,
        "emotion_logits": modality_emo_logits,
        "personality_scores": modality_per_scores,
    }

    _FUSION.eval()
    with torch.set_grad_enabled(True):
        outputs = _FUSION(fusion_input)
        emotion_logits = outputs["emotion_logits"]
        personality_scores = outputs["personality_scores"]
        prob_emo = torch.softmax(emotion_logits, dim=-1)
        prob_per = personality_scores

    attribution = {
        "features": {"emotion": {}, "personality": {}},
        "emotion_logits": {"emotion": {}, "personality": {}},
        "personality_scores": {"emotion": {}, "personality": {}},
    }

    def compute_ixg(input_tensor, target_scalar):
        grad = torch.autograd.grad(target_scalar, input_tensor, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(input_tensor)
        return (input_tensor * grad).squeeze(0).detach().cpu().numpy()

    for mod in modality_features:
        emb = modality_features[mod]
        attribution["features"]["emotion"][mod] = np.array([compute_ixg(emb, prob_emo[0, i]) for i in range(7)])
        attribution["features"]["personality"][mod] = np.array([compute_ixg(emb, prob_per[0, i]) for i in range(5)])

    for mod in modality_emo_logits:
        log = modality_emo_logits[mod]
        attribution["emotion_logits"]["emotion"][mod] = np.array([compute_ixg(log, prob_emo[0, i]) for i in range(7)])
        attribution["emotion_logits"]["personality"][mod] = np.array([compute_ixg(log, prob_per[0, i]) for i in range(5)])

    for mod in modality_per_scores:
        scr = modality_per_scores[mod]
        attribution["personality_scores"]["emotion"][mod] = np.array([compute_ixg(scr, prob_emo[0, i]) for i in range(7)])
        attribution["personality_scores"]["personality"][mod] = np.array([compute_ixg(scr, prob_per[0, i]) for i in range(5)])

    return {
        "emotion_logits": prob_emo.detach().cpu().squeeze().numpy(),
        "personality_scores": prob_per.detach().cpu().squeeze().numpy(),
        "attribution": attribution
    }
