import torch
import numpy as np
import torchaudio
try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency
    sf = None
from transformers import Wav2Vec2Processor

from .architectures import EmotionModel
from .model_loader import load_fusion_model

class PretrainedAudioEmbeddingExtractor:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        fusion_ckpt: str = "core/modalities/audio/checkpoints/best_fusion_overall_mamba.pt",
        emo_ckpt:   str = "core/modalities/audio/checkpoints/final_best_model_uni_mamba.pt",
        per_ckpt:   str = "core/modalities/audio/checkpoints/best_mamba_regressor.pth",
        target_sr:  int = 16_000,
    ):
        self.device   = torch.device(device)
        self.target_sr = target_sr

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wav2vec   = EmotionModel.from_pretrained(model_name).to(self.device).eval()

        self.fusion, _ = load_fusion_model(
            fusion_ckpt, emo_ckpt, per_ckpt, device=self.device
        )

    @torch.no_grad()
    def extract(
        self,
        *,
        audio_path: str | None = None,
        waveform:   np.ndarray | None = None,
    ) -> dict:
        if waveform is None and audio_path is None:
            raise ValueError("Укажи либо `audio_path`, либо `waveform`")

        if waveform is None:
            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception:
                if sf is None:
                    raise RuntimeError(
                        "torchaudio backend unavailable. Install 'soundfile' or enable "
                        "an ffmpeg backend for torchaudio."
                    )
                data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
                waveform = torch.from_numpy(data.T)
            if waveform.ndim == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            waveform = waveform.squeeze(0).cpu().numpy()
        elif isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().cpu().numpy()
        else:
            waveform = np.asarray(waveform).squeeze()

        if waveform.ndim == 1:
            waveform = waveform[None, :]  # [1, num_samples]
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T

        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        ).to(self.device)


        hidden = self.wav2vec(inputs["input_values"])

        if "attention_mask" in inputs:
            lengths = inputs["attention_mask"].sum(dim=1)  # (B,)
        else:
            lengths = torch.tensor(
                [hidden.shape[1]] * hidden.shape[0],
                device=self.device,
                dtype=torch.long,
            )

        out = self.fusion(
            emo_input=(hidden, lengths),
            per_input=(hidden, lengths),
            return_features=True
        )

        return {
            "emotion_logits":     out["emotion_logits"].cpu(),
            "personality_scores": out["personality_scores"].cpu(),
            "last_emo_encoder_features": out["last_emo_encoder_features"].cpu(),
            "last_per_encoder_features": out["last_per_encoder_features"].cpu(),
        }
