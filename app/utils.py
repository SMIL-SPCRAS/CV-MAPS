from __future__ import annotations

import base64
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch

from app.config import EMO_ORDER, PERS_ORDER


_ATTEMPT_TIMER = {"start_ts": None, "attempt": None}


def start_attempt_timer(attempt_id: int | None) -> None:
    _ATTEMPT_TIMER["start_ts"] = time.time()
    _ATTEMPT_TIMER["attempt"] = attempt_id
    label = f"attempt {attempt_id}" if attempt_id else "attempt"
    print(f"[timer] {label} total start")


def finish_attempt_timer() -> None:
    start_ts = _ATTEMPT_TIMER.get("start_ts")
    attempt_id = _ATTEMPT_TIMER.get("attempt")
    if start_ts:
        elapsed = time.time() - float(start_ts)
        label = f"attempt {attempt_id}" if attempt_id else "attempt"
        print(f"[timer] {label} total end total={elapsed:.2f}s")
    _ATTEMPT_TIMER["start_ts"] = None
    _ATTEMPT_TIMER["attempt"] = None


def reset_attempt_timer() -> None:
    _ATTEMPT_TIMER["start_ts"] = None
    _ATTEMPT_TIMER["attempt"] = None


def load_css() -> str:
    css_path = Path(__file__).resolve().parents[1] / "assets" / "app.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def choose_device(requested: str | None) -> str:
    if requested in ("cuda", "cpu"):
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_session_dir(base_dir: str, max_sessions: int = 5) -> str:
    root = ensure_dir(os.path.join(base_dir, "sessions"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int(time.time() * 1000) % 1000
    session_dir = os.path.join(root, f"session_{stamp}_{millis:03d}")
    ensure_dir(session_dir)

    if max_sessions and max_sessions > 0:
        entries = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path) and name.startswith("session_"):
                try:
                    entries.append((os.path.getmtime(path), path))
                except OSError:
                    continue
        entries.sort(key=lambda item: item[0], reverse=True)
        for _, path in entries[max_sessions:]:
            try:
                shutil.rmtree(path)
            except OSError:
                pass
    return session_dir


def map_scores(labels: List[str], scores: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, label in enumerate(labels):
        val = float(scores[i]) if i < len(scores) else 0.0
        out[label] = val
    return out


def build_output_payload(result: dict) -> dict:
    emo_probs = list(map(float, result.get("emotion_logits", [])))
    if not emo_probs:
        emo_probs = [0.0] * len(EMO_ORDER)
    if len(emo_probs) < len(EMO_ORDER):
        emo_probs.extend([0.0] * (len(EMO_ORDER) - len(emo_probs)))

    top_idx = max(range(len(emo_probs)), key=lambda i: emo_probs[i]) if emo_probs else 0
    emo_dist = map_scores(EMO_ORDER, emo_probs)

    per_scores = list(map(float, result.get("personality_scores", [])))
    if not per_scores:
        per_scores = [0.0] * len(PERS_ORDER)
    if len(per_scores) < len(PERS_ORDER):
        per_scores.extend([0.0] * (len(PERS_ORDER) - len(per_scores)))
    per_map = map_scores(PERS_ORDER, per_scores)

    return {
        "emotion": {
            "top": EMO_ORDER[top_idx],
            "top_prob": float(emo_probs[top_idx]),
            "distribution": emo_dist,
        },
        "personality": per_map,
    }


def render_gallery_html(
    paths,
    captions,
    uid: str = "g",
    thumb_h: int = 150,
    full_h: Optional[int] = None,
) -> str:
    def _img_to_data_uri(path: str, target_h: Optional[int]) -> str:
        ext = "jpeg"
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2.imread failed")
            if target_h and target_h > 0:
                h, w = img.shape[:2]
                if h > target_h:
                    scale = float(target_h) / float(h)
                    new_w = max(1, int(w * scale))
                    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                b64 = base64.b64encode(buf).decode("ascii")
                return f"data:image/{ext};base64,{b64}"
        except Exception:
            pass

        ext = (os.path.splitext(path)[1].lower().lstrip(".") or "jpeg")
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/{ext};base64,{b64}"

    tiles = []
    full_h = full_h if full_h is not None else max(thumb_h * 3, 360)
    for i, (p, cap) in enumerate(zip(paths, captions)):
        try:
            thumb_src = _img_to_data_uri(p, thumb_h)
            full_src = _img_to_data_uri(p, full_h)
        except Exception:
            thumb_src = ""
            full_src = ""
        fig_id = f"{uid}_img{i}"
        tiles.append(
            f"""
        <figure style="margin:0 12px 0 0; display:inline-block; text-align:center; vertical-align:top;">
          <a href="#{fig_id}" style="display:inline-block;">
            <img src="{thumb_src}" style="height:{thumb_h}px; width:auto; border-radius:12px; cursor:pointer; background:#000; box-shadow:0 2px 10px rgba(0,0,0,.35);" />
          </a>
          <figcaption style="font-size:11px; color:#ddd; opacity:0.9; margin-top:6px;">{cap}</figcaption>
        </figure>

        <div id="{fig_id}" class="lightbox">
          <a href="#" class="lightbox-close"></a>
          <img src="{full_src}" class="lightbox-content" />
        </div>
        """
        )

    strip_id = f"{uid}_strip"
    style = f"""
    <style>
    #{strip_id} {{
        white-space: nowrap;
        overflow-x: auto; overflow-y: hidden;
        padding: 6px 2px 10px 2px;
        border-radius: 10px;
        scrollbar-width: thin;
    }}

    #{strip_id}::-webkit-scrollbar {{ height: 8px; }}
    #{strip_id}::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,.2); border-radius: 6px; }}

    .lightbox {{
        display:none; position:fixed; inset:0; z-index:9999;
        background:rgba(0,0,0,0.85);
        justify-content:center; align-items:center;
    }}
    .lightbox:target {{ display:flex; }}
    .lightbox-content {{
        max-width:90%; max-height:90%;
        border-radius:12px; box-shadow:0 0 24px rgba(0,0,0,.55);
        animation:zoomIn .2s ease;
    }}
    @keyframes zoomIn {{ from{{transform:scale(.9); opacity:.7;}} to{{transform:scale(1); opacity:1;}} }}
    .lightbox-close {{ position:absolute; inset:0; }}
    </style>
    """

    script = f"""
    <script>
    (function(){{
      const el = document.getElementById("{strip_id}");
      if (!el) return;
      el.addEventListener('wheel', function(e){{
        if (e.deltaY === 0) return;
        e.preventDefault();
        el.scrollLeft += e.deltaY;
      }}, {{passive:false}});
    }})();
    </script>
    """

    return style + f'<div id="{strip_id}">{"".join(tiles)}</div>' + script


def convert_webm_to_mp4(input_file: str) -> str:
    path_save = os.path.splitext(input_file)[0] + ".mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        path_save,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path_save
