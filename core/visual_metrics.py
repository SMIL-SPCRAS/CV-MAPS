from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import (
    VISUAL_DISTANCE_BODY_CLOSE,
    VISUAL_DISTANCE_BODY_FAR,
    VISUAL_DISTANCE_FACE_CLOSE,
    VISUAL_DISTANCE_FACE_FAR,
    VISUAL_CENTER_RATIO_GOOD,
    VISUAL_LIGHT_BACKLIT_DELTA,
    VISUAL_LIGHT_DARK_RATIO,
    VISUAL_LIGHT_FLAT_CONTRAST,
    VISUAL_LIGHT_LOW_P95,
    VISUAL_LIGHT_OVEREXPOSED_RATIO,
    VISUAL_POSITION_CENTER_TOL,
    VISUAL_POSITION_STABILITY_BAD,
)

BBox = Tuple[int, int, int, int]

_LIGHTING_NOTES = {
    "normal": "Lighting looks even and sufficient. Face details are clear.",
    "low": "Lighting is too dim; facial details may be hard to see.",
    "overexposed": "Lighting is too bright; highlights are blown out.",
    "flat": "Lighting is flat with low contrast; the image looks gray.",
    "backlit": "Backlight detected; background is brighter than the face.",
    "unknown": "Lighting could not be assessed.",
}



def _clamp_bbox(bbox: Optional[BBox], w: int, h: int) -> Optional[BBox]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def compute_frame_metrics(
    frame_bgr: np.ndarray,
    face_bbox: Optional[BBox],
    body_bbox: Optional[BBox],
) -> Dict[str, Optional[float]]:
    h, w = frame_bgr.shape[:2]

    face_bbox = _clamp_bbox(face_bbox, w, h)
    body_bbox = _clamp_bbox(body_bbox, w, h)

    bbox = face_bbox or body_bbox
    bbox_kind = "face" if face_bbox is not None else ("body" if body_bbox is not None else "")

    dx = dy = area_ratio = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = (cx - w / 2.0) / float(w)
        dy = (cy - h / 2.0) / float(h)
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)

    y_plane = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    p05, p50, p95 = np.percentile(y_plane, [5, 50, 95])
    dark_ratio = float(np.mean(y_plane < 20))
    bright_ratio = float(np.mean(y_plane > 235))
    contrast = float(p95 - p05)

    face_median = None
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        face_roi = y_plane[y1:y2, x1:x2]
        if face_roi.size:
            face_median = float(np.median(face_roi))

    return {
        "dx": dx,
        "dy": dy,
        "area_ratio": area_ratio,
        "bbox_kind": bbox_kind,
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "contrast": contrast,
        "face_median": face_median,
    }


def aggregate_interview_metrics(frame_stats: List[Dict[str, Optional[float]]]) -> Dict[str, object]:
    if not frame_stats:
        return {
            "body_position": "unknown",
            "center_ratio": 0.0,
            "position_stability": 0.0,
            "distance_to_camera": "unknown",
            "distance_ratio": 0.0,
            "distance_source": "",
            "framing_note": "Framing could not be assessed.",
            "lighting": "unknown",
            "lighting_stats": {},
            "frames_used": 0,
        }

    dxs = [s["dx"] for s in frame_stats if s.get("dx") is not None]
    dys = [s["dy"] for s in frame_stats if s.get("dy") is not None]
    area = [s["area_ratio"] for s in frame_stats if s.get("area_ratio") is not None]
    kinds = [s.get("bbox_kind") for s in frame_stats if s.get("bbox_kind")]

    center_ratio = 0.0
    pos_label = "unknown"
    stability = 0.0
    framing_note = "Framing could not be assessed."
    if dxs and dys:
        dx_med = float(np.median(dxs))
        dy_med = float(np.median(dys))
        within = [
            (abs(dx) <= VISUAL_POSITION_CENTER_TOL and abs(dy) <= VISUAL_POSITION_CENTER_TOL)
            for dx, dy in zip(dxs, dys)
        ]
        center_ratio = float(np.mean(within)) if within else 0.0
        if abs(dx_med) <= VISUAL_POSITION_CENTER_TOL and abs(dy_med) <= VISUAL_POSITION_CENTER_TOL:
            pos_label = "center"
        elif abs(dx_med) >= abs(dy_med):
            pos_label = "right" if dx_med > 0 else "left"
        else:
            pos_label = "down" if dy_med > 0 else "up"
        stability = float(np.sqrt(np.var(dxs) + np.var(dys)))
        framing_parts = []
        if pos_label == "center" and center_ratio >= VISUAL_CENTER_RATIO_GOOD:
            framing_parts.append("Subject is centered in the frame.")
        else:
            framing_parts.append(f"Subject is off-center ({pos_label}).")
        if stability > VISUAL_POSITION_STABILITY_BAD:
            framing_parts.append("Camera framing is unstable; reduce movement.")
        else:
            framing_parts.append("Framing is stable.")
        framing_note = " ".join(framing_parts)

    distance_label = "unknown"
    distance_ratio = 0.0
    distance_source = ""
    if area:
        distance_ratio = float(np.median(area))
        distance_source = "face" if "face" in kinds else "body"
        if distance_source == "face":
            if distance_ratio >= VISUAL_DISTANCE_FACE_CLOSE:
                distance_label = "close"
            elif distance_ratio < VISUAL_DISTANCE_FACE_FAR:
                distance_label = "far"
            else:
                distance_label = "normal"
        else:
            if distance_ratio >= VISUAL_DISTANCE_BODY_CLOSE:
                distance_label = "close"
            elif distance_ratio < VISUAL_DISTANCE_BODY_FAR:
                distance_label = "far"
            else:
                distance_label = "normal"

    p95_vals = [s.get("p95") for s in frame_stats if s.get("p95") is not None]
    contrast_vals = [s.get("contrast") for s in frame_stats if s.get("contrast") is not None]
    dark_vals = [s.get("dark_ratio") for s in frame_stats if s.get("dark_ratio") is not None]
    bright_vals = [s.get("bright_ratio") for s in frame_stats if s.get("bright_ratio") is not None]
    face_medians = [s.get("face_median") for s in frame_stats if s.get("face_median") is not None]
    frame_medians = [s.get("p50") for s in frame_stats if s.get("p50") is not None]

    p95_med = float(np.median(p95_vals)) if p95_vals else 0.0
    contrast_med = float(np.median(contrast_vals)) if contrast_vals else 0.0
    dark_med = float(np.median(dark_vals)) if dark_vals else 0.0
    bright_med = float(np.median(bright_vals)) if bright_vals else 0.0
    face_med = float(np.median(face_medians)) if face_medians else None
    frame_med = float(np.median(frame_medians)) if frame_medians else None

    lighting = "normal"
    if bright_med > VISUAL_LIGHT_OVEREXPOSED_RATIO:
        lighting = "overexposed"
    elif face_med is not None and frame_med is not None and (frame_med - face_med) > VISUAL_LIGHT_BACKLIT_DELTA:
        lighting = "backlit"
    elif p95_med < VISUAL_LIGHT_LOW_P95 or dark_med > VISUAL_LIGHT_DARK_RATIO:
        lighting = "low"
    elif contrast_med < VISUAL_LIGHT_FLAT_CONTRAST:
        lighting = "flat"

    lighting_stats = {
        "p95": round(p95_med, 2),
        "contrast": round(contrast_med, 2),
        "dark_ratio": round(dark_med, 3),
        "bright_ratio": round(bright_med, 3),
    }

    return {
        "body_position": pos_label,
        "center_ratio": round(center_ratio, 3),
        "position_stability": round(stability, 4),
        "distance_to_camera": distance_label,
        "distance_ratio": round(distance_ratio, 4),
        "distance_source": distance_source,
        "framing_note": framing_note,
        "lighting": lighting,
        "lighting_note": _LIGHTING_NOTES.get(lighting, _LIGHTING_NOTES["unknown"]),
        "lighting_stats": lighting_stats,
        "frames_used": int(len(frame_stats)),
    }
