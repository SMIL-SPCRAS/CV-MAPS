from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MATCH_THRESHOLD = 0.8
DEFAULT_TITLE_MATCH_THRESHOLD = 0.6
DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY = "euclidean"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "profession-profiles.csv"

_DF_CACHE: Optional[pd.DataFrame] = None

# Mapping from model keys -> possible dataset columns (first existing wins).
_COLUMN_ALIASES: Dict[str, List[str]] = {
    "Openness": ["Openness"],
    "Conscientiousness": ["Conscientiousness", "Conscientousness"],
    "Extraversion": ["Extraversion"],
    "Agreeableness": ["Agreeableness"],
    "Non-Neuroticism": ["Non-Neuroticism", "Emotional_Range"],
    "Conversation": ["Conversation"],
    "Openness to Change": ["Openness to Change", "Openness_to_Change"],
    "Hedonism": ["Hedonism"],
    "Self-enhancement": ["Self-enhancement", "Self_enhancement"],
    "Self-transcendence": ["Self-transcendence", "Self_transcendence"],
}

_BASE_KEYS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Non-Neuroticism",
]

_EXTRA_KEYS = [
    "Conversation",
    "Openness to Change",
    "Hedonism",
    "Self-enhancement",
    "Self-transcendence",
]

_TRAIT_VECTOR_KEYS = _BASE_KEYS + _EXTRA_KEYS


def _load_profiles(path: Path = DATA_PATH) -> pd.DataFrame:
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = pd.read_csv(path)
    return _DF_CACHE


def list_professions(path: Path = DATA_PATH) -> List[str]:
    df = _load_profiles(path)
    profs = [str(x).strip() for x in df.get("Profession", []) if str(x).strip()]
    return sorted(set(profs))


def get_profession_trait_vector(
    job_title: str,
    keys: Optional[Sequence[str]] = None,
    path: Path = DATA_PATH,
) -> Optional[List[float]]:
    title = (job_title or "").strip()
    if not title:
        return None

    df = _load_profiles(path)
    if "Profession" not in df.columns:
        return None

    mask = df["Profession"].astype(str).str.strip().str.lower() == title.lower()
    if not mask.any():
        return None

    row = df[mask].iloc[0]
    ordered_keys = list(keys or _TRAIT_VECTOR_KEYS)

    col_map: Dict[str, Optional[str]] = {}
    for key in ordered_keys:
        col_map[key] = next((c for c in _COLUMN_ALIASES.get(key, []) if c in df.columns), None)

    vec: List[float] = []
    for key in ordered_keys:
        col = col_map.get(key)
        if not col:
            vec.append(0.0)
            continue
        try:
            vec.append(float(row[col]))
        except Exception:
            vec.append(0.0)
    return vec


def _resolve_columns(df: pd.DataFrame, keys: List[str]) -> List[Tuple[str, str]]:
    resolved: List[Tuple[str, str]] = []
    for key in keys:
        for col in _COLUMN_ALIASES.get(key, []):
            if col in df.columns:
                resolved.append((key, col))
                break
    return resolved


def _normalize_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def _best_title_match(
    df: pd.DataFrame, job_title: str, threshold: float
) -> Tuple[Optional[int], float]:
    title_tokens = _normalize_text(job_title)
    if not title_tokens:
        return None, 0.0
    title_set = set(title_tokens)

    best_idx = None
    best_score = 0.0
    for idx, name in df["Profession"].items():
        name_tokens = _normalize_text(str(name))
        if not name_tokens:
            continue
        name_set = set(name_tokens)
        overlap = len(title_set & name_set) / max(1, len(title_set))
        if overlap > best_score:
            best_score = overlap
            best_idx = idx

    if best_idx is not None and best_score >= threshold:
        return best_idx, float(best_score)
    return None, float(best_score)


def match_profession(
    user_scores: Dict[str, float],
    job_title: str = "",
    top_k: int = DEFAULT_TOP_K,
    similarity: str = DEFAULT_SIMILARITY,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    title_match_threshold: float = DEFAULT_TITLE_MATCH_THRESHOLD,
    use_extended: bool = False,
) -> Dict[str, object]:
    df = _load_profiles()

    keys = list(_BASE_KEYS)
    if use_extended:
        keys.extend(_EXTRA_KEYS)

    resolved = _resolve_columns(df, keys)
    used_cols = [col for _, col in resolved]
    if not used_cols:
        return {
            "top_professions": [],
            "matched_profession": None,
            "used_features": [],
            "similarity": similarity,
            "threshold": threshold,
        }

    user_vec = np.array([float(user_scores.get(key, 0.0)) for key, _ in resolved], dtype=float)
    mat = df[used_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if similarity == "cosine":
        sims = cosine_similarity([user_vec], mat)[0]
    else:
        diffs = mat - user_vec
        dist = np.linalg.norm(diffs, axis=1)
        sims = 1.0 / (1.0 + dist)

    df_work = df.copy()
    df_work["similarity"] = sims

    top = df_work.nlargest(int(top_k), "similarity")
    top_list = []
    for _, row in top.iterrows():
        top_list.append(
            {
                "profession": row["Profession"],
                "similarity": float(row["similarity"]),
                "traits": {col: float(row[col]) for col in used_cols},
                "n": int(row["n"]) if "n" in row and not pd.isna(row["n"]) else None,
            }
        )

    matched_prof = None
    match_source = None
    title_match_score = 0.0
    if job_title:
        title = job_title.strip().lower()
        exact_mask = df_work["Profession"].astype(str).str.strip().str.lower() == title
        if exact_mask.any():
            row = df_work[exact_mask].iloc[0]
            matched_prof = {
                "profession": row["Profession"],
                "similarity": float(row["similarity"]),
                "traits": {col: float(row[col]) for col in used_cols},
                "n": int(row["n"]) if "n" in row and not pd.isna(row["n"]) else None,
            }
            match_source = "exact"
            title_match_score = 1.0
        else:
            match_idx, title_match_score = _best_title_match(
                df_work, job_title, title_match_threshold
            )
            if match_idx is not None:
                row = df_work.loc[match_idx]
                matched_prof = {
                    "profession": row["Profession"],
                    "similarity": float(row["similarity"]),
                    "traits": {col: float(row[col]) for col in used_cols},
                    "n": int(row["n"]) if "n" in row and not pd.isna(row["n"]) else None,
                }
                match_source = "title"

    if matched_prof is None and top_list:
        if top_list[0]["similarity"] >= float(threshold):
            matched_prof = top_list[0]
            match_source = "top1"

    return {
        "top_professions": top_list,
        "matched_profession": matched_prof,
        "match_source": match_source,
        "title_match_score": float(title_match_score),
        "used_features": used_cols,
        "similarity": similarity,
        "threshold": float(threshold),
        "title_match_threshold": float(title_match_threshold),
    }
