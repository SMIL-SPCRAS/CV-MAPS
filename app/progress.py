from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import io
import time
from PIL import Image

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import DEFAULT_HISTORY_MAX, EMO_ORDER, PERS_ORDER, TARGET_TRAIT_NAMES, ENABLE_RADAR_PLOTS
from app.utils import finish_attempt_timer
from core.llm.qwen_client import generate_progress_report, unload_model
from core.matching import DEFAULT_SIMILARITY, get_profession_trait_vector

_MIN_DELTA = 0.0
_ATTEMPT_COLORS = ["#f97316", "#0ea5e9", "#22c55e"]
_ARROW_UP = "▲"
_ARROW_DOWN = "▼"
_ARROW_FLAT = "•"


def _vector_from_personality(personality: Dict[str, float]) -> List[float]:
    return [float(personality.get(k, 0.0)) for k in PERS_ORDER]


def _compute_similarity(
    user_vec: List[float],
    target_vec: List[float],
    method: str,
) -> Tuple[float, float]:
    user = np.asarray(user_vec, dtype=float)
    target = np.asarray(target_vec, dtype=float)
    if method == "cosine":
        sim = float(cosine_similarity([user], [target])[0][0])
        dist = float(np.linalg.norm(target - user))
        return sim, dist

    dist = float(np.linalg.norm(target - user))
    sim = 1.0 / (1.0 + dist)
    return sim, dist


def _plot_radar_grid(
    history: List[Dict[str, object]],
    target_vec: List[float],
    target_label: str,
) -> Optional[plt.Figure]:
    if not history:
        return None

    labels = ["O", "C", "E", "A", "N"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    target = list(target_vec) + [target_vec[0]]

    count = min(3, len(history))
    fig, axes = plt.subplots(
        1,
        count,
        figsize=(3.3 * count, 3.1),
        dpi=110,
        subplot_kw={"polar": True},
    )
    if count == 1:
        axes = [axes]

    for idx in range(count):
        attempt = history[idx]
        attempt_vals = [float(attempt["scores"].get(k, 0.0)) for k in PERS_ORDER]
        attempt_vals = attempt_vals + [attempt_vals[0]]
        color = _ATTEMPT_COLORS[idx % len(_ATTEMPT_COLORS)]
        label = f"Attempt {attempt['attempt']}"
        target_legend = target_label or "Target" if idx == 0 else "_nolegend_"

        ax = axes[idx]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.25)

        ax.plot(
            angles,
            target,
            color="#111827",
            linewidth=1.4,
            linestyle="--",
            label=target_legend,
        )
        ax.plot(angles, attempt_vals, color=color, linewidth=2, label=label)
        ax.fill(angles, attempt_vals, color=color, alpha=0.25)
        ax.set_title(label, fontsize=9, pad=6)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2 if idx == 0 else 1,
            fontsize=7,
            frameon=False,
        )

    fig.tight_layout()
    return fig


def _fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    buf.close()
    return img


def _plot_progress_bars(
    history: List[Dict[str, object]],
    target_vec: List[float],
    target_label: str,
):
    labels = ["O", "C", "E", "A", "N"]
    attempts = [h["attempt"] for h in history]
    sims = [float(h["similarity"]) for h in history]
    values = [
        [float(h["scores"].get(k, 0.0)) for k in PERS_ORDER] for h in history
    ]

    n = len(values)
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=160)
    x = np.arange(len(labels))
    bar_width = 0.7 / max(1, n)
    start = -0.35 + bar_width / 2

    for i in range(n):
        offset = start + i * bar_width
        vals = values[i]
        color = _ATTEMPT_COLORS[i % len(_ATTEMPT_COLORS)]
        ax.bar(
            x + offset,
            vals,
            width=bar_width,
            label=f"Attempt {attempts[i]} (sim {sims[i]:.3f})",
            color=color,
        )

    ax.scatter(
        x,
        target_vec,
        color="#111827",
        marker="x",
        s=35,
        label=target_label or "Target",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    vals_all = np.array(values + [target_vec], dtype=float)
    y_min = float(np.min(vals_all))
    y_max = float(np.max(vals_all))
    margin = 0.08
    low = max(0.0, y_min - margin)
    high = min(1.0, y_max + margin)
    if high - low < 0.15:
        pad = 0.15 - (high - low)
        low = max(0.0, low - pad / 2)
        high = min(1.0, high + pad / 2)
    ax.set_ylim(low, high)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        ncol=1,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    ax.set_ylabel("Score", fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _format_trait_changes(
    prev: Dict[str, object],
    curr: Dict[str, object],
    target_vec: List[float],
) -> Tuple[str, str]:
    prev_scores = prev.get("scores", {})
    curr_scores = curr.get("scores", {})

    deltas: List[Tuple[str, float]] = []
    for idx, trait in enumerate(PERS_ORDER):
        target = float(target_vec[idx])
        prev_gap = abs(target - float(prev_scores.get(trait, 0.0)))
        curr_gap = abs(target - float(curr_scores.get(trait, 0.0)))
        deltas.append((trait, prev_gap - curr_gap))

    improved = [(t, d) for t, d in deltas if d > _MIN_DELTA]
    worsened = [(t, d) for t, d in deltas if d < -_MIN_DELTA]

    improved.sort(key=lambda x: x[1], reverse=True)
    worsened.sort(key=lambda x: x[1])

    improved_txt = ", ".join(f"{t} ({d:+.3f})" for t, d in improved[:2])
    worsened_txt = ", ".join(f"{t} ({d:+.3f})" for t, d in worsened[:2])
    return improved_txt, worsened_txt


def _build_progress_summary(
    history: List[Dict[str, object]],
    target_vec: List[float],
) -> str:
    if len(history) < 2:
        return ""

    lines = []
    for idx in range(1, len(history)):
        prev = history[idx - 1]
        curr = history[idx]

        prev_sim = float(prev["similarity"])
        curr_sim = float(curr["similarity"])
        sim_delta = curr_sim - prev_sim

        prev_dist = float(prev["distance"])
        curr_dist = float(curr["distance"])
        dist_delta = prev_dist - curr_dist

        improved_txt, worsened_txt = _format_trait_changes(prev, curr, target_vec)

        line = (
            f"Attempt {prev['attempt']} -> {curr['attempt']}: "
            f"Similarity {prev_sim:.3f} -> {curr_sim:.3f} ({sim_delta:+.3f}); "
            f"Distance {prev_dist:.3f} -> {curr_dist:.3f} ({dist_delta:+.3f})"
        )
        if improved_txt:
            line += f"; Improved traits: {improved_txt}"
        if worsened_txt:
            line += f"; Worsened traits: {worsened_txt}"
        if not improved_txt and not worsened_txt:
            line += "; No notable trait changes."
        lines.append(line)

    return "<br>".join(lines)


def _build_progress_table(
    history: List[Dict[str, object]],
    target_vec: List[float],
    target_label: str,
) -> str:
    if not history:
        return ""

    trait_labels = ["Openness (O)", "Conscientiousness (C)", "Extraversion (E)", "Agreeableness (A)", "Non-neuroticism (N)"]
    headers = [f"Target ({target_label})"]
    prev_sim = None
    for item in history:
        sim_pct = float(item.get("similarity", 0.0)) * 100.0
        arrow = _ARROW_FLAT
        arrow_cls = "flat"
        if prev_sim is not None:
            delta = sim_pct - prev_sim
            if delta > _MIN_DELTA * 100.0:
                arrow = _ARROW_UP
                arrow_cls = "up"
            elif delta < -_MIN_DELTA * 100.0:
                arrow = _ARROW_DOWN
                arrow_cls = "down"
        prev_sim = sim_pct
        headers.append(
            f"Attempt {item['attempt']} ({sim_pct:.1f}%) "
            f"<span class='delta {arrow_cls}'>({arrow})</span>"
        )

    lines = [
        "<table class='progress-table'>",
        "<thead><tr><th>Trait</th>",
        "".join(f"<th>{h}</th>" for h in headers),
        "</tr></thead>",
        "<tbody>",
    ]

    for idx, trait in enumerate(PERS_ORDER):
        target_pct = float(target_vec[idx]) * 100.0
        row = [f"<td class='trait-cell'>{trait_labels[idx]}</td>"]
        row.append(f"<td>{target_pct:.1f}%</td>")

        prev_gap = None
        for attempt in history:
            curr_val = float(attempt["scores"].get(trait, 0.0))
            arrow = _ARROW_FLAT
            arrow_cls = "flat"
            curr_gap = abs(target_vec[idx] - curr_val)
            if prev_gap is not None:
                if curr_gap < prev_gap - _MIN_DELTA:
                    arrow = _ARROW_UP
                    arrow_cls = "up"
                elif curr_gap > prev_gap + _MIN_DELTA:
                    arrow = _ARROW_DOWN
                    arrow_cls = "down"
            prev_gap = curr_gap
            curr_pct = curr_val * 100.0
            row.append(
                f"<td>{curr_pct:.1f}% <span class='delta {arrow_cls}'>({arrow})</span></td>"
            )

        lines.append("<tr>" + "".join(row) + "</tr>")

    lines.append("</tbody></table>")
    return "".join(lines)


def update_history(
    payload: dict,
    job_title: str,
    language: str,
    history: Optional[List[Dict[str, object]]],
    view_mode: int = 0,
    max_items: int = DEFAULT_HISTORY_MAX,
    similarity_method: str = DEFAULT_SIMILARITY,
):
    if not isinstance(payload, dict) or payload.get("error"):
        finish_attempt_timer()
        return (
            history or [],
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    personality = payload.get("personality") or {}
    full_target = get_profession_trait_vector(job_title or "")
    if not full_target:
        prof_match = payload.get("profession_match", {}) or {}
        fallback_name = ""
        matched = prof_match.get("matched_profession") or {}
        if matched:
            fallback_name = str(matched.get("profession", "") or "")
        if not fallback_name:
            top = prof_match.get("top_professions", []) or []
            if top:
                fallback_name = str(top[0].get("profession", "") or "")
        if fallback_name:
            full_target = get_profession_trait_vector(fallback_name)
            if not job_title:
                job_title = fallback_name
    if not full_target:
        finish_attempt_timer()
        return (
            history or [],
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    if len(full_target) < len(TARGET_TRAIT_NAMES):
        full_target = list(full_target) + [0.0] * (len(TARGET_TRAIT_NAMES) - len(full_target))

    target_vec = [float(v) for v in full_target[: len(PERS_ORDER)]]

    user_vec = _vector_from_personality(personality)
    similarity, distance = _compute_similarity(user_vec, target_vec, similarity_method)

    history_list = list(history or [])
    last_attempt = history_list[-1]["attempt"] if history_list else 0
    attempt_id = int(last_attempt) + 1

    emotion_dist = payload.get("emotion", {}).get("distribution", {}) or {}
    emotion_names = list(EMO_ORDER)
    emotion_probs = [float(emotion_dist.get(lbl, 0.0)) for lbl in emotion_names]

    profession_match = payload.get("profession_match", {}) or {}
    top_professions = profession_match.get("top_professions", []) or []

    history_list.append(
        {
            "attempt": attempt_id,
            "scores": {k: float(personality.get(k, 0.0)) for k in PERS_ORDER},
            "similarity": float(similarity),
            "distance": float(distance),
            "predicted_trait_names": list(PERS_ORDER),
            "predicted_trait_scores": [float(personality.get(k, 0.0)) for k in PERS_ORDER],
            "emotion_names": emotion_names,
            "emotion_probs": emotion_probs,
            "transcription": str(payload.get("transcript", "") or ""),
            "body_position": str(payload.get("body_position", "center")),
            "distance_to_camera": str(payload.get("distance_to_camera", "normal")),
            "lighting": str(payload.get("lighting", "normal")),
            "target_similarity": float(similarity),
            "top_professions": top_professions,
        }
    )

    if max_items and len(history_list) > int(max_items):
        history_list = history_list[-int(max_items) :]

    limit_reached = bool(max_items) and len(history_list) >= int(max_items)

    limit_msg = (
        f"Maximum attempts reached ({max_items}). Start a new session to continue."
        if limit_reached
        else ""
    )
    limit_update = gr.update(value=limit_msg, visible=bool(limit_msg))
    run_update = gr.update(interactive=not limit_reached)
    try_again_update = gr.update(
        visible=(len(history_list) >= 1 and not limit_reached),
        interactive=not limit_reached,
    )

    target_label = (job_title or "").strip() or "Target"

    target_label = (job_title or "").strip() or "Target"

    radar_img = None
    if ENABLE_RADAR_PLOTS:
        radar_fig = _plot_radar_grid(history_list, target_vec, target_label)
        if radar_fig is not None:
            radar_img = _fig_to_png(radar_fig)
    show_radar = int(view_mode or 0) == 0
    radar_update = gr.update(value=radar_img)
    bar_update = gr.update(value=_plot_progress_bars(history_list, target_vec, target_label))
    radar_group_update = gr.update(visible=bool(radar_img) and show_radar)
    bar_group_update = gr.update(visible=bool(history_list) and not show_radar)
    table_html = _build_progress_table(history_list, target_vec, target_label)
    table_update = gr.update(value=table_html, visible=bool(table_html))

    if len(history_list) < 2:
        finish_attempt_timer()
        return (
            history_list,
            "Record another video to compare progress.",
            table_update,
            gr.update(visible=True),
            try_again_update,
            run_update,
            limit_update,
            radar_update,
            bar_update,
            gr.update(visible=True),
            radar_group_update,
            bar_group_update,
        )

    summary = ""
    try:
        attempt_label = history_list[-1]["attempt"] if history_list else "?"
        llm_start = time.perf_counter()
        print(f"[timer] llm_progress start (attempt {attempt_label})")
        summary = generate_progress_report(
            target_profession=target_label,
            target_trait_names=list(TARGET_TRAIT_NAMES),
            target_trait_scores=[float(v) for v in full_target],
            baseline_submit=history_list[-2],
            new_submits=[history_list[-1]],
            target_language=language or "English",
        )
        llm_elapsed = time.perf_counter() - llm_start
        print(f"[timer] llm_progress end (attempt {attempt_label}) total={llm_elapsed:.2f}s")
    except Exception:
        summary = ""
    finally:
        unload_model()

    if not summary:
        summary = _build_progress_summary(history_list, target_vec)
    summary = summary.replace("\n", "<br>")

    finish_attempt_timer()
    return (
        history_list,
        summary,
        table_update,
        gr.update(visible=True),
        try_again_update,
        run_update,
        limit_update,
        radar_update,
        bar_update,
        gr.update(visible=True),
        radar_group_update,
        bar_group_update,
    )
