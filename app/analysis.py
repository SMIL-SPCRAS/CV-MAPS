from __future__ import annotations

import os
import re
import shutil
import time
from typing import List

import gradio as gr

from app.config import (
    DEFAULT_CHECKPOINT,
    DEMO_DIR,
    EMO_ORDER,
    EMO_ORDER_FOR_BARS,
    MATCH_SIMILARITY_THRESHOLD,
    PERS_COLORS,
    PERS_ORDER,
    TARGET_TRAIT_NAMES,
    VIDEO_EXTS,
)
from app.utils import (
    build_output_payload,
    choose_device,
    convert_webm_to_mp4,
    create_session_dir,
    ensure_dir,
    render_gallery_html,
    reset_attempt_timer,
    start_attempt_timer,
)
from core.attribution import (
    compute_contributions_from_result,
    format_contribution_summary,
    plot_emotion_probs_barchart,
    visualize_all_task_heatmaps,
)
from core.media_utils import extract_keyframes_from_result
from core.matching import get_profession_trait_vector, match_profession
from core.llm.qwen_client import generate_explanation, generate_explanation_v2, unload_model
from core.runtime import analyze_video_basic, get_multitask_pred_with_attribution

_RUN_STAGE_TIMER = {"start_ts": None, "attempt": None}


def run_basic_and_split_heatmap(
    video_path,
    checkpoint_path,
    device_choice,
    segment_length,
    output_dir,
    target_features,
    inputs_choice,
    uid_suffix: str = "",
):
    empty_txt = ""
    if isinstance(video_path, dict):
        video_path = video_path.get("name") or video_path.get("path")
    if video_path is None:
        return (
            {"error": "Please upload a video."},
            empty_txt,
            None,
            None,
            None,
            "Please upload a video.",
            "",
            "",
            "",
            None,
            0.0,
        )

    ckpt = (checkpoint_path or "").strip() or DEFAULT_CHECKPOINT
    dev = choose_device(device_choice)
    save_dir = ensure_dir(output_dir)

    if os.path.splitext(video_path)[1].lower() == ".webm":
        video_path = convert_webm_to_mp4(video_path)

    res_basic = analyze_video_basic(
        video_path=video_path,
        checkpoint_path=ckpt,
        segment_length=int(segment_length),
        device=dev,
        save_dir=save_dir,
    )
    video_duration = float(res_basic.get("video_duration_sec", 0.0))

    payload = build_output_payload(res_basic)
    audio_metrics = res_basic.get("audio_metrics", {})
    if isinstance(payload, dict) and isinstance(audio_metrics, dict) and audio_metrics:
        payload.update(audio_metrics)
    transcript = res_basic.get("transcript", "")
    emo_prob = list(map(float, res_basic.get("emotion_logits", [0.0] * len(EMO_ORDER))))
    per_prob = list(map(float, res_basic.get("personality_scores", [0.0] * len(PERS_ORDER))))

    osc = res_basic.get("oscilloscope_path")
    osc_path = osc if (osc and os.path.exists(osc)) else None
    if osc_path:
        osc_unique = os.path.join(save_dir, f"osc{uid_suffix}.jpg")
        try:
            shutil.copyfile(osc_path, osc_unique)
            osc_path = osc_unique
        except OSError:
            pass

    bars_png = os.path.join(save_dir, f"emo_bars{uid_suffix}.png")
    remap = [EMO_ORDER.index(lbl) for lbl in EMO_ORDER_FOR_BARS]
    emo_prob_for_bars = [emo_prob[i] for i in remap]

    emo_colors = {
        "Neutral": "#9CA3AF",
        "Anger": "#EF4444",
        "Disgust": "#10B981",
        "Fear": "#8B5CF6",
        "Happiness": "#F59E0B",
        "Sadness": "#3B82F6",
        "Surprise": "#EC4899",
    }
    palette = [emo_colors[lbl] for lbl in EMO_ORDER_FOR_BARS]

    plot_emotion_probs_barchart(
        probs=emo_prob_for_bars,
        labels=EMO_ORDER_FOR_BARS,
        out_path=bars_png,
        colors=palette,
        figsize=(8, 3.7),
        auto_ylim=False,
    )

    pers_bars_png = os.path.join(save_dir, f"pers_bars{uid_suffix}.png")
    plot_emotion_probs_barchart(
        probs=per_prob,
        labels=PERS_ORDER,
        out_path=pers_bars_png,
        title="Score (%)",
        colors=[PERS_COLORS[lbl] for lbl in PERS_ORDER],
        figsize=(8, 4),
        auto_ylim=True,
        ylim_margin=3.0,
    )

    res_attr = get_multitask_pred_with_attribution(
        video_path=video_path,
        checkpoint_path=ckpt,
        segment_length=int(segment_length),
        device=dev,
        save_dir=save_dir,
    )

    combined_heatmap = visualize_all_task_heatmaps(
        result=res_attr,
        name_video=os.path.basename(video_path),
        target_features=int(target_features),
        inputs=inputs_choice,
        out_dir=save_dir,
    )
    if combined_heatmap and os.path.exists(combined_heatmap):
        heatmap_unique = os.path.join(save_dir, f"heatmap{uid_suffix}.png")
        try:
            os.replace(combined_heatmap, heatmap_unique)
            combined_heatmap = heatmap_unique
        except OSError:
            pass

    contrib = compute_contributions_from_result(
        result=res_attr,
        inputs=inputs_choice,
        modality_order=["body", "face", "scene", "audio", "text"],
        target_features=int(target_features),
        visual_granularity="detailed",
    )
    explain_md = format_contribution_summary(contrib, emo_labels=EMO_ORDER)

    sample_dir = os.path.join(save_dir, "samples")
    frames_info = extract_keyframes_from_result(
        video_path=video_path,
        result=res_attr,
        out_dir=sample_dir,
        n_default=8,
    )
    metrics = frames_info.get("metrics", {})
    if isinstance(payload, dict) and metrics:
        payload.update(metrics)

    indices = frames_info.get("indices", [])
    body_paths = frames_info.get("body", [])
    face_paths = frames_info.get("face", [])
    scene_paths = frames_info.get("scene", [])

    captions = [f"Frame {idx}" for idx in indices]

    thumb_h = 120
    body_html = render_gallery_html(body_paths, captions, uid=f"body{uid_suffix}", thumb_h=thumb_h)
    face_html = render_gallery_html(face_paths, captions, uid=f"face{uid_suffix}", thumb_h=thumb_h)
    scene_html = render_gallery_html(scene_paths, captions, uid=f"scene{uid_suffix}", thumb_h=thumb_h)

    return (
        payload,
        transcript,
        osc_path,
        bars_png,
        combined_heatmap,
        explain_md,
        body_html,
        face_html,
        scene_html,
        pers_bars_png,
        video_duration,
    )


def list_demo_videos() -> List[str]:
    if not os.path.isdir(DEMO_DIR):
        return []

    files = []
    for name in os.listdir(DEMO_DIR):
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTS:
            files.append(os.path.join(DEMO_DIR, name))
    return sorted(files)


def load_demo_video(path: str):
    return path


def run_and_show(
    job_title,
    video_path,
    checkpoint_path,
    device_choice,
    segment_length,
    out_dir,
    target_features,
    inputs_choice,
    session_dir,
    history,
):
    base_dir = ensure_dir((out_dir or "").strip() or "outputs")
    if not session_dir or not os.path.isdir(session_dir):
        session_dir = create_session_dir(base_dir, max_sessions=5)
    attempt_id = len(history or []) + 1
    attempt_dir = os.path.join(session_dir, f"attempt_{attempt_id:02d}")
    if os.path.isdir(attempt_dir):
        existing = []
        for name in os.listdir(session_dir):
            if name.startswith("attempt_"):
                try:
                    existing.append(int(name.split("_", 1)[1]))
                except Exception:
                    continue
        attempt_id = (max(existing) + 1) if existing else 1
        attempt_dir = os.path.join(session_dir, f"attempt_{attempt_id:02d}")
    attempt_dir = ensure_dir(attempt_dir)
    uid_suffix = f"_a{attempt_id:02d}_{int(time.time() * 1000)}"
    start_attempt_timer(attempt_id)
    _RUN_STAGE_TIMER["start_ts"] = time.perf_counter()
    _RUN_STAGE_TIMER["attempt"] = attempt_id
    print(f"[timer] run_stage start (attempt {attempt_id})")

    start = time.time()
    result = run_basic_and_split_heatmap(
        video_path,
        checkpoint_path,
        device_choice,
        segment_length,
        attempt_dir,
        target_features,
        inputs_choice,
        uid_suffix,
    )
    *core_outputs, video_duration = result
    payload = core_outputs[0]
    transcript = core_outputs[1]
    explain_md = core_outputs[5]

    if isinstance(payload, dict) and payload.get("error"):
        _RUN_STAGE_TIMER["start_ts"] = None
        _RUN_STAGE_TIMER["attempt"] = None
        return (
            payload,
            payload,
            {},
            gr.update(value=payload["error"], visible=True),
            "",
            gr.update(visible=False),
            session_dir,
        )

    if isinstance(payload, dict):
        payload["transcript"] = transcript or ""
        payload["multimodal_summary"] = _strip_html(explain_md)
        user_scores = payload.get("personality", {})
        match_result = match_profession(
            user_scores=user_scores,
            job_title=job_title or "",
            threshold=MATCH_SIMILARITY_THRESHOLD,
        )
        payload["profession_match"] = match_result
        core_outputs[0] = payload

    elapsed = time.time() - start
    runtime_txt = f"Video duration: {video_duration:.1f} sec | Inference time: {elapsed:.1f} sec"
    osc_path = core_outputs[2]
    bars_png = core_outputs[3]
    heatmap = core_outputs[4]
    explain_md = core_outputs[5]
    body_html = core_outputs[6]
    face_html = core_outputs[7]
    scene_html = core_outputs[8]
    pers_bars_png = core_outputs[9]

    viz_state = {
        "transcript": transcript or "",
        "osc_path": osc_path,
        "bars_png": bars_png,
        "heatmap": heatmap,
        "explain_md": explain_md or "",
        "body_html": body_html or "",
        "face_html": face_html or "",
        "scene_html": scene_html or "",
        "pers_bars_png": pers_bars_png,
    }

    return (
        payload,
        payload,
        viz_state,
        gr.update(value="", visible=False),
        runtime_txt,
        gr.update(visible=True),
        session_dir,
    )


def generate_llm_recommendation(payload: dict, job_title: str, language: str):
    if not isinstance(payload, dict) or payload.get("error"):
        return payload, payload, ""

    llm_start = time.perf_counter()
    print("[timer] llm_recommendation start")
    llm_text = build_llm_explanation(
        payload=payload,
        job_title=job_title or "",
        language=language or "English",
    )
    llm_elapsed = time.perf_counter() - llm_start
    print(f"[timer] llm_recommendation end total={llm_elapsed:.2f}s")

    stage_start = _RUN_STAGE_TIMER.get("start_ts")
    attempt_id = _RUN_STAGE_TIMER.get("attempt")
    if stage_start:
        total_elapsed = time.perf_counter() - float(stage_start)
        label = f"attempt {attempt_id}" if attempt_id else "attempt"
        print(f"[timer] run_stage end ({label}) total={total_elapsed:.2f}s")
    _RUN_STAGE_TIMER["start_ts"] = None
    _RUN_STAGE_TIMER["attempt"] = None
    if llm_text:
        payload = dict(payload)
        payload["llm_explanation"] = llm_text
    elif isinstance(payload, dict):
        llm_text = "Could not generate recommendation. Please retry."
    unload_model()
    return payload, payload, llm_text or ""


def reset_analysis_only():
    reset_attempt_timer()
    _RUN_STAGE_TIMER["start_ts"] = None
    _RUN_STAGE_TIMER["attempt"] = None
    return (
        gr.update(value=None),
        {},
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        {},
        gr.update(value="", visible=False),
        "",
        gr.update(visible=False),
        {},
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def build_llm_explanation(payload: dict, job_title: str, language: str) -> str:
    emotion = payload.get("emotion", {})
    personality = payload.get("personality", {})
    match = payload.get("profession_match", {})

    try:
        match_prof = match.get("matched_profession") or {}
        match_source = match.get("match_source") or ""

        top_roles = match.get("top_professions", []) or []
        best_match_prof = ""
        best_match_similarity = 0.0
        if top_roles:
            best_match_prof = str(top_roles[0].get("profession", ""))
            best_match_similarity = float(top_roles[0].get("similarity", 0.0))
        elif match_prof:
            best_match_prof = str(match_prof.get("profession", ""))
            best_match_similarity = float(match_prof.get("similarity", 0.0))

        target_profession = (job_title or "").strip() or best_match_prof
        target_traits = get_profession_trait_vector(target_profession)
        if not target_traits and best_match_prof:
            target_traits = get_profession_trait_vector(best_match_prof)
        if not target_traits:
            target_traits = [0.0] * len(TARGET_TRAIT_NAMES)
        if len(target_traits) < len(TARGET_TRAIT_NAMES):
            target_traits = list(target_traits) + [0.0] * (len(TARGET_TRAIT_NAMES) - len(target_traits))

        candidate_big5 = [float(personality.get(k, 0.0)) for k in PERS_ORDER]
        transcription = str(payload.get("transcript", "") or "")

        predicted_emotion = str(emotion.get("top") or "")
        emotion_confidence = float(emotion.get("top_prob", 0.0)) * 100.0

        if match_source in {"title", "exact"}:
            target_similarity = float(match_prof.get("similarity", 0.0))
        else:
            target_similarity = 0.0

        threshold = float(match.get("threshold", 0.8))

        predicted_trait_names = list(PERS_ORDER)
        predicted_trait_scores = [float(personality.get(k, 0.0)) for k in PERS_ORDER]
        emotion_dist = emotion.get("distribution", {}) if isinstance(emotion, dict) else {}
        emotion_names = list(EMO_ORDER)
        emotion_probs = [float(emotion_dist.get(lbl, 0.0)) for lbl in emotion_names]

        body_position = str(payload.get("body_position", "unknown"))
        distance_to_camera = str(payload.get("distance_to_camera", "unknown"))
        framing_note = str(payload.get("framing_note", ""))
        lighting = str(payload.get("lighting", "unknown"))
        lighting_note = str(payload.get("lighting_note", ""))
        audio_loudness = str(payload.get("audio_loudness", "unknown"))
        audio_noise = str(payload.get("audio_noise", "unknown"))
        audio_note = str(payload.get("audio_note", ""))

        return generate_explanation_v2(
            target_profession=target_profession,
            target_trait_names=list(TARGET_TRAIT_NAMES),
            target_trait_scores=target_traits,
            predicted_trait_names=predicted_trait_names,
            predicted_trait_scores=predicted_trait_scores,
            emotion_names=emotion_names,
            emotion_probs=emotion_probs,
            multimodal_summary=str(payload.get("multimodal_summary", "")),
            transcription=transcription,
            body_position=body_position,
            distance_to_camera=distance_to_camera,
            framing_note=framing_note,
            lighting=lighting,
            lighting_note=lighting_note,
            audio_loudness=audio_loudness,
            audio_noise=audio_noise,
            audio_note=audio_note,
            best_match_profession=best_match_prof or target_profession,
            best_match_similarity=best_match_similarity,
            target_similarity=target_similarity,
            threshold=threshold,
            target_language=language or "English",
        )
    except Exception:
        return ""


def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\\s+", " ", cleaned)
    return cleaned.strip()
