from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
_CACHE_DIR = Path(
    os.environ.get("MEPR_HF_CACHE_DIR", Path(__file__).resolve().parents[2] / "hf_cache")
)

_TOKENIZER = None
_MODEL = None
_MODEL_ID = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_id: str = DEFAULT_MODEL_ID):
    global _TOKENIZER, _MODEL, _MODEL_ID
    if _TOKENIZER is not None and _MODEL is not None and _MODEL_ID == model_id:
        return _TOKENIZER, _MODEL

    _TOKENIZER = AutoTokenizer.from_pretrained(model_id, cache_dir=_CACHE_DIR)
    dtype = torch.float16 if _get_device() == "cuda" else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=_CACHE_DIR,
    )
    _MODEL.to(_get_device())
    _MODEL.eval()
    _MODEL_ID = model_id
    return _TOKENIZER, _MODEL


def _parse_questions(text: str, max_items: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                items = [str(x).strip() for x in data if str(x).strip()]
                return items[:max_items]
        except Exception:
            pass

    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^\s*(!=:\d+[\).]|[-*]+)\s*", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines[:max_items]


def _ensure_paragraphs(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if "\n" in cleaned:
        return cleaned
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(parts) <= 1:
        return cleaned
    if len(parts) <= 3:
        return "\n\n".join(parts)
    n = len(parts)
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, 2 * n // 3)
    return "\n\n".join(
        [
            " ".join(parts[:cut1]),
            " ".join(parts[cut1:cut2]),
            " ".join(parts[cut2:]),
        ]
    )


def generate_questions(
    job_title: str,
    num_questions: int = 5,
    model_id: str = DEFAULT_MODEL_ID,
    language: str = "English",
) -> List[str]:
    tokenizer, model = _load_model(model_id=model_id)

    lang = (language or "English").strip().lower()
    lang_hint = "English" if lang.startswith("en") else "Russian"

    prompt = (
        "You are a hiring interviewer. Based on the job title, "
        f"generate {num_questions} interview questions. "
        "Focus on behavioral and skills-based questions. "
        f"Write in {lang_hint}. Return ONLY a JSON array of strings.\n\n"
        f"Job title: {job_title}"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return _parse_questions(content, max_items=int(num_questions))

def generate_recommendations(
    job_title: str,
    trait_scores: List[float],
    target_language: str,
    num_recommendations: int = 5,
    word_limit: int = 10,
    model_id: str = DEFAULT_MODEL_ID,
) -> List[str]:
    tokenizer, model = _load_model(model_id=model_id)

    prompt = (
        "A candidate is applying for a role in the profession listed below. They may or may not have direct experience in it. "
        f"Your task is to suggest {num_recommendations} simple, everyday conversation topics they can talk about to naturally reveal whether their personality aligns with the typical profile for this profession. "
        "You are given the profession name and a vector of 10 normalized trait scores (0.0-1.0): "
        "Openness, Conscientiousness, Extraversion, Agreeableness, Emotional Stability, "
        "Conversation, Openness to Change, Hedonism, Self-enhancement, Self-transcendence. "
        "Interpret the scores as follows: "
        "- >0.6 = strong tendency, 0.4-0.6 = moderate, <0.4 = weak. "
        f"Profession: {job_title}\\n"
        f"Trait scores: {trait_scores}\\n"
        f"Target language: {target_language}\\n"
        "Instructions:\\n"
        f"- Generate exactly {num_recommendations} short prompts (about {word_limit} words each) phrased as gentle suggestions: 'Tell about...', 'Describe...', 'Share...', etc.\\n"
        "- Each prompt must invite the candidate to talk about personal experiences, preferences, habits, or values from everyday life, NOT job-specific tasks.\\n"
        "- BUT each prompt MUST be tailored to reflect how someone with this specific trait profile would behave or think in the context of THIS profession. For example:\\n"
        "   - If Openness is high, focus on curiosity, imagination, unconventional interests.\\n"
        "   - If Conscientiousness is high, focus on planning, responsibility, attention to detail.\\n"
        "   - If Extraversion is low, focus on preference for quiet, independent work.\\n"
        "   - If Self-transcendence is high, focus on helping others, ethical choices, meaning.\\n"
        "- Use the PROFESSION NAME to subtly shape the context. For 'College Director', think about leadership, mentoring, long-term vision; for 'Energy Analyst', think about precision, risk assessment, future trends, but DO NOT mention jargon or tools.\\n"
        "- Avoid any assumption of professional experience. Do NOT use terms like 'trading', 'coding', 'patients', 'students'. Instead, ask about underlying traits that make someone good at this role.\\n"
        "- Cover diverse life domains: decision-making, social energy, routines, hobbies, coping, values, learning, leisure, ethics, future goals, relationships, etc.\\n"
        "- Keep language natural, simple, and accessible to anyone.\\n"
        f"- Return ONLY a JSON array of {num_recommendations} strings in the target language.\\n"
        "- Do NOT include explanations, numbering, or any extra text."
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return _parse_questions(content, max_items=int(num_recommendations))

def generate_explanation(
    target_profession: str,
    target_traits: list,
    candidate_big5: list,
    predicted_emotion: str,
    emotion_confidence: float,
    candidate_text: str,
    similarity_score: float,
    threshold: float,
    best_match_profession: str,
    best_match_similarity: float,
    target_language: str,
    words_limit: int=200,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    tokenizer, model = _load_model(model_id=model_id)

    target_traits_pct = [round(float(v) * 100.0, 1) for v in target_traits]
    candidate_big5_pct = [round(float(v) * 100.0, 1) for v in candidate_big5]
    similarity_pct = float(similarity_score) * 100.0
    best_match_similarity_pct = float(best_match_similarity) * 100.0
    threshold_pct = float(threshold) * 100.0

    prompt = (
        "You are an AI career advisor. Analyze the candidate's match to their desired profession using the data below, "
        "and generate a concise, constructive response in the target language. "

        "=== INPUT DATA ===\n"
        f"Target profession: {target_profession}\n"
        f"Target trait profile (10 traits, 0-100%): {target_traits_pct}\n"
        f"Candidate's predicted Big Five scores (0-100%): {candidate_big5_pct}\n"
        f"Predicted dominant emotion: {predicted_emotion} ({emotion_confidence:.1f}%)\n"
        f"Candidate's spoken text: \"{candidate_text}\"\n"
        f"Similarity to target: {similarity_pct:.1f}%\n"
        f"Good-fit threshold: {threshold_pct:.0f}%\n"
        f"Best-matching profession: {best_match_profession} (similarity: {best_match_similarity_pct:.1f}%)\n"
        f"Target language: {target_language}\n"

        "=== STRICT INSTRUCTIONS ===\n"
        "1. Start with: 'You are a strong match', 'You partially match', or 'You currently do not match' the target profession.\n"
        "2. If similarity < threshold, state the numeric similarity and threshold as percentages (e.g., 'similarity is 64.5%, below the 90% threshold').\n"
        "3. If the dominant emotion is Anger, Fear, Sadness, Disgust, or Surprise, say it may distort the assessment and advise recording in a calm, neutral state.\n"
        "4. Give 2-3 concrete, natural topics to discuss in a new video that reflect traits the profession requires but the candidate currently under-expresses (e.g., for high Extraversion: 'Describe how you energize a team').\n"
        "5. If similarity < threshold, add: 'Your profile currently aligns more closely with the role of [Best-Matching Profession].'\n"
        f"6. Keep the entire response under {words_limit} words.\n"
        "7. Write in the target language, but NEVER translate: profession names (e.g., 'College Director'), trait names (e.g., 'Openness', 'Extraversion'), or emotion names (e.g., 'Anger').\n"
        "8. Return ONLY the response text. No labels, explanations, or formatting."
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content.strip()

def generate_explanation_v2(
    target_profession: str,
    target_trait_names: list,          # ['Openness', 'Conscientiousness', ..., 'Self-transcendence'] (10)
    target_trait_scores: list,         # [0.77, 0.55, ..., 0.17] (10 floats)
    predicted_trait_names: list,       # ['Openness', ..., 'Non-Neuroticism'] (5)
    predicted_trait_scores: list,      # [0.52, 0.36, ..., 0.44] (5 floats)
    emotion_names: list,               # ['Neutral', 'Anger', ..., 'Surprise'] (7)
    emotion_probs: list,               # [0.35, 0.49, ...] (7 floats)
    multimodal_summary: str,           # "Predicted emotion: Anger (49.3%)..."
    transcription: str,
    body_position: str,                # e.g., 	"left", "center", "right"
    distance_to_camera: str,           # e.g., "far", "normal", "close"
    framing_note: str,
    lighting: str,                     # e.g., "low", "normal", "overexposed"
    lighting_note: str,
    audio_loudness: str,               # e.g., "quiet", "normal", "loud"
    audio_noise: str,                  # e.g., "clean", "moderate", "noisy"
    audio_note: str,
    best_match_profession: str,
    best_match_similarity: float,
    target_similarity: float,
    threshold: float = 0.8,
    target_language: str = "English",
    words_limit: int = 400,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    tokenizer, model = _load_model(model_id=model_id)

    dominant_emotion_idx = int(np.argmax(emotion_probs))
    dominant_emotion = emotion_names[dominant_emotion_idx]
    dominant_prob = emotion_probs[dominant_emotion_idx]

    neutral_idx = emotion_names.index("Neutral")
    neutral_prob = emotion_probs[neutral_idx]

    target_profile = {k: round(float(v) * 100.0, 1) for k, v in zip(target_trait_names, target_trait_scores)}
    candidate_profile = {k: round(float(v) * 100.0, 1) for k, v in zip(predicted_trait_names, predicted_trait_scores)}
    emotion_probs_pct = [round(float(p) * 100.0, 1) for p in emotion_probs]
    similarity_pct = float(target_similarity) * 100.0
    best_match_similarity_pct = float(best_match_similarity) * 100.0
    threshold_pct = float(threshold) * 100.0

    prompt = (
        "You are an AI career advisor. Analyze the candidate's multimodal profile and give a concise, practical recommendation.\n\n"
        "=== INPUT DATA ===\n"
        f"Target profession: {target_profession}\n"
        f"Target personality profile (10 traits, 0-100%): {target_profile}\n"
        f"Candidate's predicted Big Five (5 traits, 0-100%): {candidate_profile}\n"
        f"Emotion probabilities in % (order: {emotion_names}): {emotion_probs_pct}\n"
        f"Dominant emotion: {dominant_emotion} ({dominant_prob:.1%})\n"
        f"Multimodal analysis summary: {multimodal_summary}\n"
        f"Spoken transcription: \"{transcription}\"\n"
        f"Body position in frame: {body_position}\n"
        f"Framing note: {framing_note}\n"
        f"Distance to camera: {distance_to_camera}\n"
        f"Lighting: {lighting}\n"
        f"Lighting note: {lighting_note}\n"
        f"Audio loudness: {audio_loudness}\n"
        f"Audio noise level: {audio_noise}\n"
        f"Audio note: {audio_note}\n"
        f"Similarity to target profession: {similarity_pct:.1f}%\n"
        f"Similarity to best-matching profession ('{best_match_profession}'): {best_match_similarity_pct:.1f}%\n"
        f"Good-fit threshold: {threshold_pct:.0f}%\n"
        f"Target language: {target_language}\n\n"

        "=== RESPONSE REQUIREMENTS ===\n"
        "1. Start with one sentence: 'You are a strong match', 'You partially match', or 'You currently do not match' the target profession.\n"
        "2. Mention the similarity to the target and whether it meets the threshold. Use percentages. If below, mention the best-matching profession.\n"
        "3. Briefly describe 2 strongest and 2 weakest Big Five traits relative to the target profile (use trait names as-is).\n"
        "4. Emotion check: if dominant emotion is not 'Neutral' AND dominant_prob > 0.4 AND neutral_prob < 0.4, warn it may distort the assessment and suggest re-recording in a calm state.\n"
        "5. Visual feedback: comment on framing and lighting. Use the framing_note. If distance_to_camera != 'normal', advise adjusting distance. If lighting != 'normal', use the lighting_note.\n"
        "6. Audio feedback: mention if audio is quiet or noisy using the audio_note.\n"
        "7. Give 2-3 natural topics to discuss in the next video that could better reveal required traits.\n"
        f"8. Keep total length under {words_limit} words.\n"
        f"9. Write in {target_language}. NEVER translate profession names, trait names, or emotion names.\n"
        "10. Use 2-3 short paragraphs separated by blank lines.\n"
        "11. Return only the response text. No headings, bullet points, or markdown."
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return _ensure_paragraphs(content)

def generate_progress_report(
    target_profession: str,
    target_trait_names: list,
    target_trait_scores: list,
    baseline_submit: dict,
    new_submits: list,  # list of dicts, each = one new attempt
    target_language: str = "English",
    words_limit: int = 350,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    tokenizer, model = _load_model(model_id=model_id)

    target_profile = {k: round(float(v) * 100.0, 1) for k, v in zip(target_trait_names, target_trait_scores)}

    baseline_attempt = int(baseline_submit.get("attempt", 1))
    baseline_traits = {
        k: round(float(v) * 100.0, 1)
        for k, v in zip(
            baseline_submit["predicted_trait_names"],
            baseline_submit["predicted_trait_scores"],
        )
    }
    baseline_emotion_idx = int(np.argmax(baseline_submit["emotion_probs"]))
    baseline_emotion = baseline_submit["emotion_names"][baseline_emotion_idx]
    baseline_emotion_prob = baseline_submit["emotion_probs"][baseline_emotion_idx]

    new_attempts = []
    for i, sub in enumerate(new_submits, 1):
        traits = {
            k: round(float(v) * 100.0, 1)
            for k, v in zip(sub["predicted_trait_names"], sub["predicted_trait_scores"])
        }
        emo_idx = int(np.argmax(sub["emotion_probs"]))
        emo = sub["emotion_names"][emo_idx]
        emo_prob = sub["emotion_probs"][emo_idx]
        top_prof = sub["top_professions"][0] if sub["top_professions"] else {"profession": "None", "similarity": 0.0}
        attempt_id = int(sub.get("attempt", i))
        new_attempts.append({
            "attempt": attempt_id,
            "traits": traits,
            "emotion": f"{emo} ({emo_prob:.1%})",
            "transcription": sub["transcription"],
            "body_position": sub["body_position"],
            "distance_to_camera": sub["distance_to_camera"],
            "lighting": sub["lighting"],
            "target_similarity": float(sub["target_similarity"]) * 100.0,
            "best_match_profession": top_prof["profession"],
            "best_match_similarity": float(top_prof["similarity"]) * 100.0,
        })

    prompt = (
        "You are an AI career coach. Compare the candidate's baseline video with one or more follow-up attempts. "
        "Analyze progress toward the target profession and provide clear, constructive feedback in the target language.\n\n"

        "=== TARGET PROFESSION ===\n"
        f"Role: {target_profession}\n"
        f"Required traits: {target_profile}\n\n"

        f"=== BASELINE SUBMIT (Attempt {baseline_attempt}) ===\n"
        f"Personality: {baseline_traits}\n"
        f"Dominant emotion: {baseline_emotion} ({baseline_emotion_prob:.1%})\n"
        f"Transcription: \"{baseline_submit['transcription']}\"\n"
        f"Frame: position={baseline_submit['body_position']}, distance={baseline_submit['distance_to_camera']}, lighting={baseline_submit['lighting']}\n"
        f"Similarity to target: {float(baseline_submit['target_similarity']) * 100.0:.1f}%\n\n"

        "=== NEW SUBMITS ===\n"
    )

    for att in new_attempts:
        prompt += (
            f"--- Attempt {att['attempt']} ---\n"
            f"Personality: {att['traits']}\n"
            f"Dominant emotion: {att['emotion']}\n"
            f"Transcription: \"{att['transcription']}\"\n"
            f"Frame: position={att['body_position']}, distance={att['distance_to_camera']}, lighting={att['lighting']}\n"
            f"Similarity to target: {att['target_similarity']:.1f}%\n"
            f"Best match: {att['best_match_profession']} ({att['best_match_similarity']:.1f}%)\n\n"
        )

    prompt += (
        f"=== INSTRUCTIONS ===\n"
        f"1. Compare baseline vs latest new submit (or best among new).\n"
        f"2. Highlight which traits moved CLOSER to the target profile (use percentages, e.g., 'Your Extraversion increased from 33% to 58%, closer to the required 72%').\n"
        f"3. Note if emotional state improved (e.g., less Anger, more Neutral).\n"
        f"4. Comment on improvements in framing (centered position, normal distance, good lighting).\n"
        f"5. State whether similarity to target profession increased, and if the candidate is now closer to a different role.\n"
        f"6. Give a clear verdict: 'You are on the right track', 'Some improvements, but key traits still lag', or 'Changes did not align with target - refocus'.\n"
        f"7. Keep under {words_limit} words.\n"
        f"8. Write in {target_language}.\n"
        f"9. NEVER translate: profession names (e.g., 'College Director'), trait names (e.g., 'Openness'), or emotion names (e.g., 'Neutral').\n"
        f"10. Return ONLY the response text."
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return _ensure_paragraphs(content)

def unload_model() -> None:
    global _TOKENIZER, _MODEL, _MODEL_ID
    _TOKENIZER = None
    _MODEL_ID = None
    if _MODEL is not None:
        try:
            del _MODEL
        except Exception:
            pass
    _MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
