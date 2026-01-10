from __future__ import annotations

from typing import List, Optional, Sequence
import time

import gradio as gr

from app.config import DEFAULT_NUM_QUESTIONS
from app.utils import reset_attempt_timer
from core.llm.qwen_client import generate_recommendations, unload_model
from core.matching import get_profession_trait_vector

PROFESSION_SEARCH_LIMIT = 25


def format_questions_markdown(questions: List[str]) -> str:
    if not questions:
        return ""
    lines = [f"{i + 1}. {q}" for i, q in enumerate(questions)]
    return "\n".join(lines)

def filter_professions(
    query: str,
    options: Sequence[str],
    current_choice: Optional[str] = None,
):
    q = (query or "").strip().lower()
    if not q:
        return gr.update(choices=[], value=None)

    matches = [p for p in options if q in p.lower()]
    if matches:
        prefix = [p for p in matches if p.lower().startswith(q)]
        rest = [p for p in matches if not p.lower().startswith(q)]
        matches = prefix + rest
    if len(matches) > PROFESSION_SEARCH_LIMIT:
        matches = matches[:PROFESSION_SEARCH_LIMIT]

    value = None
    if current_choice and current_choice in matches:
        value = current_choice
    return gr.update(choices=matches, value=value)


def select_profession(choice: str):
    if not choice:
        return gr.update()
    return gr.update(value=choice)


def generate_questions_ui(job_title: str, language: str):
    title = (job_title or "").strip()
    if not title:
        return (
            [],
            "",
            "Please select a profession from the list.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    trait_scores = get_profession_trait_vector(title)
    if not trait_scores:
        return (
            [],
            "",
            "Please select a profession from the suggestions.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    try:
        llm_start = time.perf_counter()
        print("[timer] llm_questions start")
        questions = generate_recommendations(
            job_title=title,
            trait_scores=trait_scores,
            target_language=language or "English",
            num_recommendations=DEFAULT_NUM_QUESTIONS,
        )
        llm_elapsed = time.perf_counter() - llm_start
        print(f"[timer] llm_questions end total={llm_elapsed:.2f}s")
    finally:
        unload_model()

    if not questions:
        return (
            [],
            "",
            "Could not generate questions. Please try again.",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    return (
        questions,
        format_questions_markdown(questions),
        "",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def regenerate_questions_ui(job_title: str, language: str):
    title = (job_title or "").strip()
    if not title:
        return (
            [],
            "",
            "Please select a profession from the list.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    trait_scores = get_profession_trait_vector(title)
    if not trait_scores:
        return (
            [],
            "",
            "Please select a profession from the suggestions.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    try:
        llm_start = time.perf_counter()
        print("[timer] llm_questions start (regen)")
        questions = generate_recommendations(
            job_title=title,
            trait_scores=trait_scores,
            target_language=language or "English",
            num_recommendations=DEFAULT_NUM_QUESTIONS,
        )
        llm_elapsed = time.perf_counter() - llm_start
        print(f"[timer] llm_questions end total={llm_elapsed:.2f}s (regen)")
    finally:
        unload_model()

    if not questions:
        return (
            [],
            "",
            "Could not generate questions. Please try again.",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    return (
        questions,
        format_questions_markdown(questions),
        "",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def ready_to_start():
    unload_model()
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(interactive=False),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False),
    )


def reset_session():
    unload_model()
    reset_attempt_timer()
    return (
        gr.update(value="English", interactive=True),
        gr.update(value="", interactive=True),
        gr.update(choices=[], value=None, interactive=True, visible=True),
        [],
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None),
        {},
        "",
        gr.update(visible=False),
        {},
        "",
        [],
        {},
        gr.update(value=0),
        gr.update(value="", visible=False),
        "",
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        "",
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )
