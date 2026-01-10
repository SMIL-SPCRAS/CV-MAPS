from __future__ import annotations

import gradio as gr

from app.analysis import (
    generate_llm_recommendation,
    list_demo_videos,
    load_demo_video,
    reset_analysis_only,
    run_and_show,
)
from app.chat import (
    filter_professions,
    generate_questions_ui,
    ready_to_start,
    regenerate_questions_ui,
    reset_session,
    select_profession,
)
from app.config import DEFAULT_CHECKPOINT, SHOW_JSON_BLOCK
from app.progress import update_history
from core.matching import list_professions


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CV-MAPS", theme=gr.themes.Default()) as demo:
        with gr.Column(elem_id="page_wrap"):
            gr.Markdown("# CV-MAPS: CV - Multi-Agent Personality-aware System")

            demo_video_paths = list_demo_videos()
            profession_choices = list_professions()
            questions_state = gr.State([])
            payload_state = gr.State({})
            history_state = gr.State([])
            session_state = gr.State("")
            viz_state = gr.State({})
            profession_state = gr.State(profession_choices)
            progress_view_state = gr.State(0)

            with gr.Column(elem_id="question_panel"):
                gr.Markdown("## Select Language and Search Desired Profession")
                lang_dd = gr.Dropdown(
                    choices=["English", "Russian"],
                    value="English",
                    label="Language",
                )
                job_title_input = gr.Textbox(
                    label="Profession",
                    placeholder="Start typing to see suggestions...",
                )
                job_suggestions = gr.Radio(
                    choices=[],
                    label="Suggestions",
                    interactive=True,
                )
                questions_status = gr.Markdown("")
                questions_loader = gr.HTML(
                    "<div class='loader'>Generating interview recommendations...</div>",
                    visible=False,
                    elem_id="questions_loader",
                )
                questions_md = gr.Markdown("")
                with gr.Row():
                    btn_generate = gr.Button("Generate recommendations", variant="primary")
                    btn_regen = gr.Button("Regenerate recommendations", visible=False, elem_id="btn_regen")
                    btn_ready = gr.Button("Upload / record a video", visible=False, elem_id="btn_ready")
                    btn_reset = gr.Button("Restart session", variant="primary", visible=False, elem_id="btn_reset")

            gr.Markdown("", elem_classes=["divider"])

            with gr.Column(visible=False, elem_id="analysis_panel") as analysis_panel:
                gr.Markdown("## Upload / Record a Multimodal Self-Presentation")
                with gr.Row(elem_id="input_panel"):
                    with gr.Column(scale=4, min_width=480, elem_id="video_col"):
                        webcam_constraints = {
                            "width": {"ideal": 640},
                            "height": {"ideal": 360},
                            "frameRate": {"ideal": 24, "max": 30},
                        }
                        in_video = gr.Video(
                            label="Video",
                            elem_id="main_video",
                            include_audio=True,
                            height=420,
                            webcam_options=gr.WebcamOptions(
                                constraints=webcam_constraints,
                                mirror=False,
                            ),
                        )
                    with gr.Column(scale=2, min_width=220, elem_id="demo_col"):
                        if demo_video_paths:
                            with gr.Column(elem_id="demo_buttons"):
                                for idx, path in enumerate(demo_video_paths, start=1):
                                    label = f"Example {idx}"
                                    btn = gr.Button(label, variant="secondary", size="sm")
                                    btn.click(fn=load_demo_video, inputs=gr.State(path), outputs=in_video)
                        else:
                            gr.Markdown(
                                "No files found in the `demo_video` folder. "
                                "Create the folder next to `app.py` and put some videos there."
                            )

                in_ckpt = gr.Textbox(
                    label="Checkpoint path",
                    value=DEFAULT_CHECKPOINT,
                    visible=False,
                )
                in_device = gr.Dropdown(
                    label="Device",
                    choices=["auto (select automatically)", "cuda", "cpu"],
                    value="auto (select automatically)",
                    visible=False,
                )
                in_seglen = gr.Slider(
                    5,
                    60,
                    value=30,
                    step=1,
                    label="Segment length (sec.)",
                    visible=False,
                )
                in_outdir = gr.Textbox(
                    label="Output directory",
                    value="outputs",
                    visible=False,
                )
                in_targetfeat = gr.Slider(
                    8,
                    128,
                    value=16,
                    step=2,
                    label="target_features",
                    visible=False,
                )
                in_inputs = gr.Dropdown(
                    label="Attribution source",
                    choices=["features", "emotion_logits", "personality_scores"],
                    value="features",
                    visible=False,
                )

                with gr.Row(elem_id="run_row"):
                    btn_run = gr.Button("Evaluate emotions and personality traits", variant="primary", elem_id="run_btn")
                status_md = gr.Markdown("", elem_id="status_text")
                runtime_md = gr.Markdown("", elem_id="runtime_text")

                with gr.Column(visible=False, elem_id="analysis_outputs") as analysis_outputs:
                    out_json = gr.JSON(
                        label="Model outputs (JSON)",
                        elem_id="json_block",
                        visible=SHOW_JSON_BLOCK,
                    )

                    with gr.Accordion("Visualization of the results of emotion and personality trait assessment", open=False, elem_id="viz_panel"):
                        @gr.render(inputs=viz_state)
                        def render_visualization(viz):
                            if not isinstance(viz, dict) or not viz:
                                return

                            with gr.Column(elem_id="results_group"):
                                with gr.Row(elem_id="gallery_row"):
                                    with gr.Column(scale=1, min_width=0):
                                        gr.Markdown("### **Body - key gesture frames**")
                                        gr.HTML(value=viz.get("body_html", ""))
                                    with gr.Column(scale=1, min_width=0):
                                        gr.Markdown("### **Face - key expression frames**")
                                        gr.HTML(value=viz.get("face_html", ""))
                                    with gr.Column(scale=1, min_width=0):
                                        gr.Markdown("### **Scene - key context frames**")
                                        gr.HTML(value=viz.get("scene_html", ""))

                                gr.Markdown("", elem_classes=["divider"])
                                with gr.Row(elem_id="audio_bars_row"):
                                    with gr.Column(scale=2, min_width=0, elem_id="audio_text_col"):
                                        gr.Markdown("## Input audio and text data")
                                        gr.Markdown("<div align='center'><h4>Audio waveform</h4></div>")
                                        gr.Image(
                                            value=viz.get("osc_path"),
                                            label="Waveform",
                                            elem_id="osc_img",
                                            interactive=False,
                                            container=False,
                                            height=160,
                                        )
                                        gr.Markdown("<div align='center'><h4>Text transcript</h4></div>")
                                        gr.Textbox(
                                            value=viz.get("transcript", ""),
                                            label=None,
                                            show_label=False,
                                            lines=3,
                                            container=False,
                                            elem_id="transcript_box",
                                        )

                                    with gr.Column(scale=5, min_width=0, elem_id="bars_col"):
                                        gr.Markdown("## Prediction result")
                                        with gr.Row(elem_id="bars_row_inner"):
                                            with gr.Column(scale=1, min_width=0):
                                                gr.Markdown(
                                                    "<div align='center'><h4>Probability distribution of emotions</h4></div>"
                                                )
                                                gr.Image(
                                                    value=viz.get("bars_png"),
                                                    label="Emotion Probabilities (Bars)",
                                                    container=False,
                                                    height=200,
                                                    elem_id="emo_bar",
                                                )
                                            with gr.Column(scale=1, min_width=0):
                                                gr.Markdown(
                                                    "<div align='center'><h4>Personality Traits Scores</h4></div>"
                                                )
                                                gr.Image(
                                                    value=viz.get("pers_bars_png"),
                                                    label="Personality Traits Scores (Bars)",
                                                    container=False,
                                                    height=230,
                                                    elem_id="pers_bar",
                                                )

                                gr.Markdown("", elem_classes=["divider"])
                                gr.Markdown("## Visualization of Attention")
                                with gr.Row(elem_classes=["viz_row", "heat_row"]):
                                    gr.Image(
                                        value=viz.get("heatmap"),
                                        label=None,
                                        show_label=False,
                                        container=False,
                                        elem_classes=["heat_img"],
                                        height=240,
                                    )

                                gr.Markdown("", elem_classes=["divider"])
                                gr.Markdown("## Summary")
                                gr.Markdown(value=viz.get("explain_md", ""), label="Explanation")

                    gr.Markdown("", elem_classes=["divider"])
                    gr.Markdown("## Explanations for Improving Self-Presentation", elem_id="recommendation_title")
                    llm_loader = gr.HTML(
                        "<div class='loader'>Generating recommendations...</div>",
                        visible=False,
                        elem_id="llm_loader",
                    )
                    out_llm = gr.Markdown("", elem_id="llm_text")

                    gr.Markdown("", elem_classes=["divider"])
                    with gr.Column(visible=False, elem_id="progress_panel") as progress_panel:
                        gr.Markdown("## Improvement Dynamics and Further Recommendations")
                        progress_md = gr.Markdown("", elem_id="progress_md")
                        progress_loader = gr.HTML(
                            "<div class='loader'></div>",
                            visible=False,
                            elem_id="progress_loader",
                        )
                        with gr.Column(elem_id="progress_graph"):
                            with gr.Row(elem_id="progress_view_row") as progress_view_row:
                                btn_view_prev = gr.Button("<", size="sm", elem_id="btn_view_prev")
                                gr.HTML("<div class='progress_spacer'></div>", elem_id="progress_spacer")
                                btn_view_next = gr.Button(">", size="sm", elem_id="btn_view_next")
                            with gr.Group(visible=False, elem_id="progress_radar") as progress_radar:
                                radar_plot = gr.Image(label="Trait profile progress", elem_id="radar_plot")
                            with gr.Group(visible=False, elem_id="progress_bars") as progress_bars:
                                progress_bar = gr.Plot(label="Trait comparison", elem_id="progress_bar")
                        gr.Markdown("", elem_id="ocean_note", visible=False)
                        progress_table = gr.HTML("", elem_id="progress_table", visible=False)
                        with gr.Row(elem_id="progress_actions"):
                            btn_reset_bottom = gr.Button(
                                "Start new session",
                                variant="primary",
                                visible=False,
                                elem_id="btn_reset_bottom",
                            )
                            btn_try_again = gr.Button(
                                "Try another video",
                                variant="secondary",
                                visible=False,
                                elem_id="btn_try_again",
                            )
                        limit_md = gr.Markdown("", elem_id="limit_md", visible=False)

        btn_generate.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[questions_loader],
        ).then(
            fn=generate_questions_ui,
            inputs=[job_title_input, lang_dd],
            outputs=[
                questions_state,
                questions_md,
                questions_status,
                btn_ready,
                btn_regen,
                questions_loader,
                btn_generate,
            ],
            show_progress="hidden",
        )

        job_title_input.input(
            fn=filter_professions,
            inputs=[job_title_input, profession_state, job_suggestions],
            outputs=[job_suggestions],
        )
        job_title_input.change(
            fn=filter_professions,
            inputs=[job_title_input, profession_state, job_suggestions],
            outputs=[job_suggestions],
        )
        job_title_input.submit(
            fn=filter_professions,
            inputs=[job_title_input, profession_state, job_suggestions],
            outputs=[job_suggestions],
        )

        job_suggestions.change(
            fn=select_profession,
            inputs=[job_suggestions],
            outputs=[job_title_input],
        )

        btn_regen.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[questions_loader],
        ).then(
            fn=regenerate_questions_ui,
            inputs=[job_title_input, lang_dd],
            outputs=[
                questions_state,
                questions_md,
                questions_status,
                btn_ready,
                btn_regen,
                questions_loader,
                btn_generate,
            ],
            show_progress="hidden",
        )

        btn_ready.click(
            fn=ready_to_start,
            inputs=[],
            outputs=[
                analysis_panel,
                btn_generate,
                btn_ready,
                btn_regen,
                btn_reset,
                analysis_outputs,
                job_title_input,
                job_suggestions,
                lang_dd,
            ],
        )

        btn_reset.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                lang_dd,
                job_title_input,
                job_suggestions,
                questions_state,
                questions_md,
                questions_status,
                questions_loader,
                btn_generate,
                btn_ready,
                btn_regen,
                btn_reset,
                analysis_panel,
                analysis_outputs,
                in_video,
                out_json,
                out_llm,
                llm_loader,
                payload_state,
                session_state,
                history_state,
                viz_state,
                progress_view_state,
                status_md,
                runtime_md,
                limit_md,
                btn_run,
                progress_md,
                progress_table,
                progress_panel,
                btn_try_again,
                btn_reset_bottom,
                radar_plot,
                progress_bar,
                progress_radar,
                progress_bars,
            ],
        )

        btn_reset_bottom.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                lang_dd,
                job_title_input,
                job_suggestions,
                questions_state,
                questions_md,
                questions_status,
                questions_loader,
                btn_generate,
                btn_ready,
                btn_regen,
                btn_reset,
                analysis_panel,
                analysis_outputs,
                in_video,
                out_json,
                out_llm,
                llm_loader,
                payload_state,
                session_state,
                history_state,
                viz_state,
                progress_view_state,
                status_md,
                runtime_md,
                limit_md,
                btn_run,
                progress_md,
                progress_table,
                progress_panel,
                btn_try_again,
                btn_reset_bottom,
                radar_plot,
                progress_bar,
                progress_radar,
                progress_bars,
            ],
        )

        btn_run.click(
            fn=lambda: (
                gr.update(value="Processing...", visible=True),
                "",
                gr.update(visible=False),
                "",
                gr.update(visible=False),
                {},
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            inputs=[],
            outputs=[
                status_md,
                runtime_md,
                analysis_outputs,
                out_llm,
                llm_loader,
                viz_state,
                progress_panel,
                progress_radar,
                progress_bars,
            ],
        ).then(
            fn=run_and_show,
            inputs=[
                job_title_input,
                in_video,
                in_ckpt,
                in_device,
                in_seglen,
                in_outdir,
                in_targetfeat,
                in_inputs,
                session_state,
                history_state,
            ],
            outputs=[
                out_json,
                payload_state,
                viz_state,
                status_md,
                runtime_md,
                analysis_outputs,
                session_state,
            ],
        ).then(
            fn=lambda: (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(visible=False),
            ),
            inputs=[],
            outputs=[
                progress_panel,
                progress_radar,
                progress_bars,
                progress_md,
                progress_loader,
            ],
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[llm_loader],
        ).then(
            fn=generate_llm_recommendation,
            inputs=[payload_state, job_title_input, lang_dd],
            outputs=[out_json, payload_state, out_llm],
            show_progress="hidden",
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[llm_loader],
        ).then(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value="Generating progress report...", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),

            inputs=[],
            outputs=[
                progress_panel,
                progress_loader,
                progress_md,
                progress_radar,
                progress_bars,
                btn_try_again,
                progress_view_row,
                btn_view_prev,
                btn_view_next,
            ],
        ).then(
            fn=update_history,
            inputs=[payload_state, job_title_input, lang_dd, history_state, progress_view_state],
            outputs=[
                history_state,
                progress_md,
                progress_table,
                progress_panel,
                btn_try_again,
                btn_run,
                limit_md,
                radar_plot,
                progress_bar,
                btn_reset_bottom,
                progress_radar,
                progress_bars,
            ],
            show_progress="hidden",
        ).then(
            fn=lambda: (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            ),
            inputs=[],
            outputs=[progress_loader, progress_view_row, btn_view_prev, btn_view_next],
        )

        def _toggle_progress_view(current_choice: int):
            view_choice = 1 - int(current_choice or 0)
            show_radar = view_choice == 0
            return (
                view_choice,
                gr.update(visible=show_radar),
                gr.update(visible=not show_radar),
            )

        btn_view_prev.click(
            fn=_toggle_progress_view,
            inputs=[progress_view_state],
            outputs=[progress_view_state, progress_radar, progress_bars],
        )
        btn_view_next.click(
            fn=_toggle_progress_view,
            inputs=[progress_view_state],
            outputs=[progress_view_state, progress_radar, progress_bars],
        )

        btn_try_again.click(
            fn=reset_analysis_only,
            inputs=[],
            outputs=[
                in_video,
                out_json,
                out_llm,
                llm_loader,
                progress_loader,
                payload_state,
                status_md,
                runtime_md,
                analysis_outputs,
                viz_state,
                progress_table,
                progress_panel,
                progress_radar,
                progress_bars,
                progress_view_row,
            ],
        )

    return demo
