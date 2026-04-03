from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.artifact_service import (
    EvalArtifacts,
    GenerationRunArtifacts,
    ProfileArtifacts,
    load_eval_artifacts,
    load_latest_generation_run,
    load_profile_artifacts,
    read_binary_file,
)
from app.services.pipeline_service import build_or_load_profile, load_available_users, run_generation_for_user
from app.services.viz_service import build_user_generation_figure


st.set_page_config(
    page_title="Gen4Rec Demo",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_available_users() -> list[str]:
    return load_available_users()


def _format_metric_value(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_profile_section(profile_artifacts: ProfileArtifacts) -> None:
    prompt = profile_artifacts.prompt
    summary = (prompt or {}).get("input_summary") or profile_artifacts.summary or {}
    if not prompt:
        st.info("No prompt artifact is available yet. Click `Load profile` to build or load it.")
        return

    st.subheader("LLM listener profile")
    st.write(prompt.get("profile_paragraph", ""))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Style keywords**")
        st.write(", ".join(prompt.get("style_keywords", [])) or "None")
        st.markdown("**Top genres**")
        st.write(", ".join(summary.get("top_genres", [])) or "None")
        st.markdown("**Top tags**")
        st.write(", ".join(summary.get("top_tags", [])) or "None")
    with right:
        st.markdown("**Representative artists**")
        st.write(", ".join(summary.get("representative_artists", [])) or "None")
        st.markdown("**Representative tracks**")
        representative_tracks = summary.get("representative_tracks", [])
        if representative_tracks:
            for item in representative_tracks:
                st.write(f"- {item.get('artist', 'Unknown')} - {item.get('song', 'Unknown')}")
        else:
            st.write("None")

    st.subheader("Recent listening snapshot")
    audio_profile = summary.get("audio_profile", {})
    metrics = st.columns(4)
    metrics[0].metric("Danceability", audio_profile.get("danceability_mean"))
    metrics[1].metric("Energy", audio_profile.get("energy_mean"))
    metrics[2].metric("Valence", audio_profile.get("valence_mean"))
    metrics[3].metric("Tempo", audio_profile.get("tempo_mean"))
    if summary.get("mood_summary"):
        st.caption("Mood summary: " + ", ".join(summary["mood_summary"]))

    if profile_artifacts.validation:
        with st.expander("Validation summary"):
            validation = profile_artifacts.validation
            if validation.get("human_readable_summary"):
                st.write(validation["human_readable_summary"])
            st.json(validation)


def _render_track_card(track, index: int) -> None:
    st.markdown(f"### {index}. {track.title}")
    left, right = st.columns([1, 2])
    with left:
        if track.cover_large_url or track.cover_url:
            st.image(track.cover_large_url or track.cover_url, width="stretch")
        else:
            st.caption("No cover image available.")
    with right:
        badges = []
        if track.is_selected:
            badges.append("selected")
        if track.call_index is not None:
            badges.append(f"call {track.call_index}")
        if track.variant_index is not None:
            badges.append(f"variant {track.variant_index}")
        if badges:
            st.caption(" | ".join(badges))

        if track.rerank_score is not None:
            st.metric("CLAP rerank score", f"{track.rerank_score:.4f}")
        if track.duration_seconds is not None:
            st.caption(f"Duration: {track.duration_seconds:.2f}s")

        audio_bytes = read_binary_file(track.path)
        if audio_bytes is not None:
            st.audio(audio_bytes, format="audio/mpeg")
        elif track.source_url:
            st.audio(track.source_url, format="audio/mpeg")
        else:
            st.warning("Audio file is missing.")

        if track.lyric_text:
            with st.expander("Lyrics / text companion"):
                st.text(track.lyric_text)

        with st.expander("Track details"):
            st.write(f"Local audio path: `{track.path}`")
            if track.metadata_path:
                st.write(f"Metadata path: `{track.metadata_path}`")
            if track.lyric_path:
                st.write(f"Lyrics path: `{track.lyric_path}`")
            if track.prompt:
                st.write("Prompt:")
                st.write(track.prompt)
            if track.style:
                st.write("Style:")
                st.write(track.style)


def _render_generation_section(run_artifacts: GenerationRunArtifacts | None) -> None:
    st.subheader("Generated tracks")
    if run_artifacts is None:
        st.info("No generation run found yet. Use the form below to generate tracks.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Run ID", run_artifacts.run_id)
    summary_cols[1].metric("Candidates", len(run_artifacts.tracks))
    selected_count = len([track for track in run_artifacts.tracks if track.is_selected])
    summary_cols[2].metric("Selected", selected_count)
    summary_cols[3].metric("Provider", run_artifacts.manifest.get("provider", "unknown"))

    if run_artifacts.prompt_input:
        with st.expander("Generation prompt input"):
            profile_paragraph = run_artifacts.prompt_input.get("profile_paragraph")
            style_keywords = run_artifacts.prompt_input.get("style_keywords") or []
            if profile_paragraph:
                st.markdown("**Profile paragraph used for generation**")
                st.write(profile_paragraph)
            if style_keywords:
                st.markdown("**Style keywords used**")
                st.write(", ".join(style_keywords))
            if run_artifacts.prompt_input_path:
                st.caption(f"Prompt input path: `{run_artifacts.prompt_input_path}`")
            st.json(run_artifacts.prompt_input)

    selected_tab, candidates_tab, report_tab = st.tabs(["Selected tracks", "All candidates", "Run report"])

    with selected_tab:
        selected_tracks = [track for track in run_artifacts.tracks if track.is_selected]
        if not selected_tracks:
            st.info("No selected tracks found in rerank output.")
        else:
            for idx, track in enumerate(selected_tracks, start=1):
                _render_track_card(track, idx)

    with candidates_tab:
        if not run_artifacts.tracks:
            st.info("No candidate tracks found.")
        else:
            for idx, track in enumerate(run_artifacts.tracks, start=1):
                _render_track_card(track, idx)

    with report_tab:
        if run_artifacts.report_markdown:
            st.markdown(run_artifacts.report_markdown)
        else:
            st.info("No markdown report found for this run.")
        if run_artifacts.prompt_input:
            with st.expander("Prompt input JSON"):
                st.json(run_artifacts.prompt_input)
        with st.expander("Manifest JSON"):
            st.json(run_artifacts.manifest)
        if run_artifacts.rerank:
            with st.expander("Rerank JSON"):
                st.json(run_artifacts.rerank)


def _render_saved_eval_summary(eval_artifacts: EvalArtifacts) -> None:
    if eval_artifacts.summary:
        run_summary = eval_artifacts.summary.get("run", {})
        panels = eval_artifacts.summary.get("metric_panels", {})
        personalization_panel = panels.get("personalization", {})
        reference_topk_value = personalization_panel.get("selected_reference_topk_mean_cosine_mean")
        if reference_topk_value is None:
            reference_topk_value = personalization_panel.get("selected_reference_topn_mean_cosine_mean")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Eval encoder", run_summary.get("encoder", "unknown"))
        summary_cols[1].metric("Eval candidates", run_summary.get("candidate_count"))
        summary_cols[2].metric("Eval selected", run_summary.get("selected_count"))
        summary_cols[3].metric("Recent-K", eval_artifacts.summary.get("reference_set", {}).get("recent_k"))

        personalization_col, diversity_col, risk_col = st.columns(3)
        with personalization_col:
            st.markdown("**Personalization**")
            st.metric(
                "User alignment",
                _format_metric_value(personalization_panel.get("selected_user_embedding_cosine_mean")),
            )
            st.metric(
                "Centroid alignment",
                _format_metric_value(personalization_panel.get("selected_recent_centroid_cosine_mean")),
            )
            st.metric(
                "Reference top-k",
                _format_metric_value(reference_topk_value),
            )

        with diversity_col:
            st.markdown("**Diversity**")
            st.metric(
                "Selected pairwise cosine",
                _format_metric_value(panels.get("diversity", {}).get("selected_mean_pairwise_cosine")),
            )
            st.metric(
                "Selected nearest-neighbor",
                _format_metric_value(panels.get("diversity", {}).get("selected_mean_nearest_neighbor_cosine")),
            )
            st.caption("Lower values usually mean the kept songs are less redundant.")

        with risk_col:
            st.markdown("**Risk**")
            st.metric(
                "Selected too-close count",
                _format_metric_value(panels.get("risk", {}).get("selected_too_close_to_reference_count")),
            )
            st.metric(
                "Candidate too-close count",
                _format_metric_value(panels.get("risk", {}).get("candidate_too_close_to_reference_count")),
            )
            st.caption("Tracks flagged here may be overly close to one reference song.")

        with st.expander("Eval summary JSON"):
            st.json(eval_artifacts.summary)

    if eval_artifacts.report_markdown:
        with st.expander("Eval report"):
            st.markdown(eval_artifacts.report_markdown)


def _render_visualization_section(user_id: str, run_artifacts: GenerationRunArtifacts | None) -> None:
    st.subheader("Embedding visualization")
    if run_artifacts is None:
        st.info("Generate or load a run first to visualize it in embedding space.")
        return

    eval_artifacts = load_eval_artifacts(user_id, run_artifacts.run_id)
    has_saved_plot = eval_artifacts.plot_path.exists()
    mode_options = ["saved eval plot", "live compute"]
    default_mode = "saved eval plot" if has_saved_plot else "live compute"
    mode_key = f"viz_mode::{user_id}::{run_artifacts.run_id}"
    selected_mode = st.radio(
        "Visualization mode",
        options=mode_options,
        horizontal=True,
        index=mode_options.index(default_mode),
        key=mode_key,
    )

    if selected_mode == "saved eval plot":
        if not has_saved_plot:
            st.info(
                "No saved eval plot was found for this run yet. "
                "Run eval with `--save-plot`, or switch to `live compute`."
            )
            return
        st.image(str(eval_artifacts.plot_path), width="stretch")
        st.caption(f"Saved eval plot: `{eval_artifacts.plot_path}`")
        _render_saved_eval_summary(eval_artifacts)
        return

    viz_state_key = f"show_viz::{user_id}::{run_artifacts.run_id}"
    viz_button_key = f"render_viz_button::{user_id}::{run_artifacts.run_id}"
    if st.button("Render embedding space", key=viz_button_key):
        st.session_state[viz_state_key] = True

    if not st.session_state.get(viz_state_key):
        st.caption("Rendering the plot loads CLAP and embeds recent listens plus generated tracks, so it is kept explicit.")
        return

    with st.spinner("Building embedding-space visualization..."):
        figure, plot_df, encoder_name = build_user_generation_figure(
            user_id=user_id,
            run_root=run_artifacts.run_root,
        )
    st.pyplot(figure, clear_figure=True)
    st.caption(f"Visualization encoder: {encoder_name}")
    st.dataframe(
        plot_df[["encoder", "group", "label", "rerank_score", "path", "x", "y"]],
        width="stretch",
    )


def main() -> None:
    st.title("Gen4Rec Streamlit Demo")
    st.write(
        "A local demo for user profile analysis, one-click Suno generation, reranking, "
        "automatic eval, and embedding-space visualization."
    )

    try:
        available_users = get_available_users()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not available_users:
        st.error("No users are available in `outputs/embeddings/music4all/user_ids.npy`.")
        st.stop()

    default_user = st.session_state.get("selected_user_id") or available_users[0]
    default_index = available_users.index(default_user) if default_user in available_users else 0
    with st.sidebar:
        st.header("Controls")
        selected_user = st.selectbox("Known user IDs", available_users, index=default_index)
        custom_user = st.text_input("Or enter a user_id", value=selected_user)
        user_id = custom_user.strip() or selected_user
        st.session_state["selected_user_id"] = user_id

        st.markdown("### Profile build")
        top_k = st.number_input("Retrieval top-k", min_value=5, max_value=100, value=20, step=5)
        top_n = st.number_input("Summary top-n", min_value=5, max_value=100, value=20, step=5)
        exclude_recent = st.checkbox("Exclude recently listened songs", value=True)
        openai_model = st.text_input("OpenAI model", value="gpt-4.1-mini")
        force_rebuild_profile = st.checkbox("Force rebuild profile artifacts", value=False)

        load_profile_clicked = st.button("Load profile")
        refresh_run_clicked = st.button("Refresh latest run")

    profile_artifacts = load_profile_artifacts(user_id)
    run_artifacts = load_latest_generation_run(user_id)

    if load_profile_clicked:
        with st.spinner("Building or loading profile artifacts..."):
            profile_artifacts = build_or_load_profile(
                user_id=user_id,
                top_k=int(top_k),
                top_n=int(top_n),
                exclude_recent=exclude_recent,
                openai_model=openai_model,
                force_rebuild=force_rebuild_profile,
            )
        st.success("Profile artifacts are ready.")

    if refresh_run_clicked:
        run_artifacts = load_latest_generation_run(user_id)
        if run_artifacts is None:
            st.warning("No generation run found for this user yet.")
        else:
            st.success(f"Loaded latest run: {run_artifacts.run_id}")

    profile_col, generation_col = st.columns([1, 1])
    with profile_col:
        _render_profile_section(profile_artifacts)
    with generation_col:
        st.subheader("Generation controls")
        with st.form("generation_form"):
            generation_model = st.text_input("Generation model", value="chirp-v4-5")
            num_calls = st.number_input("Number of API calls", min_value=1, max_value=10, value=5, step=1)
            max_concurrency = st.number_input("Max concurrency", min_value=1, max_value=5, value=2, step=1)
            negative_prompt = st.text_input("Negative prompt", value="")
            lyrics = st.text_area("Lyrics or timestamp cues", value="", height=120)
            tempo_hint_bpm = st.number_input("Tempo hint (BPM)", min_value=0, max_value=300, value=0, step=1)
            duration_hint_seconds = st.number_input("Duration hint (seconds)", min_value=0, max_value=600, value=0, step=1)
            rerank_top_k = st.number_input("Rerank keep top-k", min_value=1, max_value=10, value=2, step=1)
            diversity_threshold_text = st.text_input("Diversity threshold (optional)", value="")
            rerank_encoder = st.selectbox("Rerank encoder", ["auto", "finetuned", "zeroshot"], index=0)
            generate_clicked = st.form_submit_button("Generate songs")

        if generate_clicked:
            if not profile_artifacts.prompt:
                with st.spinner("No prompt artifact found. Building profile first..."):
                    profile_artifacts = build_or_load_profile(
                        user_id=user_id,
                        top_k=int(top_k),
                        top_n=int(top_n),
                        exclude_recent=exclude_recent,
                        openai_model=openai_model,
                        force_rebuild=False,
                    )

            try:
                parsed_diversity = float(diversity_threshold_text) if diversity_threshold_text.strip() else None
            except ValueError:
                st.error("Diversity threshold must be empty or a valid float.")
                st.stop()

            with st.spinner("Running generation, rerank, and eval..."):
                generation_result = run_generation_for_user(
                    user_id=user_id,
                    prompt_output=profile_artifacts.prompt,
                    generation_model=generation_model,
                    num_calls=int(num_calls),
                    max_concurrency=int(max_concurrency),
                    negative_prompt=negative_prompt.strip() or None,
                    lyrics=lyrics,
                    tempo_hint_bpm=int(tempo_hint_bpm) or None,
                    duration_hint_seconds=int(duration_hint_seconds) or None,
                    rerank_top_k=int(rerank_top_k),
                    rerank_diversity_threshold=parsed_diversity,
                    rerank_encoder=rerank_encoder,
                )
            run_artifacts = generation_result["run_artifacts"]
            st.success(f"Generation, rerank, and eval finished. Latest run: {run_artifacts.run_id}")

    _render_generation_section(run_artifacts)
    _render_visualization_section(user_id, run_artifacts)


if __name__ == "__main__":
    main()
