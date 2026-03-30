"""
Run the Suno generation add-on for a single user.

This script intentionally reuses an existing prompt JSON produced by the
current retrieval/profile/prompt pipeline without modifying `src/profile_prompt/`.
It converts that prompt output into a generation spec, calls the Suno backend,
and writes the downloaded audio plus a manifest and lightweight report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate.artifacts import build_artifact_paths
from src.generate.base import GenerationResult, GenerationSpec
from src.generate.open_source_stub import OpenSourceGeneratorStub
from src.generate.reporting import build_user_facing_profile, save_json, write_markdown_report
from src.generate.suno import SunoGenerator


def merge_generation_results(results: list[GenerationResult]) -> GenerationResult:
    if not results:
        raise ValueError("No generation results to merge.")

    merged_samples = []
    call_summaries = []
    for call_index, result in enumerate(results, start=1):
        for sample in result.samples:
            sample.call_index = call_index
            if sample.variant_index is None:
                sample.variant_index = len(merged_samples) + 1
            merged_samples.append(sample)
        call_summaries.append(
            {
                "call_index": call_index,
                "response_metadata": result.response_metadata,
                "sample_count": len(result.samples),
            }
        )

    base_result = results[0]
    return GenerationResult(
        provider=base_result.provider,
        model=base_result.model,
        prompt_used=base_result.prompt_used,
        negative_prompt_used=base_result.negative_prompt_used,
        request_payload={
            **base_result.request_payload,
            "num_calls": len(results),
        },
        response_metadata={
            "call_count": len(results),
            "variant_count": len(merged_samples),
            "calls": call_summaries,
        },
        samples=merged_samples,
    )


def build_generation_spec(
    prompt_output: dict,
    *,
    provider_target: str,
    prompt_version: str,
    negative_prompt: str | None,
    lyrics: str,
    tempo_hint_bpm: int | None,
    duration_hint_seconds: int | None,
) -> GenerationSpec:
    summary = dict(prompt_output.get("input_summary", {}))
    return GenerationSpec(
        schema_version="1.0",
        user_id=str(prompt_output["user_id"]),
        provider_target=provider_target,
        prompt_version=prompt_version,
        generation_prompt=str(prompt_output["suno_generation_prompt"]),
        negative_prompt=negative_prompt,
        style_keywords=[str(x) for x in prompt_output.get("style_keywords", [])],
        instrumentation=[],
        lyrics=lyrics,
        sections=[],
        tempo_hint_bpm=tempo_hint_bpm,
        duration_hint_seconds=duration_hint_seconds,
        profile_paragraph=str(prompt_output.get("profile_paragraph", "")),
        input_summary=summary,
    )


def select_generator(provider: str, model: str):
    normalized = provider.lower()
    if normalized in {"suno", "ace-suno", "ace_suno"}:
        return SunoGenerator(model=model)
    if normalized in {"open-source", "local", "open_source_stub"}:
        return OpenSourceGeneratorStub()
    raise ValueError(f"Unsupported provider: {provider}")


def _run_single_generation_call(
    *,
    provider: str,
    model: str,
    spec: GenerationSpec,
    call_dir: Path,
) -> GenerationResult:
    generator = select_generator(provider=provider, model=model)
    return generator.generate(spec, call_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Suno generation pipeline from an existing prompt JSON.")
    parser.add_argument("--prompt-json", required=True, help="Path to an existing prompt JSON from the profile-prompt stage.")
    parser.add_argument("--user-id", default=None, help="Optional explicit user ID override.")
    parser.add_argument("--provider", default="suno", help="Generation backend provider name.")
    parser.add_argument("--generation-model", default="chirp-v4-5", help="Hosted music model to call.")
    parser.add_argument("--num-calls", type=int, default=1, help="How many API calls to make for the same prompt. Each Suno call can return two variants.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum number of API calls to run in parallel (default `2`). Use a low value to avoid API throttling.",
    )
    parser.add_argument("--negative-prompt", default=None, help="Optional explicit negative prompt for generation.")
    parser.add_argument("--lyrics-file", default=None, help="Optional path to a text file containing lyrics or timestamp cues.")
    parser.add_argument("--tempo-hint-bpm", type=int, default=None, help="Optional BPM hint for the generation backend.")
    parser.add_argument("--duration-hint-seconds", type=int, default=None, help="Optional duration hint for generation.")
    parser.add_argument("--prompt-version", default="existing-profile-prompt-v1", help="Version string recorded in the generation spec.")
    args = parser.parse_args()

    prompt_output = json.loads(Path(args.prompt_json).read_text(encoding="utf-8"))
    user_id = args.user_id or str(prompt_output["user_id"])

    run_id, artifact_paths = build_artifact_paths(user_id=user_id, provider=args.provider)
    artifact_paths.ensure_directories()
    save_json(prompt_output, artifact_paths.prompt_input_json)
    user_profile = build_user_facing_profile(prompt_output)

    lyrics = ""
    if args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")

    spec = build_generation_spec(
        prompt_output,
        provider_target=args.provider,
        prompt_version=args.prompt_version,
        negative_prompt=args.negative_prompt,
        lyrics=lyrics,
        tempo_hint_bpm=args.tempo_hint_bpm,
        duration_hint_seconds=args.duration_hint_seconds,
    )
    save_json(spec.to_dict(), artifact_paths.generation_spec_json)

    total_calls = max(1, args.num_calls)
    max_concurrency = max(1, min(args.max_concurrency, total_calls))

    call_results_by_index: dict[int, GenerationResult] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_call_index = {}
        for call_index in range(1, total_calls + 1):
            call_dir = artifact_paths.audio_dir / f"call_{call_index:02d}"
            future = executor.submit(
                _run_single_generation_call,
                provider=args.provider,
                model=args.generation_model,
                spec=spec,
                call_dir=call_dir,
            )
            future_to_call_index[future] = call_index

        for future in as_completed(future_to_call_index):
            call_index = future_to_call_index[future]
            call_results_by_index[call_index] = future.result()

    call_results = [call_results_by_index[idx] for idx in sorted(call_results_by_index)]

    merged_result = merge_generation_results(call_results)
    candidate_audio_paths = [sample.path for sample in merged_result.samples]

    manifest = {
        "run_id": run_id,
        "user_id": user_id,
        "provider": args.provider,
        "generation_model": args.generation_model,
        "num_calls": total_calls,
        "max_concurrency": max_concurrency,
        "artifacts": artifact_paths.to_dict(),
        "generation_spec": spec.to_dict(),
        "candidate_audio_paths": candidate_audio_paths,
        "rerank_ready": {
            "user_id": user_id,
            "candidate_count": len(candidate_audio_paths),
            "next_inputs": {
                "user_id": user_id,
                "user_embedding": "load from outputs/embeddings/music4all/user_embeddings.npy",
                "generated_audio_file_paths": candidate_audio_paths,
            },
        },
        "result": merged_result.to_dict(),
    }
    save_json(manifest, artifact_paths.manifest_json)
    write_markdown_report(
        path=artifact_paths.report_md,
        run_id=run_id,
        user_id=user_id,
        provider=args.provider,
        prompt_output=prompt_output,
        spec=spec,
        result=merged_result,
    )

    print("Generation run completed.")
    print(f"Run ID: {run_id}")
    print(f"Generation spec: {artifact_paths.generation_spec_json}")
    print(f"Manifest: {artifact_paths.manifest_json}")
    print(f"Report: {artifact_paths.report_md}")


if __name__ == "__main__":
    main()
