"""Bounded Cosmos forward-dynamics action-grounding probe suite.

This module is deliberately *not* a policy evaluation loop. It reuses one
loaded local ``Cosmos3OmniPipeline`` for a fixed vendor fixture baseline and a
small set of controlled action interventions. Results are Gate-B evidence,
never qualified policy cells.

Default run (on an allocated GPU node after the local snapshot is present):

.. code-block:: bash

   python -m plumb.adapters.probe_suite \
     --model-path /scratch/.../models/nvidia--Cosmos3-Nano \
     --actions /scratch/.../fixtures/bridge_actions.json \
     --vision /scratch/.../fixtures/bridge_video.mp4 \
     --output-dir /scratch/.../evidence/cosmos-probes

The explicit defaults are 256 action-resolution tier, 5 fps, 30 inference
steps, guidance scale 1.0, and seed 0. Each case gets a JSON report and MP4;
one failed case does not prevent later cases from running.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import FeedbackMode, WorldRequest
from .smoke import (
    OFFICIAL_BRIDGE_ACTION_COUNT,
    _load_conditioning_first_frame,
    _provenance_context,
    _read_action_rows,
    _runtime_context,
    _write_visual_artifacts,
)
from .worlds import (
    COSMOS3_NANO_MODEL_REVISION,
    DIFFUSERS_COSMOS3_PIN,
    Cosmos3NanoDiffusersAdapter,
    Cosmos3NanoDiffusersProfile,
)


ActionRows = Tuple[Tuple[float, ...], ...]
ArtifactWriter = Callable[..., Dict[str, Any]]
TEMPORAL_VAE_NOTE = (
    "Pinned Diffusers AutoencoderKLWan uses iter_=1+(num_frame-1)//4 in _encode; an action chunk of 1 requests "
    "two frames but encodes one temporal latent, and the observed one-frame decode is therefore a source-consistent "
    "blocking limitation rather than a result to pad."
)
TEMPORAL_VAE_SOURCE = (
    "https://github.com/huggingface/diffusers/blob/a3e0b8ec235c27a6c17a21976daf7fd32d819d05/"
    "src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1133-L1158"
)


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    actions: ActionRows
    kind: str
    description: str
    first_action_fixed_against_baseline: bool = False


def _case_definitions(actions: ActionRows, include_suffix_perturbation: bool) -> Tuple[ProbeCase, ...]:
    """Create the bounded interventions from the one pinned 16x10 fixture."""

    if len(actions) != OFFICIAL_BRIDGE_ACTION_COUNT:
        raise ValueError("Probe suite only accepts the full 16-action vendor fixture before taking prefixes.")
    zeros = tuple(tuple(0.0 for _ in row) for row in actions)
    cases: List[ProbeCase] = [
        ProbeCase(
            "baseline-16", actions, "baseline",
            "Pinned 16-action vendor fixture, fixed seed; normal fixture smoke only.",
        ),
        ProbeCase(
            "baseline-repeat-16", actions, "fixed_seed_repeat",
            "Exact repeat of the 16-action baseline with the same condition and seed; reproducibility diagnostic.",
        ),
        ProbeCase(
            "native-prefix-1", actions[:1], "gate_b_probe_prefix",
            "One-action native-length probe; not a policy causal-feedback claim.",
        ),
        ProbeCase(
            "native-prefix-4", actions[:4], "gate_b_probe_prefix",
            "Four-action native-length probe; not a policy causal-feedback claim.",
        ),
        ProbeCase(
            "zero-encoded-vectors-16", zeros, "encoded_vector_stress",
            "All-zero 10-D encoded-vector stress test. Rot6d zero columns are off-manifold and checkpoint normalization "
            "boundary remains unresolved, so this is not a physical no-op control.",
        ),
    ]
    if include_suffix_perturbation:
        suffix_permuted = (actions[0],) + tuple(reversed(actions[1:]))
        cases.append(
            ProbeCase(
                "future-suffix-permuted-16", suffix_permuted, "gate_b_future_suffix_permutation",
                "First action fixed to baseline; raw action rows 2..16 are reversed. This keeps each source vector intact "
                "while testing suffix sensitivity; it is a single-fixture diagnostic only.",
                first_action_fixed_against_baseline=True,
            )
        )
    return tuple(cases)


def _case_request(case: ProbeCase, *, frame: Any, prompt: str, seed: int, profile_id: str, fps: float) -> WorldRequest:
    return WorldRequest(
        conditioning_image=frame,
        prompt=prompt,
        domain="bridge_orig_lerobot",
        compiled_actions=case.actions,
        nominal_control_timestamps=tuple((index + 1) / fps for index in range(len(case.actions))),
        seed=seed,
        compatibility_profile_id=profile_id,
        feedback_mode=FeedbackMode.UNQUALIFIED,
        request_id="probe-" + case.case_id,
    )


def _profile_context(adapter: Cosmos3NanoDiffusersAdapter) -> Dict[str, Any]:
    profile = adapter.profile
    return {
        "profile_id": profile.profile_id,
        "model_path": profile.local_model_path,
        "model_revision": profile.model_revision,
        "diffusers_revision": profile.diffusers_revision,
        "container_digest": profile.container_digest,
        "normalizer_revision": profile.normalizer_revision,
        "resolution_tier": profile.resolution_tier,
        "fps": profile.fps,
        "num_inference_steps": profile.num_inference_steps,
        "guidance_scale": profile.guidance_scale,
        "scheduler_flow_shift": profile.scheduler_flow_shift,
        "allowed_action_lengths": list(profile.allowed_action_lengths),
        "probe_action_lengths": list(profile.probe_action_lengths),
    }


def _raw_frame_metrics(reference_frames: Sequence[Any], candidate_frames: Sequence[Any]) -> Dict[str, Any]:
    """Compare generated arrays before MP4 encoding (never decode artifacts)."""

    try:
        import numpy as np
        from .smoke import _normalise_frame_to_uint8
    except ImportError as error:
        return {"status": "unavailable", "reason": str(error), "comparison_space": "raw_output_type_np"}
    if len(reference_frames) != len(candidate_frames):
        return {
            "status": "incomparable",
            "reason": "different raw frame counts",
            "reference_frame_count": len(reference_frames),
            "candidate_frame_count": len(candidate_frames),
            "comparison_space": "raw_output_type_np",
        }
    try:
        reference = [_normalise_frame_to_uint8(frame).astype(np.float32) for frame in reference_frames]
        candidate = [_normalise_frame_to_uint8(frame).astype(np.float32) for frame in candidate_frames]
        labels = {"conditioning": 0, "first_future": 1, "frame_4": 4, "last": len(reference) - 1}
        selected = {}
        for label, index in labels.items():
            if index >= len(reference):
                selected[label] = None
            elif reference[index].shape != candidate[index].shape:
                selected[label] = {"status": "incomparable", "reason": "frame shape mismatch"}
            else:
                selected[label] = float(np.mean(np.abs(reference[index] - candidate[index])))

        def motion(frames):
            if len(frames) < 2:
                return None
            return float(np.mean([np.mean(np.abs(frames[index] - frames[index - 1])) for index in range(1, len(frames))]))

        reference_motion = motion(reference)
        candidate_motion = motion(candidate)
        return {
            "status": "ok",
            "comparison_space": "raw_output_type_np_uint8_before_mp4_encode",
            "pixel_mae_0_to_255": selected,
            "mean_consecutive_raw_frame_mae_0_to_255": {
                "reference": reference_motion,
                "candidate": candidate_motion,
                "absolute_difference": (
                    abs(reference_motion - candidate_motion)
                    if reference_motion is not None and candidate_motion is not None
                    else None
                ),
            },
        }
    except Exception as error:
        return {"status": "incomparable", "reason": "%s: %s" % (type(error).__name__, error), "comparison_space": "raw_output_type_np"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case_report(
    case: ProbeCase,
    *,
    adapter: Cosmos3NanoDiffusersAdapter,
    result: Any,
    output_artifacts: Mapping[str, Any],
    provenance: Mapping[str, Any],
    seed: int,
    generate_call_index: int,
    condition_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "cosmos3_action_probe",
        "status": "completed",
        "qualification": "gate_b_probe_only_not_qualified",
        "case": {
            "id": case.case_id,
            "kind": case.kind,
            "description": case.description,
            "action_count": len(case.actions),
            "action_width": len(case.actions[0]),
            "first_action_fixed_against_baseline": case.first_action_fixed_against_baseline,
            "seed": seed,
            "generate_call_index": generate_call_index,
            "pipeline_reused": generate_call_index > 1,
            "adapter_backend_call_attempts": adapter.backend_call_attempts,
            "adapter_successful_backend_calls": adapter.successful_backend_calls,
        },
        "profile": _profile_context(adapter),
        "provenance": dict(provenance),
        "condition_artifact": dict(condition_artifact),
        "runtime": _runtime_context(adapter),
        "result": {
            "backend": result.backend,
            "returned_frame_count": len(result.frames),
            "future_frame_count": len(result.future_frames),
            "conditioning_frame_included": result.conditioning_frame_included,
            "nominal_frame_timestamps": list(result.nominal_frame_timestamps),
            "timing": result.timing.as_dict(),
            "metadata": dict(result.metadata),
            "output_artifacts": dict(output_artifacts),
        },
        "note": (
            "The adapter reused one local pipeline object. These interventions test world-model behavior under a fixed "
            "condition and seed; they do not execute OpenVLA or establish causal policy feedback."
        ),
    }


def _failure_case_report(
    case: ProbeCase,
    error: BaseException,
    *,
    adapter: Cosmos3NanoDiffusersAdapter,
    provenance: Mapping[str, Any],
    seed: int,
    generate_call_index: int,
    condition_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    report = {
        "schema_version": 1,
        "kind": "cosmos3_action_probe",
        "status": "failed",
        "qualification": "gate_b_probe_only_not_qualified",
        "case": {
            "id": case.case_id,
            "kind": case.kind,
            "action_count": len(case.actions),
            "action_width": len(case.actions[0]),
            "seed": seed,
            "generate_call_index": generate_call_index,
            "pipeline_reused": generate_call_index > 1,
            "adapter_backend_call_attempts": adapter.backend_call_attempts,
            "adapter_successful_backend_calls": adapter.successful_backend_calls,
        },
        "profile": _profile_context(adapter),
        "provenance": dict(provenance),
        "condition_artifact": dict(condition_artifact),
        "runtime": _runtime_context(adapter),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "note": "Failure is persisted and the bounded suite continued with later independent cases.",
    }
    if len(case.actions) == 1:
        report["known_temporal_vae_constraint"] = {
            "status": "blocking_capability_observed",
            "source": TEMPORAL_VAE_SOURCE,
            "note": TEMPORAL_VAE_NOTE,
            "response": "No frame padding or synthetic successor was applied; this case cannot establish one-tick feedback.",
        }
    return report


def run_cases(
    adapter: Cosmos3NanoDiffusersAdapter,
    cases: Sequence[ProbeCase],
    *,
    condition_frame: Any,
    prompt: str,
    seed: int,
    output_dir: Path,
    provenance: Mapping[str, Any],
    fps: float,
    frame_selection: str,
    artifact_writer: ArtifactWriter = _write_visual_artifacts,
) -> List[Dict[str, Any]]:
    """Run every bounded intervention, persisting failure reports and continuing."""

    output_dir.mkdir(parents=True, exist_ok=True)
    condition_artifacts = artifact_writer(
        [condition_frame],
        fps=fps,
        output_video=None,
        output_frame_dir=str(output_dir / "condition"),
        frame_selection="all",
    )
    condition_artifact = {"artifacts": condition_artifacts, "shared_across_cases": True}
    reports: List[Dict[str, Any]] = []
    report_paths: Dict[str, Path] = {}
    raw_frames: Dict[str, Tuple[Any, ...]] = {}
    for call_index, case in enumerate(cases, start=1):
        report_path = output_dir / (case.case_id + ".json")
        try:
            request = _case_request(
                case,
                frame=condition_frame,
                prompt=prompt,
                seed=seed,
                profile_id=adapter.profile.profile_id,
                fps=fps,
            )
            result = adapter.generate(request)
            artifacts = artifact_writer(
                result.frames,
                fps=fps,
                output_video=str(output_dir / (case.case_id + ".mp4")),
                output_frame_dir=str(output_dir / (case.case_id + "-frames")),
                frame_selection=frame_selection,
            )
            report = _case_report(
                case,
                adapter=adapter,
                result=result,
                output_artifacts=artifacts,
                provenance=provenance,
                seed=seed,
                generate_call_index=call_index,
                condition_artifact=condition_artifact,
            )
            raw_frames[case.case_id] = tuple(result.frames)
        except Exception as error:
            report = _failure_case_report(
                case,
                error,
                adapter=adapter,
                provenance=provenance,
                seed=seed,
                generate_call_index=call_index,
                condition_artifact=condition_artifact,
            )
        _write_json(report_path, report)
        reports.append(report)
        report_paths[case.case_id] = report_path

    baseline = raw_frames.get("baseline-16")
    if baseline is not None:
        for case_id in ("baseline-repeat-16", "future-suffix-permuted-16"):
            if case_id not in raw_frames:
                continue
            report = next(item for item in reports if item["case"]["id"] == case_id)
            report["raw_frame_comparison_to_baseline"] = _raw_frame_metrics(baseline, raw_frames[case_id])
            _write_json(report_paths[case_id], report)
    return reports


def suite_command(args: argparse.Namespace) -> int:
    action_path = Path(args.actions)
    vision_path = Path(args.vision)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance: Mapping[str, Any] = {}
    try:
        provenance = _provenance_context(action_path, vision_path, model_path, args.model_manifest)
        fixture_actions = _read_action_rows(action_path)
        action_evidence = provenance.get("action_fixture")
        if not isinstance(action_evidence, Mapping) or not action_evidence.get("pinned_sha256_matches"):
            raise ValueError("Probe suite requires the SHA-pinned official [16,10] Bridge action fixture.")
        if not vision_path.is_file():
            raise FileNotFoundError("Probe suite requires a local condition image/video fixture.")
        profile = Cosmos3NanoDiffusersProfile(
            profile_id=args.profile_id,
            local_model_path=args.model_path,
            model_revision=args.model_revision,
            diffusers_revision=args.diffusers_revision,
            container_digest=args.container_digest,
            normalizer_revision=args.normalizer_revision,
            resolution_tier=args.resolution_tier,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            allowed_action_lengths=(16,),
            probe_action_lengths=(1, 4),
        )
        adapter = Cosmos3NanoDiffusersAdapter(profile)
        frame = _load_conditioning_first_frame(vision_path)
        cases = _case_definitions(fixture_actions, args.include_suffix_perturbation)
        reports = run_cases(
            adapter,
            cases,
            condition_frame=frame,
            prompt=args.prompt,
            seed=args.seed,
            output_dir=output_dir,
            provenance=provenance,
            fps=args.fps,
            frame_selection=args.frame_selection,
        )
        suite = {
            "schema_version": 1,
            "kind": "cosmos3_action_probe_suite",
            "status": "completed_with_case_failures" if any(item["status"] == "failed" for item in reports) else "completed",
            "qualification": "gate_b_probe_only_not_qualified",
            "profile": _profile_context(adapter),
            "provenance": dict(provenance),
            "runtime": _runtime_context(adapter),
            "cases": [
                {"id": item["case"]["id"], "status": item["status"], "report": item["case"]["id"] + ".json"}
                for item in reports
            ],
            "raw_frame_comparisons": {
                item["case"]["id"]: item["raw_frame_comparison_to_baseline"]
                for item in reports
                if "raw_frame_comparison_to_baseline" in item
            },
            "note": (
                "Single-fixture diagnostic only. No policy loader was invoked; this suite must not be interpreted as "
                "an OpenVLA causal feedback test or as Gate-B qualification. Raw metrics compare output_type=np arrays "
                "before MP4 encoding, so video compression does not affect them."
            ),
        }
        _write_json(output_dir / "suite.json", suite)
        print(json.dumps(suite, indent=2, sort_keys=True))
        return 0 if suite["status"] == "completed" else 2
    except Exception as error:
        failure = {
            "schema_version": 1,
            "kind": "cosmos3_action_probe_suite",
            "status": "failed_before_cases",
            "qualification": "gate_b_probe_only_not_qualified",
            "provenance": dict(provenance),
            "runtime": _runtime_context(),
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        }
        _write_json(output_dir / "suite.json", failure)
        print("probe suite failed: %s (report: %s)" % (error, output_dir / "suite.json"), file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m plumb.adapters.probe_suite")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--actions", required=True, help="SHA-pinned official [16,10] Bridge fixture JSON.")
    parser.add_argument("--vision", required=True, help="Local condition image/video fixture.")
    parser.add_argument("--output-dir", required=True, help="Per-case MP4, PNG, JSON, and suite evidence directory.")
    parser.add_argument("--model-manifest", help="Pinned model download manifest; autodiscovered beside cluster evidence when omitted.")
    parser.add_argument("--profile-id", default="cosmos3-nano-diffusers-fd-probe-suite")
    parser.add_argument("--model-revision", default=COSMOS3_NANO_MODEL_REVISION)
    parser.add_argument("--diffusers-revision", default=DIFFUSERS_COSMOS3_PIN)
    parser.add_argument("--container-digest", default=None)
    parser.add_argument("--normalizer-revision", default=None)
    parser.add_argument("--prompt", default="Put the pot to the left of the purple item.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=5.0)
    # Keep this explicit. The separate one-off smoke currently uses 480 fixture
    # framing; this bounded suite intentionally starts its own 256 profile.
    parser.add_argument("--resolution-tier", type=int, choices=(256, 480, 704, 720), default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--frame-selection", choices=("first-last", "all"), default="first-last")
    parser.add_argument("--include-suffix-perturbation", action="store_true")
    parser.set_defaults(handler=suite_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
