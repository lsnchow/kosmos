"""Offline source-conformance inspection for production Octo versus AutoEval.

This module reads only supplied, already-pinned local checkouts.  It never
imports model frameworks, loads a checkpoint, creates an action fixture, or
declares qualification.  It confirms the source-level portions that can be
checked without a manifest-bound real proprio/history trajectory, and names
the remaining action comparison as blocked rather than manufacturing inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import AUTOEVAL_POLICY_SOURCE_COMMIT
from .octo import (
    OCTO_AUTOEVAL_OCTO_POLICY_SOURCE,
    OCTO_AUTOEVAL_STATIC_RNG_SEED,
    OCTO_PRODUCTION_SOURCE_COMMIT,
    OCTO_V1_0_OBSERVATION_KEYS,
)


AUTOEVAL_RUN_EVAL = "run_eval.py"
AUTOEVAL_POLICY = "auto_eval/robot/policy.py"
OCTO_GYM_WRAPPERS = "octo/utils/gym_wrappers.py"
OCTO_MODEL = "octo/model/octo_model.py"


class OctoAutoEvalConformanceError(RuntimeError):
    """A supplied source checkout cannot support an honest conformance read."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _clean_checkout(root: Path, expected_commit: str, label: str) -> Dict[str, str]:
    root = root.resolve()
    if not (root / ".git").is_dir():
        raise OctoAutoEvalConformanceError("%s must be a git checkout: %s" % (label, root))
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OctoAutoEvalConformanceError("could not inspect %s" % label) from error
    if head != expected_commit or dirty:
        raise OctoAutoEvalConformanceError("%s must be clean at %s" % (label, expected_commit))
    return {"path": str(root), "revision": head}


def _source_text(root: Path, relative: str, label: str) -> Tuple[str, Dict[str, str]]:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise OctoAutoEvalConformanceError("%s source path escapes its checkout" % label) from error
    if not path.is_file():
        raise OctoAutoEvalConformanceError("missing %s source: %s" % (label, path))
    return path.read_text(encoding="utf-8"), {"path": str(path), "sha256": _sha256(path)}


def _required_source_fragments(
    *,
    policy_text: str,
    run_eval_text: str,
    wrappers_text: str,
    model_text: str,
) -> Dict[str, bool]:
    """Check literal behaviors that define this pinned source profile."""

    return {
        "autoeval_policy_requires_image_and_proprio": 'assert all(key in obs_dict.keys() for key in ["image_primary", "proprio"])' in policy_text,
        "autoeval_policy_caches_first_language_task": "if self.task is None:" in policy_text and "self.task = self.agent.create_tasks(texts=[language_instruction])" in policy_text,
        "autoeval_policy_static_prngkey_0": "rng=jax.random.PRNGKey(0)" in policy_text,
        "autoeval_policy_requests_native_unnormalization": 'unnormalization_statistics=self.agent.dataset_statistics["bridge_dataset"]' in policy_text,
        "autoeval_run_uses_history_horizon_2": "env = HistoryWrapper(env, horizon=2)" in run_eval_text,
        "autoeval_run_uses_temporal_ensemble_horizon_4": "env = TemporalEnsembleWrapper(env, 4)" in run_eval_text,
        "autoeval_rollout_converts_pos_euler_state_to_proprio": "StateEncoding.POS_EULER" in run_eval_text and "env = ConvertState2Proprio(env)" in run_eval_text,
        "octo_history_reset_repeats_initial_observation": "self.history.extend([obs] * self.horizon)" in wrappers_text,
        "octo_history_first_mask_marks_old_slot_padding": "timestep_pad_mask[:pad_length] = 0" in wrappers_text,
        "octo_temporal_default_exp_weight_zero": "def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):" in wrappers_text,
        "octo_temporal_uses_oldest_to_newest_weight_order": "range(num_actions - 1, -1, -1)" in wrappers_text and "np.exp(-self.exp_weight * np.arange(num_actions))" in wrappers_text,
        "octo_model_defaults_to_timestep_pad_mask": 'timestep_pad_mask = observations["timestep_pad_mask"]' in model_text,
        "octo_model_native_normal_unnormalization_uses_mask": "(action * unnormalization_statistics[\"std\"])" in model_text and "jnp.ones_like(unnormalization_statistics[\"mean\"], dtype=bool)" in model_text,
    }


def inspect_pinned_sources(autoeval_checkout: Path, octo_checkout: Path) -> Dict[str, Any]:
    """Return hash-bound source findings without loading a runtime/model."""

    autoeval = _clean_checkout(autoeval_checkout, AUTOEVAL_POLICY_SOURCE_COMMIT, "AutoEval source")
    octo = _clean_checkout(octo_checkout, OCTO_PRODUCTION_SOURCE_COMMIT, "Octo source")
    policy_text, policy = _source_text(autoeval_checkout, AUTOEVAL_POLICY, "AutoEval OctoPolicy")
    run_eval_text, run_eval = _source_text(autoeval_checkout, AUTOEVAL_RUN_EVAL, "AutoEval rollout")
    wrappers_text, wrappers = _source_text(octo_checkout, OCTO_GYM_WRAPPERS, "Octo wrappers")
    model_text, model = _source_text(octo_checkout, OCTO_MODEL, "Octo model")
    findings = _required_source_fragments(
        policy_text=policy_text,
        run_eval_text=run_eval_text,
        wrappers_text=wrappers_text,
        model_text=model_text,
    )
    if not all(findings.values()):
        missing = sorted(name for name, present in findings.items() if not present)
        raise OctoAutoEvalConformanceError("pinned sources do not expose expected conformance fragments: %s" % ", ".join(missing))
    return {
        "schema": "plumb-octo-autoeval-source-conformance-v1",
        "qualified": False,
        "status": "source_checked_unqualified",
        "autoeval": {**autoeval, "policy": policy, "run_eval": run_eval},
        "octo": {**octo, "gym_wrappers": wrappers, "model": model},
        "findings": findings,
        "plumb_profile": {
            "observation_mask_key": OCTO_V1_0_OBSERVATION_KEYS.pad_mask_key,
            "autoeval_static_rng_seed": OCTO_AUTOEVAL_STATIC_RNG_SEED,
            "autoeval_policy_source": OCTO_AUTOEVAL_OCTO_POLICY_SOURCE,
        },
        "action_comparison": {
            "status": "blocked_missing_manifest_bound_real_proprio_history",
            "reason": (
                "The currently staged canonical PNG manifest binds RGB frames only. AutoEval OctoPolicy requires and "
                "forwards a history of ConvertState2Proprio output from a POS_EULER environment. Supplying raw LeRobot "
                "state, zeros, or a guessed conversion would fabricate a conformance fixture."
            ),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autoeval-source", type=Path, required=True, help="Existing clean AutoEval checkout at 3ea3...; no clone/download is attempted.")
    parser.add_argument("--octo-source", type=Path, required=True, help="Existing clean Octo checkout at 241fb...; no clone/download is attempted.")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(inspect_pinned_sources(args.autoeval_source, args.octo_source), sort_keys=True, indent=2))
    except Exception as error:
        print(json.dumps({"qualified": False, "status": "failed", "error": {"type": type(error).__name__, "message": str(error)}}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
