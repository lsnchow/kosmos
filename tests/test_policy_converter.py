"""Adversarial tests for the safe MiniVLA/OpenPiZero checkpoint converter.

No torch, no network, no checkpoints.  Torch and safetensors are injected through
:class:`cluster.convert_policy_checkpoint.ConverterRuntime`, and the process
environment through :class:`~cluster.convert_policy_checkpoint.Environment`, so
every refusal this tool exists for is exercised on a CPU planning host.

The cases that matter most are the ones that assert the converter refuses: an
unreviewed pickle global, a source-hash mismatch, an existing artifact, a missing
license acknowledgement, and credentials left in the environment.  The rest
assert the emitted report actually satisfies what ``plumb/policies/minivla.py``
and ``plumb/policies/openpizero.py`` require, rather than a schema invented here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import pytest

from cluster import convert_policy_checkpoint as converter
from plumb.policies.contracts import NativeWrapperEntryPoint
from plumb.policies.minivla import (
    KNOWN_LICENSE_STATUSES,
    MINIVLA_CHECKPOINT_FILE,
    MINIVLA_REFUSED_WEIGHT_SUFFIXES,
    MINIVLA_VQ_CHECKPOINT_FILE,
    MINIVLA_VQ_LICENSE_STATUS,
    MINIVLA_VQ_REDISTRIBUTION,
    ConvertedWeightArtifact,
)
from plumb.policies.native import OPEN_PI_ZERO_SOURCE_COMMIT
from plumb.policies.openpizero import (
    BRIDGE_8D_PASSTHROUGH_LAYOUT,
    OPEN_PI_ZERO_CHECKPOINT_FILE,
    PALIGEMMA_REQUIRED_FILES,
    OpenPiZeroPolicyProfile,
    PaliGemmaSupportFiles,
)


REVISION = "a" * 40
LOADER_REVISION = "d" * 40
SOURCE_BYTES = b"legacy pickle bytes, never deserialized in this test suite"


# -- injected torch ------------------------------------------------------------


class FakeTensor:
    """The three tensor properties the converter reads, and nothing else."""

    def __init__(self, count: int, dtype: str = "torch.bfloat16", element_size: int = 2) -> None:
        self.count = count
        self.dtype = dtype
        self._element_size = element_size
        self.detached = False

    def numel(self) -> int:
        return self.count

    def element_size(self) -> int:
        return self._element_size

    def detach(self) -> "FakeTensor":
        self.detached = True
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def contiguous(self) -> "FakeTensor":
        return self


class FakeSerialization:
    def __init__(self, unsafe_globals: Sequence[str]) -> None:
        self._unsafe_globals = tuple(unsafe_globals)
        self.enumerated: List[str] = []
        self.alias_pairs: Optional[List[Tuple[Any, str]]] = None

    def get_unsafe_globals_in_checkpoint(self, path: str) -> List[str]:
        self.enumerated.append(str(path))
        return list(self._unsafe_globals)

    @contextmanager
    def safe_globals(self, pairs: Sequence[Tuple[Any, str]]) -> Iterator[None]:
        self.alias_pairs = list(pairs)
        yield


class FakeTorch:
    __version__ = "2.5.1"

    def __init__(self, checkpoint: Any, unsafe_globals: Sequence[str] = ()) -> None:
        self.serialization = FakeSerialization(unsafe_globals)
        self.checkpoint = checkpoint
        self.load_calls: List[Dict[str, Any]] = []

    def load(self, path: str, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is not True:
            raise AssertionError("the converter must never load a pickle without weights_only=True")
        if self.serialization.alias_pairs is None:
            raise AssertionError("the converter must load inside torch.serialization.safe_globals")
        self.load_calls.append({"path": str(path), **kwargs})
        return self.checkpoint

    @staticmethod
    def is_tensor(value: Any) -> bool:
        return isinstance(value, FakeTensor)


class RecordingSaveFile:
    """Stands in for ``safetensors.torch.save_file`` and writes real bytes."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, tensors: Mapping[str, Any], path: str, metadata: Optional[Mapping[str, str]] = None) -> None:
        self.calls.append({"keys": sorted(tensors), "path": str(path), "metadata": dict(metadata or {})})
        for key, value in (metadata or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise AssertionError("safetensors metadata must be string to string")
        payload = json.dumps({"keys": sorted(tensors), "metadata": dict(metadata or {})}, sort_keys=True)
        Path(path).write_text(payload)


def minivla_checkpoint() -> Dict[str, Any]:
    """Nested per-submodule state dicts under ``model``, plus ignored bookkeeping."""

    return {
        "model": {
            "projector": {"fc.weight": FakeTensor(4), "fc.bias": FakeTensor(2)},
            "vision_backbone": {"patch.weight": FakeTensor(8, dtype="torch.float32", element_size=4)},
        },
        "epoch": 21,
        "run_id": "step-362500",
    }


def clean_environment(tmp_path: Path) -> converter.Environment:
    home = tmp_path / "isolated-home"
    home.mkdir(exist_ok=True)
    return converter.Environment(variables={"PATH": "/usr/bin", "LANG": "C"}, euid=1000, home=str(home))


def write_source(tmp_path: Path, name: str = "step-362500-epoch-21-loss=0.2259.pt") -> Path:
    path = tmp_path / name
    path.write_bytes(SOURCE_BYTES)
    return path


def source_digest() -> str:
    return hashlib.sha256(SOURCE_BYTES).hexdigest()


def request_from(argv: Sequence[str]) -> converter.ConversionRequest:
    parser = converter.build_parser()
    return converter.parse_request(parser, parser.parse_args(list(argv)))


def invoke(
    tmp_path: Path,
    argv: Sequence[str],
    *,
    checkpoint: Any = None,
    unsafe_globals: Sequence[str] = (),
    environment: Optional[converter.Environment] = None,
) -> Tuple[Dict[str, Any], FakeTorch, RecordingSaveFile]:
    torch = FakeTorch(minivla_checkpoint() if checkpoint is None else checkpoint, unsafe_globals)
    save_file = RecordingSaveFile()
    runtime = converter.ConverterRuntime(
        torch=torch, save_file=save_file, torch_version="2.5.1", safetensors_version="0.4.5"
    )
    summary = converter.run(
        request_from(argv),
        runtime_factory=lambda: runtime,
        environment=environment or clean_environment(tmp_path),
    )
    return summary, torch, save_file


def execute_argv(
    tmp_path: Path,
    policy: str = "minivla",
    *,
    source: Optional[Path] = None,
    output: Optional[Path] = None,
    extra: Sequence[str] = (),
) -> List[str]:
    source = source if source is not None else write_source(tmp_path)
    output = output if output is not None else tmp_path / "converted.safetensors"
    argv = [
        "--policy",
        policy,
        "--source",
        str(source),
        "--output",
        str(output),
        "--source-revision",
        REVISION,
        "--expected-source-sha256",
        source_digest(),
        "--acknowledge-unresolved-license",
        "--execute",
    ]
    return argv + list(extra)


# -- the plan is the single source of truth -------------------------------------


def test_converted_paths_match_the_adapter_constants() -> None:
    expected = {
        "minivla": MINIVLA_CHECKPOINT_FILE,
        "minivla_vq": MINIVLA_VQ_CHECKPOINT_FILE,
        "openpizero": OPEN_PI_ZERO_CHECKPOINT_FILE,
    }
    for entry in converter.policy_catalogue()["policies"]:
        assert entry["repo_path"] == expected[entry["policy"]]


def test_license_status_vocabulary_matches_the_minivla_adapter() -> None:
    assert converter.ADAPTER_KNOWN_LICENSE_STATUSES == KNOWN_LICENSE_STATUSES
    assert converter.ADAPTER_PROHIBITED_REDISTRIBUTION == MINIVLA_VQ_REDISTRIBUTION


def test_output_suffix_is_not_one_the_adapters_refuse() -> None:
    assert converter.OUTPUT_SUFFIX not in MINIVLA_REFUSED_WEIGHT_SUFFIXES


def test_module_never_mentions_an_unsafe_load() -> None:
    source = Path(converter.__file__).read_text()
    assert "weights_only=False" not in source
    assert "weights_only=True" in source


# -- globals: enumerate first, refuse by name ----------------------------------


def test_unapproved_global_is_rejected_and_named(tmp_path: Path) -> None:
    summary, torch, save_file = invoke(
        tmp_path, execute_argv(tmp_path), unsafe_globals=["robot_eval_logger.Logger", "wandb.sdk.wandb_run.Run"]
    )
    report = summary["report"]
    assert report["status"] == "failed"
    assert report["error"]["type"] == "ConversionRefusal"
    assert "robot_eval_logger.Logger" in report["error"]["message"]
    assert "wandb.sdk.wandb_run.Run" in report["error"]["message"]
    assert "stub unpickler is not a security boundary" in report["error"]["message"]
    assert report["deserialization"]["rejected_globals"] == [
        "robot_eval_logger.Logger",
        "wandb.sdk.wandb_run.Run",
    ]
    assert torch.load_calls == []
    assert save_file.calls == []
    assert not (tmp_path / "converted.safetensors").exists()


def test_globals_are_enumerated_before_any_load(tmp_path: Path) -> None:
    summary, torch, _ = invoke(tmp_path, execute_argv(tmp_path))
    assert summary["report"]["status"] == "completed"
    assert torch.serialization.enumerated == [str(write_source(tmp_path))]
    assert summary["report"]["deserialization"]["globals_enumerated_before_load"] is True
    assert torch.load_calls[0]["weights_only"] is True
    assert torch.load_calls[0]["map_location"] == "cpu"


def test_the_operator_may_not_approve_executable_machinery(tmp_path: Path) -> None:
    summary, torch, _ = invoke(
        tmp_path,
        execute_argv(tmp_path, extra=["--approve-global", "posix.system", "--reviewer", "someone"]),
        unsafe_globals=["posix.system"],
    )
    message = summary["report"]["error"]["message"]
    assert summary["report"]["status"] == "failed"
    assert "Refusing to approve pickle global posix.system" in message
    assert "can execute code or reach the host" in message
    assert torch.load_calls == []


def test_an_observed_never_approvable_global_refuses_the_checkpoint_outright(tmp_path: Path) -> None:
    summary, torch, save_file = invoke(
        tmp_path, execute_argv(tmp_path), unsafe_globals=["subprocess.Popen"]
    )
    report = summary["report"]
    message = report["error"]["message"]
    assert report["status"] == "failed"
    assert "subprocess.Popen" in message
    assert "--approve-global does not override this" in message
    assert report["deserialization"]["never_approvable_globals_observed"] == ["subprocess.Popen"]
    assert torch.load_calls == []
    assert save_file.calls == []


def test_operator_approval_requires_a_named_reviewer(tmp_path: Path) -> None:
    summary, _, _ = invoke(
        tmp_path,
        execute_argv(tmp_path, extra=["--approve-global", "somelib.Config"]),
        unsafe_globals=["somelib.Config"],
    )
    assert summary["report"]["status"] == "failed"
    assert "--reviewer" in summary["report"]["error"]["message"]


def test_approved_global_is_aliased_to_the_inert_recorder(tmp_path: Path) -> None:
    summary, torch, _ = invoke(
        tmp_path,
        execute_argv(tmp_path, extra=["--approve-global", "somelib.Config", "--reviewer", "k. valencia"]),
        unsafe_globals=["somelib.Config"],
    )
    report = summary["report"]
    assert report["status"] == "completed"
    assert report["deserialization"]["approved_globals_used"] == ["somelib.Config"]
    assert report["deserialization"]["inert_aliased_globals"] == ["somelib.Config"]
    assert report["deserialization"]["operator_reviewer"] == "k. valencia"
    assert torch.serialization.alias_pairs == [(converter.InertMetadata, "somelib.Config")]


def test_inert_recorder_runs_no_upstream_code() -> None:
    value = converter.InertMetadata("positional", keyword=1)
    value.__setstate__({"anything": True})
    assert value.args == ("positional",)
    assert value.kwargs == {"keyword": 1}
    assert value.state == {"anything": True}


def test_inert_builtin_containers_resolve_to_the_real_object() -> None:
    decision = converter.GlobalsDecision(
        observed=("collections.OrderedDict",),
        reviewed=("collections.OrderedDict", "builtins.int"),
        operator_approved=(),
        reviewer=None,
    )
    pairs = dict((name, target) for target, name in converter.safe_alias_pairs(decision))
    import collections

    assert pairs["collections.OrderedDict"] is collections.OrderedDict
    assert pairs["builtins.int"] is int
    assert converter.InertMetadata not in pairs.values()


def test_a_reviewed_allowlist_may_never_contain_executable_machinery() -> None:
    assert converter.never_approvable_reason("builtins.int") is None
    assert converter.never_approvable_reason("collections.OrderedDict") is None
    for name in ("os.system", "posix.system", "subprocess.Popen", "builtins.eval", "urllib.request.urlopen"):
        assert converter.never_approvable_reason(name) is not None
    with pytest.raises(ValueError):
        converter.PolicyTarget(
            policy="bad",
            asset_plan_name="minivla-vq-bridge",
            artifact_label="bad",
            adapter_fields_kind="minivla_converted_artifact",
            reviewed_globals=("os.system",),
            state_dict_keys=(),
            expectation="",
        )


# -- inspection ----------------------------------------------------------------


def test_inspect_reports_rejected_globals_without_refusing_and_writes_nothing(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    summary, torch, save_file = invoke(
        tmp_path,
        ["--policy", "minivla", "--source", str(source), "--inspect"],
        unsafe_globals=["robot_eval_logger.Logger"],
    )
    report = summary["report"]
    assert report["status"] == "inspected"
    assert report["error"] is None
    assert report["deserialization"]["observed_globals"] == ["robot_eval_logger.Logger"]
    assert report["deserialization"]["rejected_globals"] == ["robot_eval_logger.Logger"]
    assert report["output"] is None
    assert torch.load_calls == []
    assert save_file.calls == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["isolated-home", source.name]


def test_inspect_keeps_its_evidence_when_a_report_path_is_named(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    evidence = tmp_path / "minivla-inspection.json"
    summary, _, _ = invoke(
        tmp_path,
        ["--policy", "minivla", "--source", str(source), "--inspect", "--report", str(evidence)],
        unsafe_globals=["robot_eval_logger.Logger"],
    )
    assert summary["report"]["status"] == "inspected"
    assert summary["report_path"] == str(evidence)
    written = json.loads(evidence.read_text())
    assert written["status"] == "inspected"
    assert written["deserialization"]["rejected_globals"] == ["robot_eval_logger.Logger"]
    assert summary["conversion_report_sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()


def test_inspect_evidence_is_never_overwritten(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    evidence = tmp_path / "minivla-inspection.json"
    evidence.write_text("{}")
    summary, _, _ = invoke(
        tmp_path, ["--policy", "minivla", "--source", str(source), "--inspect", "--report", str(evidence)]
    )
    assert summary["report"]["status"] == "failed"
    assert "Refusing to overwrite" in summary["report"]["error"]["message"]
    assert evidence.read_text() == "{}"


def test_inspect_needs_no_hash_or_revision_but_records_the_blockers(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    summary, _, _ = invoke(tmp_path, ["--policy", "minivla_vq", "--source", str(source), "--inspect"])
    report = summary["report"]
    assert report["status"] == "inspected"
    assert report["source"]["sha256"] == source_digest()
    assert report["source"]["expected_sha256"] is None
    assert report["source"]["sha256_verified"] is False
    assert converter.MISSING_HASH_BLOCKER in report["verification_blockers"]
    assert converter.UNVERIFIED_SOURCE_SHA_BLOCKER in report["verification_blockers"]
    assert converter.UNRESOLVED_REVISION_BLOCKER in report["verification_blockers"]
    assert converter.LICENSE_ACKNOWLEDGEMENT_BLOCKER in report["verification_blockers"]


def test_a_verified_source_hash_clears_only_that_blocker(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path))
    blockers = summary["report"]["verification_blockers"]
    assert summary["report"]["status"] == "completed"
    assert converter.UNVERIFIED_SOURCE_SHA_BLOCKER not in blockers
    assert converter.UNRESOLVED_REVISION_BLOCKER not in blockers
    assert converter.MISSING_HASH_BLOCKER in blockers
    assert converter.LICENSE_ACKNOWLEDGEMENT_BLOCKER not in blockers


# -- source verification -------------------------------------------------------


def test_source_hash_mismatch_refuses(tmp_path: Path) -> None:
    summary, torch, save_file = invoke(
        tmp_path,
        [
            "--policy",
            "minivla",
            "--source",
            str(write_source(tmp_path)),
            "--output",
            str(tmp_path / "converted.safetensors"),
            "--source-revision",
            REVISION,
            "--expected-source-sha256",
            "b" * 64,
            "--acknowledge-unresolved-license",
            "--execute",
        ],
    )
    report = summary["report"]
    assert report["status"] == "failed"
    assert "Source checksum mismatch" in report["error"]["message"]
    assert source_digest() in report["error"]["message"]
    assert report["source"]["sha256_verified"] is False
    assert torch.load_calls == []
    assert save_file.calls == []
    assert not (tmp_path / "converted.safetensors").exists()


def test_execute_refuses_without_an_expected_source_hash(tmp_path: Path) -> None:
    summary, _, _ = invoke(
        tmp_path,
        [
            "--policy",
            "minivla",
            "--source",
            str(write_source(tmp_path)),
            "--output",
            str(tmp_path / "converted.safetensors"),
            "--source-revision",
            REVISION,
            "--acknowledge-unresolved-license",
            "--execute",
        ],
    )
    message = summary["report"]["error"]["message"]
    assert summary["report"]["status"] == "failed"
    assert "pins no SHA-256" in message
    assert "will not fabricate one" in message


def test_execute_refuses_without_an_immutable_source_revision(tmp_path: Path) -> None:
    summary, _, _ = invoke(
        tmp_path,
        [
            "--policy",
            "minivla",
            "--source",
            str(write_source(tmp_path)),
            "--output",
            str(tmp_path / "converted.safetensors"),
            "--expected-source-sha256",
            source_digest(),
            "--acknowledge-unresolved-license",
            "--execute",
        ],
    )
    assert summary["report"]["status"] == "failed"
    assert converter.UNRESOLVED_REVISION_BLOCKER in summary["report"]["error"]["message"]


def test_a_renamed_source_is_recorded_as_a_blocker(tmp_path: Path) -> None:
    source = write_source(tmp_path, name="minivla-checkpoint.pt")
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path, source=source))
    report = summary["report"]
    assert report["status"] == "completed"
    assert report["source"]["basename_matches_asset_plan"] is False
    assert converter.SOURCE_NAME_BLOCKER in report["verification_blockers"]


# -- refuse to overwrite -------------------------------------------------------


def test_an_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "converted.safetensors"
    output.write_text("an earlier artifact whose provenance nobody has checked")
    summary, torch, save_file = invoke(tmp_path, execute_argv(tmp_path, output=output))
    assert summary["report"]["status"] == "failed"
    assert "Refusing to overwrite" in summary["report"]["error"]["message"]
    assert output.read_text() == "an earlier artifact whose provenance nobody has checked"
    assert torch.load_calls == []
    assert save_file.calls == []


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "converted.safetensors"
    report_path = output.with_suffix(".conversion-report.json")
    report_path.write_text("{}")
    summary, _, save_file = invoke(tmp_path, execute_argv(tmp_path, output=output))
    assert summary["report"]["status"] == "failed"
    assert "Refusing to overwrite" in summary["report"]["error"]["message"]
    assert report_path.read_text() == "{}"
    assert save_file.calls == []
    assert not output.exists()


def test_a_failed_execute_leaves_the_canonical_names_free_and_keeps_evidence(tmp_path: Path) -> None:
    output = tmp_path / "converted.safetensors"
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path, output=output), unsafe_globals=["wandb.Run"])
    assert summary["report"]["status"] == "failed"
    assert not output.exists()
    assert not output.with_suffix(".conversion-report.json").exists()
    refusals = sorted(tmp_path.glob("converted.conversion-report.refused-*.json"))
    assert len(refusals) == 1
    assert json.loads(refusals[0].read_text())["error"]["type"] == "ConversionRefusal"
    assert summary["refusal_report_path"] == str(refusals[0])


def test_output_must_be_safetensors(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path, output=tmp_path / "converted.pt"))
    assert summary["report"]["status"] == "failed"
    assert ".safetensors" in summary["report"]["error"]["message"]
    assert not (tmp_path / "converted.pt").exists()


# -- dry run -------------------------------------------------------------------


def test_dry_run_is_the_default_and_writes_nothing(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    output = tmp_path / "converted.safetensors"
    argv = [argument for argument in execute_argv(tmp_path, source=source, output=output) if argument != "--execute"]
    summary, torch, save_file = invoke(tmp_path, argv)
    report = summary["report"]
    assert report["status"] == "dry_run"
    assert report["mode"] == "dry_run"
    assert report["output"] is None
    assert torch.load_calls == []
    assert save_file.calls == []
    assert not output.exists()
    assert not output.with_suffix(".conversion-report.json").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["isolated-home", source.name]


def test_a_refused_dry_run_still_writes_nothing(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    argv = [argument for argument in execute_argv(tmp_path, source=source) if argument != "--execute"]
    summary, _, save_file = invoke(tmp_path, argv, unsafe_globals=["wandb.Run"])
    assert summary["report"]["status"] == "failed"
    assert save_file.calls == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["isolated-home", source.name]


def test_dry_run_applies_the_same_preconditions_as_execute(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    summary, _, _ = invoke(
        tmp_path,
        [
            "--policy",
            "minivla_vq",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "converted.safetensors"),
            "--source-revision",
            REVISION,
            "--expected-source-sha256",
            source_digest(),
        ],
    )
    assert summary["report"]["status"] == "failed"
    assert "--acknowledge-unresolved-license" in summary["report"]["error"]["message"]


# -- pretrain_vq has no license ------------------------------------------------


def test_pretrain_vq_refuses_without_the_acknowledgement_flag(tmp_path: Path) -> None:
    source = write_source(tmp_path, name="model.pt")
    argv = [
        argument
        for argument in execute_argv(tmp_path, "minivla_vq", source=source)
        if argument != "--acknowledge-unresolved-license"
    ]
    summary, torch, save_file = invoke(tmp_path, argv)
    report = summary["report"]
    assert report["status"] == "failed"
    assert "Stanford-ILIAD/pretrain_vq" in report["error"]["message"]
    assert "--acknowledge-unresolved-license" in report["error"]["message"]
    assert "will not write a license string it cannot substantiate" in report["error"]["message"]
    assert report["license"]["acknowledgement_required"] is True
    assert report["license"]["acknowledged_by_operator"] is False
    assert converter.LICENSE_ACKNOWLEDGEMENT_BLOCKER in report["verification_blockers"]
    assert torch.load_calls == []
    assert save_file.calls == []


def test_pretrain_vq_records_a_null_license_when_acknowledged(tmp_path: Path) -> None:
    source = write_source(tmp_path, name="model.pt")
    summary, _, save_file = invoke(tmp_path, execute_argv(tmp_path, "minivla_vq", source=source))
    report = summary["report"]
    assert report["status"] == "completed"
    assert report["license"]["license"] is None
    assert report["license"]["license_status"] == MINIVLA_VQ_LICENSE_STATUS
    assert report["license"]["redistribution"] == MINIVLA_VQ_REDISTRIBUTION
    assert report["license"]["acknowledged_by_operator"] is True
    assert converter.UNRESOLVED_LICENSE_BLOCKER in report["verification_blockers"]
    assert any("declares NO license" in notice for notice in report["license"]["notices"])
    metadata = save_file.calls[0]["metadata"]
    assert "license" not in metadata
    assert "PLUMB records none rather than inventing one" in metadata["license_note"]
    assert summary["policy_profile_fields"]["license"] is None


def test_openpizero_carries_the_license_the_plan_actually_records(tmp_path: Path) -> None:
    source = write_source(tmp_path, name=OPEN_PI_ZERO_CHECKPOINT_FILE)
    argv = [
        argument
        for argument in execute_argv(tmp_path, "openpizero", source=source)
        if argument != "--acknowledge-unresolved-license"
    ]
    summary, _, save_file = invoke(tmp_path, argv)
    report = summary["report"]
    assert report["status"] == "completed"
    assert report["license"]["license"] == "MIT"
    assert report["license"]["license_status"] == "advertised_unverified"
    assert report["license"]["acknowledgement_required"] is False
    assert save_file.calls[0]["metadata"]["license"] == "MIT"


# -- the report ----------------------------------------------------------------


REQUIRED_REPORT_FIELDS = (
    ("source", "repo_id"),
    ("source", "revision"),
    ("source", "repo_path"),
    ("source", "sha256"),
    ("deserialization", "approved_globals_used"),
    ("deserialization", "rejected_globals"),
    ("output", "sha256"),
    ("output", "tensor_count"),
    ("output", "total_tensor_bytes"),
    ("output", "dtype_summary"),
    ("converter", "source_revision"),
)


def test_report_carries_every_required_field_and_the_security_note(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path))
    report = summary["report"]
    assert report["status"] == "completed"
    assert report["schema_version"] == converter.SCHEMA_VERSION
    assert report["kind"] == converter.REPORT_KIND
    for section, field in REQUIRED_REPORT_FIELDS:
        assert field in report[section], "%s.%s missing" % (section, field)
        if field != "revision":
            assert report[section][field] is not None, "%s.%s is null" % (section, field)
    assert report["source"]["repo_id"] == "Stanford-ILIAD/minivla-vq-bridge-prismatic"
    assert report["source"]["repo_path"] == MINIVLA_CHECKPOINT_FILE
    assert report["source"]["revision"] == REVISION
    assert report["source"]["revision_origin"] == "operator"
    assert report["source"]["sha256"] == source_digest()
    assert report["converter"]["source_sha256"] == hashlib.sha256(
        Path(converter.__file__).read_bytes()
    ).hexdigest()
    assert report["converter"]["python_version"]
    assert report["converter"]["torch_version"] == "2.5.1"
    assert report["security_note"] == converter.SECURITY_NOTE
    assert "mitigation, not a proof of safety" in report["security_note"]
    assert "does not establish safety" in report["security_note"]


def test_report_records_the_measured_tensors(tmp_path: Path) -> None:
    summary, _, save_file = invoke(tmp_path, execute_argv(tmp_path))
    output = summary["report"]["output"]
    assert output["tensor_count"] == 3
    assert output["total_tensor_bytes"] == 4 * 2 + 2 * 2 + 8 * 4
    assert output["dtype_summary"] == {"torch.bfloat16": 2, "torch.float32": 1}
    assert output["state_dict_key"] == "model"
    assert output["ignored_top_level_keys"] == ["epoch", "run_id"]
    assert output["flatten_separator"] == "."
    assert save_file.calls[0]["keys"] == [
        "projector.fc.bias",
        "projector.fc.weight",
        "vision_backbone.patch.weight",
    ]
    assert output["bytes"] == Path(output["path"]).stat().st_size
    assert output["sha256"] == hashlib.sha256(Path(output["path"]).read_bytes()).hexdigest()


def test_report_hash_is_the_hash_of_the_written_report(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path))
    report_path = Path(summary["report_path"])
    assert report_path.is_file()
    assert summary["conversion_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert json.loads(report_path.read_text())["status"] == "completed"
    assert report_path == (tmp_path / "converted.conversion-report.json")


def test_report_names_what_it_does_not_establish(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path))
    report = summary["report"]
    assert converter.NETWORK_ISOLATION_BLOCKER in report["verification_blockers"]
    assert converter.STATE_DICT_COVERAGE_BLOCKER in report["verification_blockers"]
    assert report["isolation"]["network_isolation_verified"] is False
    assert report["isolation"]["unprivileged"] is True
    assert report["strict_state_dict_verification"]["performed"] is False
    note = report["strict_state_dict_verification"]["note"]
    assert "strict=True checks keys, not provenance or deserialization safety" in note
    assert "Do not set the OpenPiZero profile's state_dict_coverage_verified" in note


# -- what the adapters actually require ----------------------------------------


def test_minivla_fields_satisfy_the_real_converted_artifact_review(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path))
    fields = summary["policy_profile_fields"]
    artifact = ConvertedWeightArtifact(**fields)
    assert artifact.review_error() is None
    assert artifact.sha256 == summary["report"]["output"]["sha256"]
    assert artifact.source_pickle_sha256 == source_digest()
    assert artifact.conversion_report_sha256 == summary["conversion_report_sha256"]
    assert artifact.conversion_report_path == summary["report_path"]
    assert artifact.license is None
    assert artifact.redistribution == MINIVLA_VQ_REDISTRIBUTION


def test_minivla_vq_fields_satisfy_the_real_converted_artifact_review(tmp_path: Path) -> None:
    source = write_source(tmp_path, name="model.pt")
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path, "minivla_vq", source=source))
    artifact = ConvertedWeightArtifact(**summary["policy_profile_fields"])
    assert artifact.review_error() is None
    assert artifact.license is None
    assert artifact.license_status == MINIVLA_VQ_LICENSE_STATUS


def _openpizero_profile(tmp_path: Path, fields: Mapping[str, Any]) -> OpenPiZeroPolicyProfile:
    paligemma_root = tmp_path / "paligemma"
    paligemma_root.mkdir(exist_ok=True)
    for name in PALIGEMMA_REQUIRED_FILES:
        (paligemma_root / name).write_text("{}")
    values: Dict[str, Any] = {
        "profile_id": "open-pi-zero-converted",
        "checkpoint_revision": REVISION,
        "paligemma": PaliGemmaSupportFiles(
            local_path=str(paligemma_root), revision=REVISION, terms_accepted=True
        ),
        "proprio_layout": BRIDGE_8D_PASSTHROUGH_LAYOUT,
        "entry_point": NativeWrapperEntryPoint(
            module="open_pi_zero_fake",
            attribute="build_bridge_wrapper",
            loader_revision=OPEN_PI_ZERO_SOURCE_COMMIT,
        ),
        "application_cache_dir": str(tmp_path / "cache"),
    }
    values.update(fields)
    return OpenPiZeroPolicyProfile(**values)


def _install_fake_loader(
    monkeypatch: pytest.MonkeyPatch, sink: Dict[str, Any], *, missing: Sequence[str] = ()
) -> str:
    """Put a reviewed-loader stand-in on ``sys.modules`` for --strict-verify."""

    class FakeIncompatibleKeys:
        def __init__(self) -> None:
            self.missing_keys = list(missing)
            self.unexpected_keys: List[str] = []

    class FakeModel:
        def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False) -> Any:
            sink.update({"keys": sorted(state_dict), "strict": strict})
            return FakeIncompatibleKeys()

    module = types.ModuleType("plumb_fake_open_pi_zero")
    module.build_bridge_wrapper = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "plumb_fake_open_pi_zero", module)
    return "plumb_fake_open_pi_zero:build_bridge_wrapper"


def test_openpizero_fields_satisfy_the_real_profile_review_after_strict_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path, name=OPEN_PI_ZERO_CHECKPOINT_FILE)
    loaded: Dict[str, Any] = {}
    entry_point = _install_fake_loader(monkeypatch, loaded)
    summary, _, _ = invoke(
        tmp_path,
        execute_argv(
            tmp_path,
            "openpizero",
            source=source,
            extra=["--strict-verify", entry_point, "--strict-verify-loader-revision", LOADER_REVISION],
        ),
    )
    report = summary["report"]
    assert report["status"] == "completed"
    assert loaded["strict"] is True
    assert loaded["keys"] == [
        "projector.fc.bias",
        "projector.fc.weight",
        "vision_backbone.patch.weight",
    ]
    assert report["strict_state_dict_verification"]["performed"] is True
    assert report["strict_state_dict_verification"]["entry_point"] == entry_point
    assert report["strict_state_dict_verification"]["loader_revision"] == LOADER_REVISION
    assert converter.STATE_DICT_COVERAGE_BLOCKER not in report["verification_blockers"]
    fields = summary["policy_profile_fields"]
    assert fields["state_dict_coverage_verified"] is True
    profile = _openpizero_profile(tmp_path, fields)
    assert profile.review_error() is None
    assert profile.converted_checkpoint_sha256 == report["output"]["sha256"]
    assert profile.conversion_report_sha256 == summary["conversion_report_sha256"]


def test_incomplete_strict_coverage_refuses_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path, name=OPEN_PI_ZERO_CHECKPOINT_FILE)
    entry_point = _install_fake_loader(monkeypatch, {}, missing=["decoder.weight"])
    summary, _, save_file = invoke(
        tmp_path,
        execute_argv(
            tmp_path,
            "openpizero",
            source=source,
            extra=["--strict-verify", entry_point, "--strict-verify-loader-revision", LOADER_REVISION],
        ),
    )
    assert summary["report"]["status"] == "failed"
    assert "decoder.weight" in summary["report"]["error"]["message"]
    assert save_file.calls == []
    assert not (tmp_path / "converted.safetensors").exists()


def test_openpizero_profile_refuses_when_coverage_was_not_verified(tmp_path: Path) -> None:
    source = write_source(tmp_path, name=OPEN_PI_ZERO_CHECKPOINT_FILE)
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path, "openpizero", source=source))
    fields = summary["policy_profile_fields"]
    assert fields["state_dict_coverage_verified"] is False
    error = _openpizero_profile(tmp_path, fields).review_error()
    assert error is not None
    assert "state-dict coverage" in error


def test_strict_verify_requires_a_pinned_loader_revision(tmp_path: Path) -> None:
    source = write_source(tmp_path, name=OPEN_PI_ZERO_CHECKPOINT_FILE)
    summary, _, save_file = invoke(
        tmp_path,
        execute_argv(tmp_path, "openpizero", source=source, extra=["--strict-verify", "fake:factory"]),
    )
    assert summary["report"]["status"] == "failed"
    assert "--strict-verify-loader-revision" in summary["report"]["error"]["message"]
    assert save_file.calls == []


# -- environment isolation -----------------------------------------------------


@pytest.mark.parametrize(
    "variable",
    ["BASETEN_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "SOME_VENDOR_API_KEY", "MY_SERVICE_PASSWORD"],
)
def test_credentials_in_the_environment_refuse_the_run(tmp_path: Path, variable: str) -> None:
    environment = converter.Environment(
        variables={"PATH": "/usr/bin", variable: "secret-value"},
        euid=1000,
        home=str(tmp_path / "isolated-home"),
    )
    summary, torch, save_file = invoke(tmp_path, execute_argv(tmp_path), environment=environment)
    report = summary["report"]
    assert report["status"] == "failed"
    assert "credentials are present in the environment" in report["error"]["message"]
    assert variable in report["error"]["message"]
    assert report["isolation"]["credential_environment_variables_present"] == [variable]
    assert torch.serialization.enumerated == []
    assert torch.load_calls == []
    assert save_file.calls == []
    assert not (tmp_path / "converted.safetensors").exists()


def test_an_empty_credential_variable_is_not_a_refusal(tmp_path: Path) -> None:
    environment = converter.Environment(
        variables={"PATH": "/usr/bin", "HF_TOKEN": ""}, euid=1000, home=str(tmp_path / "isolated-home")
    )
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path), environment=environment)
    assert summary["report"]["status"] == "completed"


def test_credential_files_in_the_home_directory_refuse_the_run(tmp_path: Path) -> None:
    home = tmp_path / "isolated-home"
    (home / ".aws").mkdir(parents=True)
    (home / ".aws" / "credentials").write_text("[default]\n")
    environment = converter.Environment(variables={"PATH": "/usr/bin"}, euid=1000, home=str(home))
    summary, _, save_file = invoke(tmp_path, execute_argv(tmp_path), environment=environment)
    assert summary["report"]["status"] == "failed"
    assert ".aws/credentials" in summary["report"]["error"]["message"]
    assert save_file.calls == []


def test_running_as_root_refuses(tmp_path: Path) -> None:
    environment = converter.Environment(
        variables={"PATH": "/usr/bin"}, euid=0, home=str(tmp_path / "isolated-home")
    )
    summary, _, save_file = invoke(tmp_path, execute_argv(tmp_path), environment=environment)
    assert summary["report"]["status"] == "failed"
    assert "unprivileged" in summary["report"]["error"]["message"]
    assert summary["report"]["isolation"]["unprivileged"] is False
    assert save_file.calls == []


def test_isolation_report_names_what_it_checked(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path))
    isolation = summary["report"]["isolation"]
    assert "HF_TOKEN" in isolation["credential_variable_names_checked"]
    assert "BASETEN_API_KEY" in isolation["credential_variable_names_checked"]
    assert ".netrc" in isolation["credential_files_checked"]
    assert isolation["network_isolation_verified"] is False
    assert "cannot verify" in isolation["network_isolation_note"]


# -- state dict shape ----------------------------------------------------------


def test_a_non_tensor_leaf_refuses_rather_than_inventing_one(tmp_path: Path) -> None:
    checkpoint = {"model": {"projector": {"fc.weight": FakeTensor(4), "notes": "a string"}}}
    summary, _, save_file = invoke(tmp_path, execute_argv(tmp_path), checkpoint=checkpoint)
    assert summary["report"]["status"] == "failed"
    assert "projector.notes is a str, not a tensor" in summary["report"]["error"]["message"]
    assert save_file.calls == []
    assert not (tmp_path / "converted.safetensors").exists()


def test_an_unrecognisable_top_level_refuses_and_names_the_keys(tmp_path: Path) -> None:
    checkpoint = {"ema": {"fc.weight": FakeTensor(2)}, "step": 7}
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path), checkpoint=checkpoint)
    message = summary["report"]["error"]["message"]
    assert summary["report"]["status"] == "failed"
    assert "ema (dict)" in message
    assert "step (int)" in message
    assert "--state-dict-key" in message


def test_an_explicit_state_dict_key_is_honoured(tmp_path: Path) -> None:
    checkpoint = {"ema": {"fc.weight": FakeTensor(2)}, "step": 7}
    summary, _, save_file = invoke(
        tmp_path, execute_argv(tmp_path, extra=["--state-dict-key", "ema"]), checkpoint=checkpoint
    )
    assert summary["report"]["status"] == "completed"
    assert summary["report"]["output"]["state_dict_key"] == "ema"
    assert save_file.calls[0]["keys"] == ["fc.weight"]


def test_a_flat_tensor_mapping_is_converted_at_the_top_level(tmp_path: Path) -> None:
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path), checkpoint={"fc.weight": FakeTensor(2)})
    assert summary["report"]["status"] == "completed"
    assert summary["report"]["output"]["state_dict_key"] is None


def test_an_empty_state_dict_refuses(tmp_path: Path) -> None:
    summary, _, save_file = invoke(tmp_path, execute_argv(tmp_path), checkpoint={"model": {}})
    assert summary["report"]["status"] == "failed"
    assert "empty" in summary["report"]["error"]["message"]
    assert save_file.calls == []


def test_tensors_are_detached_before_saving(tmp_path: Path) -> None:
    checkpoint = minivla_checkpoint()
    summary, _, _ = invoke(tmp_path, execute_argv(tmp_path), checkpoint=checkpoint)
    assert summary["report"]["status"] == "completed"
    assert checkpoint["model"]["projector"]["fc.weight"].detached is True


# -- command line --------------------------------------------------------------


def test_list_policies_needs_no_torch_and_no_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert converter.main(["--list-policies"], runtime_factory=_no_runtime) == 0
    catalogue = json.loads(capsys.readouterr().out)
    assert [entry["policy"] for entry in catalogue["policies"]] == ["minivla", "minivla_vq", "openpizero"]
    assert catalogue["security_note"] == converter.SECURITY_NOTE


def _no_runtime() -> converter.ConverterRuntime:
    raise AssertionError("--list-policies must not touch torch")


def test_inspect_and_execute_cannot_be_combined() -> None:
    with pytest.raises(SystemExit):
        request_from(["--policy", "minivla", "--source", "/dev/null", "--inspect", "--execute"])


def test_main_returns_two_on_a_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    torch = FakeTorch(minivla_checkpoint(), ["wandb.Run"])
    runtime = converter.ConverterRuntime(
        torch=torch, save_file=RecordingSaveFile(), torch_version="2.5.1", safetensors_version="0.4.5"
    )
    code = converter.main(
        execute_argv(tmp_path),
        runtime_factory=lambda: runtime,
        environment=clean_environment(tmp_path),
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["report"]["status"] == "failed"


def test_main_returns_zero_on_a_completed_conversion(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runtime = converter.ConverterRuntime(
        torch=FakeTorch(minivla_checkpoint()),
        save_file=RecordingSaveFile(),
        torch_version="2.5.1",
        safetensors_version="0.4.5",
    )
    code = converter.main(
        execute_argv(tmp_path),
        runtime_factory=lambda: runtime,
        environment=clean_environment(tmp_path),
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["status"] == "completed"
    assert ConvertedWeightArtifact(**summary["policy_profile_fields"]).review_error() is None


def test_unknown_policy_is_refused_by_name() -> None:
    with pytest.raises(converter.ConversionRefusal) as caught:
        converter.get_target("openvla")
    assert "minivla" in str(caught.value)
