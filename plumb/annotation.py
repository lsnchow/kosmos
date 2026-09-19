"""Public entrypoint for the blinded human-annotation workflow.

The implementation lives in :mod:`plumb.calibration` so packet parsing,
assignment validation, selection freezing, and Gate-D reporting use one strict
schema.  Keeping this thin module gives the top-level CLI an intentional
``plumb annotation`` namespace without duplicating labels or policy logic.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .calibration import (
    Annotation,
    AnnotatorOwnership,
    CalibrationError,
    ClipManifestRow,
    blinded_annotation_rows,
    blinded_clip_id,
    blinded_media_ref,
    blinded_media_resolver,
    build_calibration_report,
    deterministic_annotation_assignments,
    export_annotation_packets,
    freeze_calibration_selection,
    import_annotations,
    load_frozen_calibration_selection,
    main as _calibration_main,
    validate_calibration_lineage_partition,
    validate_annotator_ownership,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``export``, ``freeze``, or ``report`` for blinded annotations."""

    return _calibration_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Annotation",
    "AnnotatorOwnership",
    "CalibrationError",
    "ClipManifestRow",
    "blinded_annotation_rows",
    "blinded_clip_id",
    "blinded_media_ref",
    "blinded_media_resolver",
    "build_calibration_report",
    "deterministic_annotation_assignments",
    "export_annotation_packets",
    "freeze_calibration_selection",
    "import_annotations",
    "load_frozen_calibration_selection",
    "main",
    "validate_calibration_lineage_partition",
    "validate_annotator_ownership",
]
