"""Fail-closed structure-only assistant loss masking for an unqualified pilot.

Teacher semantic values are deliberately untrusted here.  Given an actual
schema-valid raw teacher JSON object, this module supervises only the canonical
JSON object's keys and punctuation.  Every value span—including integrity,
collision, progress, completion, evidence indices, and free-text reasons—is
masked.  A token that overlaps both a delimiter and a value is masked too.

This is not a semantic distillation, calibration, or scoring path.  It exists
only to test whether a LoRA adapter can learn the serving judge's output shape
without receiving a teacher's outcome values as targets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PURPOSE = "judge_json_structure_only_pilot"
LABEL_SOURCE = "teacher_values_untrusted_masked"
EXPECTED_LABEL_KEYS = (
    "integrity",
    "collision",
    "progress",
    "completion_evidence",
    "evidence_frame_indices",
    "observable_reasons",
)


class FormatMaskError(ValueError):
    """The tokenizer/assistant boundary cannot safely support structure-only loss."""


@dataclass(frozen=True)
class TargetStructure:
    canonical_target: str
    target_sha256: str
    structural_characters: Tuple[bool, ...]
    value_spans: Tuple[Dict[str, Any], ...]


def _canonical_target(label: Mapping[str, Any]) -> str:
    return json.dumps(dict(label), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def target_from_actual_raw(raw_teacher_output: Any) -> TargetStructure:
    """Canonicalize only an actual schema-valid raw teacher object.

    ``parse_rubric_json`` validates the raw string itself.  A caller cannot
    provide an independently manufactured label mapping as a shortcut.
    """

    if not isinstance(raw_teacher_output, str) or not raw_teacher_output.strip():
        raise FormatMaskError("raw_teacher_output must be a nonempty actual JSON response string")
    from plumb.policies.judge import parse_rubric_json

    try:
        label = dict(parse_rubric_json(raw_teacher_output).as_dict())
    except Exception as error:
        raise FormatMaskError("raw_teacher_output is not schema-valid judge JSON") from error
    if tuple(sorted(label)) != tuple(sorted(EXPECTED_LABEL_KEYS)):
        raise FormatMaskError("parsed teacher output does not contain the frozen rubric fields")
    return structure_for_label(label)


def structure_for_label(label: Mapping[str, Any]) -> TargetStructure:
    """Return canonical bare JSON plus exact top-level semantic value spans."""

    if tuple(sorted(label)) != tuple(sorted(EXPECTED_LABEL_KEYS)):
        raise FormatMaskError("label keys do not match the frozen rubric schema")
    target = _canonical_target(label)
    decoder = json.JSONDecoder()
    structural = [False] * len(target)
    spans: List[Dict[str, Any]] = []
    position = 0
    if not target.startswith("{") or not target.endswith("}"):
        raise FormatMaskError("canonical rubric target is not a JSON object")
    structural[position] = True
    position += 1
    while position < len(target) - 1:
        try:
            key, key_end = decoder.raw_decode(target, position)
        except json.JSONDecodeError as error:
            raise FormatMaskError("cannot parse a canonical JSON key boundary") from error
        if not isinstance(key, str):
            raise FormatMaskError("canonical rubric key is not text")
        for index in range(position, key_end):
            structural[index] = True
        position = key_end
        if position >= len(target) or target[position] != ":":
            raise FormatMaskError("canonical rubric key has no colon")
        structural[position] = True
        position += 1
        value_start = position
        try:
            _, value_end = decoder.raw_decode(target, position)
        except json.JSONDecodeError as error:
            raise FormatMaskError("cannot parse a canonical JSON value boundary") from error
        spans.append({"field": key, "start": value_start, "end": value_end, "text": target[value_start:value_end]})
        position = value_end
        if position == len(target) - 1:
            break
        if target[position] != ",":
            raise FormatMaskError("canonical rubric values must be comma-separated")
        structural[position] = True
        position += 1
    structural[-1] = True
    if {span["field"] for span in spans} != set(EXPECTED_LABEL_KEYS):
        raise FormatMaskError("canonical target does not expose every expected semantic value span")
    return TargetStructure(
        canonical_target=target,
        target_sha256="sha256:" + hashlib.sha256(target.encode("utf-8")).hexdigest(),
        structural_characters=tuple(structural),
        value_spans=tuple(spans),
    )


def _unwrap_single(value: Any, label: str) -> List[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise FormatMaskError("%s must be a one-row token sequence" % label)
    return list(value)


def tokenize_target_fast(tokenizer: Any, target: TargetStructure) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Tokenize bare canonical JSON with offset mappings, or fail closed."""

    if getattr(tokenizer, "is_fast", None) is not True:
        raise FormatMaskError("structure-only masking requires a fast tokenizer with offset mappings")
    try:
        encoded = tokenizer(target.canonical_target, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError) as error:
        raise FormatMaskError("fast tokenizer did not provide offset mappings") from error
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise FormatMaskError("fast tokenizer response lacks input_ids/offset_mapping")
    ids_raw = _unwrap_single(encoded["input_ids"], "target input_ids")
    offsets_raw = _unwrap_single(encoded["offset_mapping"], "target offset_mapping")
    if not ids_raw or len(ids_raw) != len(offsets_raw):
        raise FormatMaskError("target token IDs and offsets are empty or unequal")
    ids: List[int] = []
    offsets: List[Tuple[int, int]] = []
    for index, (token_id, raw_offset) in enumerate(zip(ids_raw, offsets_raw)):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise FormatMaskError("target token %d has a non-integer ID" % index)
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise FormatMaskError("target token %d has no two-element offset" % index)
        start, end = raw_offset
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise FormatMaskError("target token %d offset is not integer" % index)
        if not 0 <= start < end <= len(target.canonical_target):
            raise FormatMaskError("target token %d has an invalid/ambiguous offset" % index)
        ids.append(token_id)
        offsets.append((start, end))
    return ids, offsets


def _find_exact_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    matches = []
    if not needle:
        raise FormatMaskError("canonical target token sequence is empty")
    for index in range(0, len(haystack) - len(needle) + 1):
        if list(haystack[index:index + len(needle)]) == list(needle):
            matches.append(index)
    if len(matches) != 1:
        raise FormatMaskError("canonical target IDs do not form one exact assistant-token subsequence; contextual BPE boundary is unsafe")
    return matches[0]


def _tail_token_text(tokenizer: Any, token_id: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise FormatMaskError("tokenizer must decode non-EOS assistant tail tokens for whitespace verification")
    try:
        return str(decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    except TypeError:
        return str(decode([token_id]))


def alignment_audit(
    target: TargetStructure,
    target_ids: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    assistant_ids: Sequence[int],
    eos_token_ids: Sequence[int],
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Map bare-JSON offsets to the actual assistant sequence conservatively."""

    start = _find_exact_subsequence(assistant_ids, target_ids)
    if start != 0:
        raise FormatMaskError("assistant labels contain unexpected tokens before canonical target")
    tail = list(assistant_ids[len(target_ids):])
    eos_positions = [index for index, token_id in enumerate(tail) if token_id in set(eos_token_ids)]
    if len(eos_positions) > 1:
        raise FormatMaskError("assistant tail contains more than one EOS token")
    if tail and not eos_positions:
        raise FormatMaskError("assistant labels contain no EOS after canonical target")
    eos_relative_index = eos_positions[0] if eos_positions else None
    whitespace_tail_indices = []
    masked_tail_indices = []
    for index, token_id in enumerate(tail):
        if index == eos_relative_index:
            continue
        if tokenizer is None:
            raise FormatMaskError("assistant whitespace tail requires tokenizer decoding for safe verification")
        if not _tail_token_text(tokenizer, token_id).isspace():
            raise FormatMaskError("assistant labels contain unexpected non-whitespace tokens around EOS")
        whitespace_tail_indices.append(index)
        masked_tail_indices.append(index)
    semantic_tokens = []
    structural_tokens = []
    for index, (offset_start, offset_end) in enumerate(offsets):
        # Crossing a value delimiter is unsafe: mask the complete token.
        overlaps_value = any(
            offset_start < int(span["end"]) and offset_end > int(span["start"])
            for span in target.value_spans
        )
        fully_structural = all(target.structural_characters[position] for position in range(offset_start, offset_end))
        if overlaps_value or not fully_structural:
            semantic_tokens.append(index)
        else:
            structural_tokens.append(index)
    if not structural_tokens:
        raise FormatMaskError("no JSON structure token can be safely supervised")
    if set(structural_tokens) & set(semantic_tokens):
        raise FormatMaskError("semantic assistant labels would be exposed")
    return {
        "assistant_target_start": start,
        "target_token_count": len(target_ids),
        "assistant_tail_token_count": len(tail),
        "assistant_eos_token_count": len(eos_positions),
        "assistant_eos_token_ids": [tail[index] for index in eos_positions],
        "assistant_eos_relative_index": eos_relative_index,
        "assistant_whitespace_tail_relative_indices": whitespace_tail_indices,
        "assistant_masked_tail_relative_indices": masked_tail_indices,
        "structural_target_token_indices": structural_tokens,
        "masked_semantic_target_token_indices": semantic_tokens,
        "target_offsets": [{"index": index, "start": start, "end": end} for index, (start, end) in enumerate(offsets)],
        "semantic_value_spans": [dict(span) for span in target.value_spans],
        "canonical_target_sha256": target.target_sha256,
        "canonical_target": target.canonical_target,
        "semantic_labels_supervised": False,
        "semantic_supervised_tokens": 0,
        "supervised_syntax_tokens": len(structural_tokens),
    }


def _eos_ids(tokenizer: Any) -> Tuple[int, ...]:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return tuple(value)
    raise FormatMaskError("tokenizer must declare an EOS token ID for structure-only assistant completion")


def validate_format_row(row: Mapping[str, Any]) -> TargetStructure:
    """Require a raw teacher string and reject an independently supplied value label."""

    if not isinstance(row, Mapping):
        raise FormatMaskError("format-only row must be an object")
    if row.get("purpose") != PURPOSE or row.get("qualified") is not False:
        raise FormatMaskError("row is not explicitly an unqualified JSON-structure-only pilot")
    if row.get("label_source") != LABEL_SOURCE:
        raise FormatMaskError("row does not declare teacher values as untrusted/masked")
    target = target_from_actual_raw(row.get("raw_teacher_output"))
    if row.get("label") != json.loads(target.canonical_target):
        raise FormatMaskError("row label differs from the actual raw teacher JSON and may be fabricated")
    return target


def collate_with_structure_audit(processor: Any, row: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Reuse the hardened serving collator then replace semantic labels by -100.

    Exactly one row is accepted.  The existing collator supplies the actual
    multimodal full-sequence prompt boundary; this function only narrows its
    assistant labels after verifying fast-tokenizer context alignment.
    """

    from deploy.baseten.training import train_judge_lora as trainer

    target = validate_format_row(row)
    prepared = trainer._as_training_rows([row])
    prepared[0]["target"] = target.canonical_target
    inputs = trainer._build_collator(processor)(prepared)
    input_ids = _unwrap_single(inputs["input_ids"], "processor input_ids")
    original_labels = _unwrap_single(inputs["labels"], "serving collator labels")
    if len(input_ids) != len(original_labels):
        raise FormatMaskError("serving collator input IDs/labels differ in length")
    assistant_positions = [index for index, value in enumerate(original_labels) if value != -100]
    if not assistant_positions:
        raise FormatMaskError("serving collator exposed no assistant completion tokens")
    if assistant_positions != list(range(assistant_positions[0], assistant_positions[-1] + 1)):
        raise FormatMaskError("serving collator assistant labels are non-contiguous")
    assistant_ids = [input_ids[index] for index in assistant_positions]
    target_ids, offsets = tokenize_target_fast(processor.tokenizer, target)
    audit = alignment_audit(target, target_ids, offsets, assistant_ids, _eos_ids(processor.tokenizer), processor.tokenizer)
    labels = inputs["labels"].clone()
    # All original assistant tokens are first masked.  We then selectively
    # restore only whole structural tokens plus an optional final EOS tail.
    for position in assistant_positions:
        labels[0, position] = -100
    for relative_index in audit["structural_target_token_indices"]:
        absolute = assistant_positions[audit["assistant_target_start"] + relative_index]
        labels[0, absolute] = input_ids[absolute]
    eos_relative_index = audit["assistant_eos_relative_index"]
    if eos_relative_index is not None:
        eos_position = assistant_positions[len(target_ids) + eos_relative_index]
        labels[0, eos_position] = input_ids[eos_position]
    semantic_positions = [assistant_positions[index] for index in audit["masked_semantic_target_token_indices"]]
    if any(int(labels[0, position].item()) != -100 for position in semantic_positions):
        raise FormatMaskError("semantic target values survived structure-only masking")
    if not any(int(labels[0, position].item()) != -100 for position in assistant_positions):
        raise FormatMaskError("structure-only mask leaves no supervised token")
    inputs["labels"] = labels
    audit.update(
        {
            "assistant_supervised_token_count_before_mask": len(assistant_positions),
            "assistant_supervised_token_count_after_mask": sum(1 for position in assistant_positions if int(labels[0, position].item()) != -100),
            "assistant_input_positions": assistant_positions,
            "masked_semantic_input_positions": semantic_positions,
            "structure_only": True,
            "semantic_supervised_tokens": 0,
            "supervised_syntax_tokens": len(audit["structural_target_token_indices"]),
        }
    )
    return inputs, audit


class StructureOnlyCollator:
    """One-row model-ready collator whose latest immutable audit is inspectable."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.last_audit: Optional[Dict[str, Any]] = None

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Any:
        if not isinstance(batch, Sequence) or len(batch) != 1:
            raise FormatMaskError("structure-only format pilot requires one-row batches")
        inputs, audit = collate_with_structure_audit(self.processor, batch[0])
        self.last_audit = audit
        return inputs


def build_structure_only_collator(processor: Any) -> StructureOnlyCollator:
    return StructureOnlyCollator(processor)


__all__ = [
    "EXPECTED_LABEL_KEYS",
    "FormatMaskError",
    "LABEL_SOURCE",
    "PURPOSE",
    "StructureOnlyCollator",
    "TargetStructure",
    "alignment_audit",
    "build_structure_only_collator",
    "collate_with_structure_audit",
    "structure_for_label",
    "target_from_actual_raw",
    "tokenize_target_fast",
    "validate_format_row",
]
