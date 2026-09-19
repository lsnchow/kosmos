"""Pure tokenizer-boundary tests for untrusted JSON-value masking."""

from __future__ import annotations

import json

import pytest

from cluster import judge_format_training as fmt


def raw_label() -> str:
    return json.dumps(
        {
            "integrity": "artifact",
            "collision": "visible",
            "progress": 5,
            "completion_evidence": "met",
            "evidence_frame_indices": [0, 15],
            "observable_reasons": "Untrusted actual teacher wording.",
        }
    )


class CharacterFastTokenizer:
    is_fast = True
    eos_token_id = 999

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False and return_offsets_mapping is True
        return {"input_ids": [ord(character) for character in text], "offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def decode(self, ids, **_kwargs):
        return "".join("\n" if token_id == 998 else "<eos>" if token_id == 999 else chr(token_id) for token_id in ids)


class NonFastTokenizer:
    is_fast = False


class FakeScalar(int):
    def item(self):
        return int(self)


class FakeMatrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def clone(self):
        return FakeMatrix(self.rows)

    def tolist(self):
        return [list(row) for row in self.rows]

    def __getitem__(self, key):
        row, column = key
        return FakeScalar(self.rows[row][column])

    def __setitem__(self, key, value):
        row, column = key
        self.rows[row][column] = int(value)


def test_values_are_all_masked_but_keys_and_object_punctuation_are_structural():
    target = fmt.target_from_actual_raw(raw_label())
    ids, offsets = fmt.tokenize_target_fast(CharacterFastTokenizer(), target)
    audit = fmt.alignment_audit(target, ids, offsets, ids + [999], [999])
    structural = set(audit["structural_target_token_indices"])
    masked = set(audit["masked_semantic_target_token_indices"])
    assert structural
    assert not structural & masked
    for span in target.value_spans:
        assert all(index in masked for index, (start, end) in enumerate(offsets) if start < span["end"] and end > span["start"])
    assert target.canonical_target.index("{") in structural
    assert audit["assistant_eos_token_count"] == 1
    assert audit["semantic_labels_supervised"] is False


def test_token_crossing_semantic_value_and_delimiter_is_masked_conservatively():
    target = fmt.target_from_actual_raw(raw_label())
    # One token spans the colon plus the opening quote/value of integrity.
    value_start = next(span["start"] for span in target.value_spans if span["field"] == "integrity")
    offsets = [(index, index + 1) for index in range(len(target.canonical_target))]
    offsets[value_start - 1:value_start + 1] = [(value_start - 1, value_start + 1)]
    ids = list(range(len(offsets)))
    audit = fmt.alignment_audit(target, ids, offsets, ids, [])
    crossing = value_start - 1
    assert crossing in audit["masked_semantic_target_token_indices"]


def test_contextual_bpe_mismatch_or_nonfast_offsets_fail_closed():
    target = fmt.target_from_actual_raw(raw_label())
    ids, offsets = fmt.tokenize_target_fast(CharacterFastTokenizer(), target)
    with pytest.raises(fmt.FormatMaskError, match="contextual BPE"):
        fmt.alignment_audit(target, ids, offsets, [12345] + ids[1:], [999])
    with pytest.raises(fmt.FormatMaskError, match="fast tokenizer"):
        fmt.tokenize_target_fast(NonFastTokenizer(), target)


def test_only_eos_is_supervised_when_chat_template_leaves_a_whitespace_tail():
    target = fmt.target_from_actual_raw(raw_label())
    tokenizer = CharacterFastTokenizer()
    ids, offsets = fmt.tokenize_target_fast(tokenizer, target)
    audit = fmt.alignment_audit(target, ids, offsets, ids + [999, 998], [999], tokenizer)
    assert audit["assistant_eos_relative_index"] == 0
    assert audit["assistant_masked_tail_relative_indices"] == [1]
    assert audit["semantic_supervised_tokens"] == 0
    assert audit["supervised_syntax_tokens"] >= 1


def test_row_must_preserve_actual_raw_teacher_json_not_an_independently_changed_label():
    target = fmt.target_from_actual_raw(raw_label())
    row = {
        "purpose": fmt.PURPOSE,
        "qualified": False,
        "label_source": fmt.LABEL_SOURCE,
        "raw_teacher_output": raw_label(),
        "label": json.loads(target.canonical_target),
    }
    assert fmt.validate_format_row(row).target_sha256 == target.target_sha256
    row["label"]["progress"] = 4
    with pytest.raises(fmt.FormatMaskError, match="fabricated"):
        fmt.validate_format_row(row)


def test_full_fake_serving_collator_replaces_spaced_target_and_masks_values_and_post_eos_newline(monkeypatch):
    from deploy.baseten.training import train_judge_lora as trainer

    target = fmt.target_from_actual_raw(raw_label())
    row = {
        "purpose": fmt.PURPOSE,
        "qualified": False,
        "label_source": fmt.LABEL_SOURCE,
        "raw_teacher_output": raw_label(),
        "label": json.loads(target.canonical_target),
    }
    tokenizer = CharacterFastTokenizer()

    def fake_rows(rows):
        # Deliberately noncanonical initial string: format collator must replace it.
        return [{"target": json.dumps(rows[0]["label"], sort_keys=True)}]

    def fake_base(_processor):
        def collate(prepared):
            assert prepared[0]["target"] == target.canonical_target
            assistant = [ord(character) for character in target.canonical_target] + [999, 998]
            ids = [11, 12] + assistant
            labels = [-100, -100] + assistant
            return {"input_ids": FakeMatrix([ids]), "labels": FakeMatrix([labels])}
        return collate

    monkeypatch.setattr(trainer, "_as_training_rows", fake_rows)
    monkeypatch.setattr(trainer, "_build_collator", fake_base)
    inputs, audit = fmt.collate_with_structure_audit(type("Processor", (), {"tokenizer": tokenizer})(), row)
    labels = inputs["labels"].tolist()[0]
    value_start = next(span["start"] for span in target.value_spans if span["field"] == "integrity")
    assert labels[2 + value_start] == -100
    assert labels[2] != -100  # opening JSON brace
    assert labels[-2] == 999  # EOS remains supervised
    assert labels[-1] == -100  # post-EOS newline is masked
    assert audit["semantic_supervised_tokens"] == 0
    assert audit["supervised_syntax_tokens"] >= 1
