"""Experimental demo task additions; the frozen benchmark registry is unchanged."""
from .provenance import canonical_json_sha256

POT_TASK_ID = "demo_pot_left_v1"
POT_INSTRUCTION = "Put the pot to the left of the purple item."
POT_RUBRIC = (
    "Assess the metal bowl/pot relative to the purple object using visible evidence. "
    "0: no directed approach. 1: approach the bowl/pot. 2: contact or grasp. "
    "3: bowl/pot visibly moves. 4: transported toward the specified position. "
    "5: bowl/pot visibly rests to the left of the purple item at the end. "
    "Distinguish its initial position from an observed completed manipulation; do not invent unseen motion."
)
POT_RUBRIC_HASH = canonical_json_sha256({"diagnostic_rubric": POT_RUBRIC})
