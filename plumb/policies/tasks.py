"""Frozen primary-scoring task registry for the five PLUMB benchmark tasks.

Primary scoring deliberately resolves instructions and rubrics from this module
by ``task_id``.  Callers cannot insert a policy name, action transcript,
published rate, or a newly worded rubric into a primary judge prompt.  Any
free-text experiment belongs to explicit diagnostic mode and is never
Gate-D-eligible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


TASK_REGISTRY_ID = "plumb-benchmark-tasks-v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    instruction: str
    rubric: str
    max_steps: int

    @property
    def rubric_hash(self) -> str:
        return _canonical_hash({"task_id": self.task_id, "rubric": self.rubric})


@dataclass(frozen=True)
class TaskRegistry:
    registry_id: str
    tasks: Tuple[BenchmarkTask, ...]

    def __post_init__(self) -> None:
        ids = tuple(task.task_id for task in self.tasks)
        if not self.registry_id or len(ids) != len(set(ids)):
            raise ValueError("TaskRegistry needs a nonempty ID and unique task IDs.")

    @property
    def registry_hash(self) -> str:
        return _canonical_hash(
            {
                "registry_id": self.registry_id,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "instruction": task.instruction,
                        "rubric": task.rubric,
                        "max_steps": task.max_steps,
                    }
                    for task in self.tasks
                ],
            }
        )

    def get(self, task_id: str) -> BenchmarkTask:
        if not isinstance(task_id, str):
            raise KeyError(task_id)
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "registry_id": self.registry_id,
            "registry_hash": self.registry_hash,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "instruction": task.instruction,
                    "rubric": task.rubric,
                    "rubric_hash": task.rubric_hash,
                    "max_steps": task.max_steps,
                }
                for task in self.tasks
            ],
        }


# Rubrics are frozen English encodings of the §5 milestone table, with the
# §3 final-state criterion substituted per exact benchmark task.  Do not edit
# an entry in place after a protocol is frozen: introduce a new registry ID.
BENCHMARK_TASK_REGISTRY = TaskRegistry(
    registry_id=TASK_REGISTRY_ID,
    tasks=(
        BenchmarkTask(
            task_id="close_drawer",
            instruction="Close the drawer",
            rubric=(
                "Drawer milestone rubric. 0: no directed approach. 1: approach handle. "
                "2: handle/contact established. 3: drawer moves toward target. "
                "4: partial target closure. 5: the drawer is completely closed in the final visible state. "
                "Score final completion from visible end state; invisible events must not be inferred."
            ),
            max_steps=70,
        ),
        BenchmarkTask(
            task_id="open_drawer",
            instruction="Open the drawer",
            rubric=(
                "Drawer milestone rubric. 0: no directed approach. 1: approach handle. "
                "2: handle/contact established. 3: drawer moves toward target. "
                "4: partial target opening. 5: the drawer is visibly opened by at least 1.5 cm in the final state. "
                "Score final completion from visible end state; invisible events must not be inferred."
            ),
            max_steps=70,
        ),
        BenchmarkTask(
            task_id="to_basket",
            instruction="Put the eggplant in the yellow basket",
            rubric=(
                "Pick/place milestone rubric. 0: no directed approach. 1: approach eggplant. "
                "2: grasp/contact established. 3: eggplant lifted. 4: transported over destination. "
                "5: eggplant is fully inside the yellow basket at episode end. "
                "Score final completion from visible end state; invisible events must not be inferred."
            ),
            max_steps=100,
        ),
        BenchmarkTask(
            task_id="to_sink",
            instruction="Put the eggplant in the blue sink",
            rubric=(
                "Pick/place milestone rubric. 0: no directed approach. 1: approach eggplant. "
                "2: grasp/contact established. 3: eggplant lifted. 4: transported over destination. "
                "5: eggplant is fully inside the blue sink at episode end. "
                "Score final completion from visible end state; invisible events must not be inferred."
            ),
            max_steps=100,
        ),
        BenchmarkTask(
            task_id="fold_cloth",
            instruction="fold the cloth from top right to bottom left",
            rubric=(
                "Cloth milestone rubric. 0: no directed approach. 1: approach designated corner. "
                "2: corner grasp/contact. 3: corner lifted. 4: moved diagonally toward target. "
                "5: cloth is folded at least one quarter diagonally from top right to bottom left in the final state. "
                "Score final completion from visible end state; invisible events must not be inferred."
            ),
            max_steps=80,
        ),
    ),
)

TASK_REGISTRY_HASH = BENCHMARK_TASK_REGISTRY.registry_hash
