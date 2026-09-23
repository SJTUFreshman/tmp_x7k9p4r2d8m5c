"""Task registry: dataset name -> adapter class."""
from __future__ import annotations

from .base import EvalResult, Problem, Split, TaskAdapter, build_task
from .conifer import ConiferTask
from .gsm_hard import GSMHardTask
from .math_comp import MathTask
from .multipl_e import MultiplETask
from .musique import MuSiQueTask

TASK_REGISTRY: dict[str, type[TaskAdapter]] = {
    GSMHardTask.name: GSMHardTask,
    MathTask.name: MathTask,
    MuSiQueTask.name: MuSiQueTask,
    ConiferTask.name: ConiferTask,
    MultiplETask.name: MultiplETask,
}

__all__ = [
    "TASK_REGISTRY",
    "TaskAdapter",
    "Problem",
    "EvalResult",
    "Split",
    "build_task",
    "GSMHardTask",
    "MathTask",
    "MuSiQueTask",
    "ConiferTask",
    "MultiplETask",
]
