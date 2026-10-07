"""
IOHinspector: parent-algorithm run analysis toolkit.
"""

import iohinspector
except ModuleNotFoundError:
    iohinspector = None

from .tools import Tool, ToolContext


# AOCC precision limits (misc.utils.aoc_logger).
AOCC_LOWER, AOCC_UPPER = 1e-8, 1e2


class IOHAnalysisTool(Tool):

    """
    Parent-run convergence stats.

    Each solution must be logged via ioh.logger.Analyzer to <log_dir>/ioh/<solution.id>, for one function/dimension.

    Args:
    budget: Evaluations per run.
    """

    name = "performance_analysis"

    description = (
        "Parent-run stats: AOCC spread (1.0 best), median best-so-far gap (f − f_opt) at 10/25/50/100% budget, and median last-improvement evaluation."
    )

    parameters = {"type": "object", "properties": {}}

    def __init__(self, budget: int):
        self.budget = budget

    def run(self, context: ToolContext) -> dict:
        if context.log_dir is None:
            raise ValueError("Run data requires logging enabled.")
        return {pid: self._analyse(context.log_dir, pid) for pid in context.parent_ids}

    def _analyse(self, log_dir: str, parent_id: str) -> dict:
        raise NotImplementedError
