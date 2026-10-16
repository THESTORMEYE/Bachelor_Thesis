"""
IOHinspector: parent-algorithm run analysis toolkit.
"""

try:
    import iohinspector
    import polars as pl
except ModuleNotFoundError:
    iohinspector = None

import os

from .tools import Tool, ToolContext

# AOCC precision limits (misc.utils.aoc_logger).
AOCC_LOWER, AOCC_UPPER = 1e-8, 1e2
CHECKPOINTS = (0.1, 0.25, 0.5, 1.0)


class IOHAnalysisTool(Tool):
    """
    Parent-run convergence stats.

    Each solution must be logged via ioh.logger.Analyzer to <log_dir>/ioh/<solution.id>, for one function/dimension.

    Args:
    budget: Evaluations per run.
    """

    name = "performance_analysis"

    description = (
        "Captures how your parent algorithms performed in practice, before you "
        "mutate them: convergence speed, stagnation points, and consistency. "
        "Metrics: AOCC dispersion (1.0 = best), median best-so-far gap "
        "(f − f_opt) at 10/25/50/100% of budget, and median last-improvement."
    )

    parameters = {"type": "object", "properties": {}}

    def __init__(self, budget: int):
        if iohinspector is None:
            raise ImportError(
                "IOHAnalysisTool depends on iohinspector. Please install it with: `uv sync --dev`"
            )
        self.budget = budget

    def run(self, context: ToolContext) -> dict:
        if context.log_dir is None:
            raise ValueError("Run data requires logging enabled.")
        return {pid: self._analyse(context.log_dir, pid) for pid in context.parent_ids}

    def _analyse(self, log_dir: str, parent_id: str) -> dict:
        manager = iohinspector.DataManager()
        try:
            manager.add_folder(os.path.join(log_dir, "ioh", parent_id))
        except FileNotFoundError:
            return {"error": "run data unavailable"}

        data = manager.load(monotonic=True)
        aocc = iohinspector.get_aocc(
            iohinspector.transform_fval(data, lb=AOCC_LOWER, ub=AOCC_UPPER),
            self.budget,
            group_cols=["run_id"],
        )["AOCC"]

        budgets = [int(f * self.budget) for f in CHECKPOINTS]
        aligned = manager.load(monotonic=True, x_values=budgets)
        best_so_far = {
            f"{f:.0%}": aligned.filter(pl.col("evaluations") == b)["raw_y"].median()
            for f, b in zip(CHECKPOINTS, budgets)
        }

        last = data.group_by("run_id").agg(pl.col("evaluations").max())
        return {
            "runs": len(aocc),
            "aocc": {
                "min": aocc.min(),
                "median": aocc.median(),
                "max": aocc.max(),
            },
            "best_so_far": best_so_far,
            "last_improvement": last["evaluations"].median(),
        }
