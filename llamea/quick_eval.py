"""
Quickly test the draft on unseen problem instances.
"""

import os

from .tools import Tool, ToolContext


class QuickEvalTool(Tool):
    """
    Runs the draft from the current reply.

    Args:
    evaluate: Callable ``evaluate(solution, runs, folder) -> dict``; may save run
        data under ``<log_dir>/quick_eval/<call_id>``. Must be picklable.
    max_runs: Max value for ``runs``.
    metrics: Text describing what ``evaluate`` returns; added to the tool
        description.
    """

    name = "quick_eval"

    def __init__(self, evaluate, max_runs: int, metrics: str):
        self.evaluate = evaluate
        self.max_runs = max_runs
        self.description = (
            "Tests the code block in this reply against unseen problem instances "
            "rather than the ones used for scoring. Additional runs yield a "
            f"more dependable estimate. {metrics}"
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "runs": {"type": "integer", "minimum": 1, "maximum": max_runs}
            },
            "required": ["runs"],
        }

    def run(self, context: ToolContext, runs: int) -> dict:
        if context.draft is None:
            raise ValueError("This reply contains no code block.")
        if context.log_dir is None:
            raise ValueError("Logging must be enabled for quick evaluation.")
        if not isinstance(runs, int) or not 1 <= runs <= self.max_runs:
            raise ValueError(
                f"`runs` has to be an integer between 1 and {self.max_runs}."
            )
        folder = os.path.join(context.log_dir, "quick_eval", context.call_id)
        return self.evaluate(context.draft, runs, folder)
