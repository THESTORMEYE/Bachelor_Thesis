"""
Quickly test the draft on unseen problem instances.
"""

from .tools import Tool, ToolContext


class QuickEvalTool(Tool):
    """
    Runs the draft from the current reply.

    Args:
    evaluate: Callable ``evaluate(context, runs) -> dict`` that runs
        ``context.draft``; may save run data under
        ``<log_dir>/quick_eval/<call_id>``. Must be picklable.
    max_runs: Max value for ``runs``.
    notes: Text describing what ``evaluate`` returns and costs; added to the
        tool description.
    """

    name = "quick_eval"

    def __init__(self, evaluate, max_runs: int, notes: str):
        self.evaluate = evaluate
        self.max_runs = max_runs
        self.description = (
            "Executes your code on function instances that scoring does not use. The code "
            "is taken from this reply, or otherwise from your latest reply that contains "
            "code, never from the prompt; the first call must include your code, in the "
            "required format, in the same reply as the tool block. "
            f"More runs yield a more reliable estimate. {notes}"
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
            raise ValueError(
                "No usable code found. Reply again with the complete code in the "
                "required format and the tool block."
            )
        if context.log_dir is None:
            raise ValueError("Logging must be enabled for quick evaluation.")
        if not isinstance(runs, int) or not 1 <= runs <= self.max_runs:
            raise ValueError(
                f"`runs` has to be an integer between 1 and {self.max_runs}."
            )
        return self.evaluate(context, runs)
