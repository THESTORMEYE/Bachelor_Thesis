"""
Run a model-written experiment on the written code.
"""

import os
import subprocess
import sys
import tempfile

from .tools import Tool, ToolContext

ENV_KEYS = ("PATH", "SYSTEMROOT")


class ExperimentTool(Tool):
    """
    Puts the reply's experiment script in a scratch dir together with the draft
    (written out as ``candidate.py``) and executes it there.

    Args:
    timeout: Kill the script after this many seconds.
    max_output_chars: How many trailing characters to keep per stream.
    setup: Factory for a context manager ``setup(context, folder)`` that wraps the
        script run; it can add files to the scratch dir. Yields a dict merged
        into the result. Must be picklable.
    notes: Text explaining what ``setup`` gives the script; appended to the tool
        description.
    """

    name = "run_experiment"

    def __init__(self, timeout: float, max_output_chars: int, setup, notes: str):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.setup = setup

        self.description = (
            "Runs this reply's experiment block as a Python script, returning "
            f"the final {max_output_chars} characters of stdout and stderr. "
            "The script may import your code via `from candidate import "
            "<ClassName>`. Your code comes from this reply. If this reply has no code, it comes from your "
            f"most recent reply containing code. It never comes from the prompt. {notes} "
            f"Timeout: {timeout} s. The initial call requires three blocks in one reply: "
            "your code in the required format, then\n"
            "```experiment\n<python script>\n```\n"
            "```tool\n"
            '{"name": "run_experiment", "arguments": {}}\n'
            "```"
        )

    def run(self, context: ToolContext) -> dict:
        if context.draft is None:
            raise ValueError(
                "No usable code found. Reply again with the complete code in the "
                "required format, the experiment block and the tool block."
            )
        if context.script is None:
            raise ValueError(
                "This reply has no experiment block. Include one in the same "
                "reply as the tool block."
            )
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.setup(context, tmp) as extra,
        ):
            for filename, text in [
                ("candidate.py", context.draft.code),
                ("experiment.py", context.script),
            ]:
                with open(os.path.join(tmp, filename), "w", encoding="utf-8") as f:
                    f.write(text)
            try:
                result = subprocess.run(
                    [sys.executable, "experiment.py"],
                    cwd=tmp,
                    env={k: os.environ[k] for k in ENV_KEYS if k in os.environ},
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                result = subprocess.CompletedProcess(e.cmd, None, e.stdout, e.stderr)
        return {
            "returncode": result.returncode,
            "stdout": self._tail(result.stdout, tmp),
            "stderr": self._tail(result.stderr, tmp),
            "timed_out": result.returncode is None,
            **extra,
        }

    def _tail(self, text, tmp: str) -> str:
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        text = text or ""
        for prefix in sorted({tmp, sys.prefix, sys.base_prefix}, key=len, reverse=True):
            text = text.replace(prefix + os.sep, "")
        return text[-self.max_output_chars :]
