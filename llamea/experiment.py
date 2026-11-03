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
    """

    name = "run_experiment"

    def __init__(self, timeout: float, max_output_chars: int):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.description = (
            "Runs the experiment block in this reply as a Python script beside "
            "your latest complete code; the script can import that code with "
            "`from candidate import <ClassName>`. Its printed output is returned. "
            "You design the experiment. numpy, scipy, and ioh are available "
            "(`ioh.get_problem(fid, instance, dimension)`, called as `problem(x)`). "
            f"Timeout: {timeout} s; each stream keeps only its last "
            f"{max_output_chars} characters. Format:\n"
            "  ```experiment\n  <python script>\n  ```"
)

    def run(self, context: ToolContext) -> dict:
        if context.draft is None:
            raise ValueError(
                "This reply has no code block. Include the complete code in the "
                "required format in the same reply as the tool block."
            )
        if context.script is None:
            raise ValueError(
                "This reply has no experiment block. Include one in the same "
                "reply as the tool block."
            )
        with tempfile.TemporaryDirectory() as tmp:
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
        }

    def _tail(self, text, tmp: str) -> str:
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        text = text or ""
        for prefix in sorted({tmp, sys.prefix, sys.base_prefix}, key=len, reverse=True):
            text = text.replace(prefix + os.sep, "")
        return text[-self.max_output_chars :]
