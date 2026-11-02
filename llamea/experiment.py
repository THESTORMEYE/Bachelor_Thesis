"""
Run a model-written experiment on the written code.
"""

import os
import subprocess
import sys
import tempfile

from .tools import Tool, ToolContext


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
            "Takes the experiment block in this reply and executes it as a "
            "Python script alongside the complete code, which must be given in "
            "the required format in the same reply; the script can pull in "
            "the latter with `from candidate import <ClassName>`. Whatever it "
            "prints comes back. numpy, scipy and ioh are on hand. Ceiling of "
            f"{timeout} s; per stream, only the trailing "
            f"{max_output_chars} characters survive."
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
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                result = subprocess.CompletedProcess(e.cmd, None, e.stdout, e.stderr)
        return {
            "returncode": result.returncode,
            "stdout": self._tail(result.stdout),
            "stderr": self._tail(result.stderr),
            "timed_out": result.returncode is None,
        }

    def _tail(self, text) -> str:
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        return (text or "")[-self.max_output_chars :]
