"""
Run a model-written experiment on the written code.
"""

import math
import os
import subprocess
import sys
import tempfile
import time

import psutil

from .tools import Tool, ToolContext

ENV_KEYS = ("PATH", "SYSTEMROOT")


def run_limited(args, timeout, max_memory=math.inf, input=None, **kwargs):
    """
    Like `subprocess.run`, but kills the process and children on `timeout` 
    seconds or `max_memory` bytes combined; returns `CompletedProcess` and kill reason ("timeout", "memory" or None).
    """
    process = psutil.Popen(args, **kwargs)
    end = time.monotonic() + timeout
    reason = None
    while True:
        try:
            out, err = process.communicate(input, timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            input = None  # already sent
        try:
            tree = [process, *process.children(recursive=True)]
            memory = sum(_memory(p) for p in tree)
        except psutil.NoSuchProcess:
            continue
        if time.monotonic() > end:
            reason = "timeout"
        elif memory > max_memory:
            reason = "memory"
        if reason:
            for p in tree:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            out, err = process.communicate()
            break
    return subprocess.CompletedProcess(args, process.returncode, out, err), reason


def _memory(process):
    """Amount of Bytes held""" 
    info = process.memory_info()
    return getattr(info, "private", info.rss)


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
    max_memory: Kill the script if more than this many bytes.
    """

    name = "run_experiment"

    def __init__(
        self,
        timeout: float,
        max_output_chars: int,
        setup,
        notes: str,
        max_memory: float = math.inf,
    ):
        self.timeout = timeout
        self.max_memory = max_memory
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
            result, killed = run_limited(
                [sys.executable, "experiment.py"],
                self.timeout,
                self.max_memory,
                cwd=tmp,
                env={k: os.environ[k] for k in ENV_KEYS if k in os.environ},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        return {
            "returncode": None if killed else result.returncode,
            "stdout": self._tail(result.stdout, tmp),
            "stderr": self._tail(result.stderr, tmp),
            "timed_out": killed == "timeout",
            **({"memory_exceeded": True} if killed == "memory" else {}),
            **extra,
        }

    def _tail(self, text, tmp: str) -> str:
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        text = text or ""
        for prefix in sorted({tmp, sys.prefix, sys.base_prefix}, key=len, reverse=True):
            text = text.replace(prefix + os.sep, "")
        return text[-self.max_output_chars :]
