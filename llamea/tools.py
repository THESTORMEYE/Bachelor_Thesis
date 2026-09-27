"""
Agent-independent abstraction for tools.

The :class:`Tool` class models a capability that a language model may elect
to invoke. It is aware only of its own inputs and a minimal
:class:`ToolContext`. It remains unaware of the optimizer, the prompt, and
the agent that invokes it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """
    Information about the invocation site, exposed to tools.

    This object carries:
    - parent_ids: The IDs of the solutions being mutated. Useful for retrieving
    run data that the evaluation function saved for those solutions.
    - log_dir: The experiment's directory, provided when logging is turned on.
    - caller: The agent's name that initiated the call.
    """

    parent_ids: list = field(default_factory=list)
    log_dir: str | None = None
    caller: str = ""


class Tool(ABC):
    """
    Base class for tools.

    This class defines the interface for tool implementations. Subclasses must
    set the following attributes:

        name: A short identifier for the tool.
        description: Text explaining what the tool returns and when it is useful.
        parameters: A JSON schema describing the tool's arguments.

    Subclasses must also implement the :meth:`run` method. By convention, this
    method returns a small JSON-serializable dictionary of evidence (such as
    measurements, statistics, or metadata). It may optionally include a short
    "summary" string. The method should not instruct the model on what to do.


    """

    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}}

    @abstractmethod
    def run(self, context: TooThe lContext, **arguments) -> dict:
        """
        Execute the tool's core logic and return its output.

        This method implements the abstract :meth:`run` interface defined in the
        base class. The returned value must be JSON-serializable.

        Returns:
            A JSON-serializable object representing the tool's output.
        """

    def invoke(self, arguments, context: ToolContext) -> dict:
        """
        Check the arguments for validity, then execute the tool's logic.

        This method is designed never to raise an exception; any failure is captured
        and returned as part of the result.

        Returns:
            dict: A dictionary describing the outcome, using one of two shapes:
                - ``{"ok": True, "result": ...}`` if the tool ran successfully.
                - ``{"ok": False, "error": ...}`` if validation or execution failed.
        """

        if not isinstance(arguments, dict):
            return {"ok": False, "error": "`arguments` must be a JSON object."}
        allowed = self.parameters.get("properties", {})
        missing = [k for k in self.parameters.get("required", []) if k not in arguments]
        unknown = [k for k in arguments if k not in allowed]
        if missing or unknown:
            return {
                "ok": False,
                "error": f"Invalid arguments (missing: {missing}, unknown: {unknown}). "
                f"Expected schema: {self.parameters}",
            }
        try:
            return {"ok": True, "result": self.run(context, **arguments)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
