"""
Tool-enabled agent.

Encapsulates an :class:`~llamea.llm.LLM` instance and delegates tool-use
decisions to the underlying model. Prior to emitting its final solution, the
model can choose whether any tools are needed and, if so, which specific tools
to invoke.
"""

import json
import re
import textwrap
import time
from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from .tools import Tool, ToolContext
from .utils import NoCodeException


def _flatten(items):
    for item in items or []:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


class Agent:
    """
    Couples an :class:`~llamea.llm.LLM` with a collection of tools and runs a
    model-driven loop that governs tool usage.

    ``Agent`` implements the methods LLaMEA expects from an ``LLM`` (``model``,
    ``query``, ``sample_solution``, and ``set_logger``), so it can be supplied
    wherever an ``LLM`` is accepted, e.g.
    ``LLaMEA(f, llm=Agent(llm, tools=[...]))``. When the tool list is empty, the
    agent forwards to the wrapped LLM and keeps an empty ``tool_calls`` record.

    Tool loop:
        The tool list is embedded in the prompt. The model may reply with one or
        more fenced ``tool`` blocks, each of the form
        ``{"name": ..., "arguments": {...}}``. For every such block, the agent
        invokes the named tool, appends the result to the conversation, and issues
        another query. The first reply that lacks tool blocks is treated as the
        solution, unless it is the model's opening reply, which gets one reminder
        to call a tool. If ``max_tool_rounds`` is reached, the model is prompted to
        answer directly without tools. If code is omitted, the previous version is
        used.

    Persistence:
        Every call lands in ``solution.metadata["tool_calls"]`` and the tools'
        shared state in ``solution.metadata["tool_state"]``. When logging is on,
        each complete entry (tool outputs, the reply's draft code and experiment
        script, and how long the call took) goes to ``tool_calls.jsonl``.

    Args:
        llm: The LLM that handles all queries and extracts the final answer.
        tools: The collection of tools exposed to the model.
        name: A label for the agent, used in traces and forwarded to each tool.
        max_tool_rounds: The maximum number of tool-invocation cycles allowed for
            one solution.
        max_result_chars: If a tool returns more than this many characters, the
            excess is removed from the message sent back to the model; the log
            retains the full output.
    """

    call_pattern = re.compile(r"```tool\s*\n(.*?)\n\s*```", re.DOTALL)
    experiment_pattern = re.compile(r"```experiment\s*\n(.*?)\n\s*```", re.DOTALL)

    def __init__(
        self,
        llm,
        tools: list[Tool] = (),
        name: str = "llamea",
        max_tool_rounds: int = 3,
        max_result_chars: int = 4000,
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique.")
        self.name = name
        self.max_tool_rounds = max_tool_rounds
        self.max_result_chars = max_result_chars
        self.logger = None

    @property
    def model(self):
        return self.llm.model

    def set_logger(self, logger):
        self.logger = logger
        self.llm.set_logger(logger)

    def query(self, session_messages):
        """
        A straightforward model query that does not invoke any tools.
        """
        return self.llm.query(session_messages)

    def tool_instructions(self) -> str:
        tools = "\n".join(
            f"- {t.name}: {t.description}\n  arguments (JSON schema): "
            f"{json.dumps(t.parameters)}"
            for t in self.tools.values()
        )
        return textwrap.dedent("""
            ## Tools
            Before giving your final answer, call at least one tool; tool calls do not affect
            your score. Call a tool with a block of the form below, either by itself or
            alongside your code:

            ```tool
            {"name": "<tool name>", "arguments": {<arguments>}}
            ```

            Results are then returned to you. You may perform at most %d rounds of tool calls. A
            response with no tool blocks is treated as your final answer: return it in the required
            format, but you may leave out the code if it has not changed since your last reply
            containing code.

            Available tools:
            """) % self.max_tool_rounds + tools

    def sample_solution(
        self,
        session_messages: list,
        parent_ids: list | None = None,
        HPO: bool = False,
        base_code: str | None = None,
        diff_mode: bool = False,
    ):
        """
        Execute the tool-invocation cycle, then interpret the resulting final
        answer as a ``Solution``.

        The method's signature and the exceptions it may raise are identical to
        those of :meth:`LLM.sample_solution`.
        """

        messages = [dict(m) for m in session_messages]
        if self.tools:
            messages[-1]["content"] += self.tool_instructions()
        context = ToolContext(
            parent_ids=list(_flatten(parent_ids)),
            log_dir=getattr(self.logger, "dirname", None),
            caller=self.name,
        )
        records = []
        solution = None
        draft = None

        def parse(reply):
            reply = self.experiment_pattern.sub("", self.call_pattern.sub("", reply))
            return self.llm.parse_solution(
                reply,
                parent_ids,
                HPO=HPO,
                base_code=base_code,
                diff_mode=diff_mode,
            )

        def latest_draft(reply):
            try:
                return parse(reply)
            except NoCodeException:
                return draft
            except Exception:
                return None

        try:
            response = self._query(messages)
            if self.tools and not self.call_pattern.search(response):
                messages.append({"role": "assistant", "content": response})
                draft = latest_draft(response)
                messages.append(
                    {
                        "role": "user",
                        "content": "No tool call has been made yet. Call at least "
                        "one tool before your final answer.",
                    }
                )
                response = self._query(messages)
            for round_ in range(1, self.max_tool_rounds + 2):
                calls = self.call_pattern.findall(response) if self.tools else []
                if not calls:
                    break
                messages.append({"role": "assistant", "content": response})
                draft = latest_draft(response)
                if round_ > self.max_tool_rounds:
                    messages.append(
                        {
                            "role": "user",
                            "content": "No rounds of tool calls are left, so these "
                            "calls were not run. Give your final answer now, without "
                            "tool blocks.",
                        }
                    )
                else:
                    script = self.experiment_pattern.search(response)
                    round_context = replace(
                        context, draft=draft, script=script and script.group(1)
                    )
                    results = [self._call(c, round_context, round_) for c in calls]
                    records += results
                    messages.append(
                        {"role": "user", "content": self._format(results, round_)}
                    )
                response = self._query(messages)
            try:
                solution = parse(response)
            except NoCodeException:
                if draft is None:
                    raise
                solution = draft
            solution.add_metadata(
                "tool_calls",
                [
                    {k: r[k] for k in ("round", "tool", "arguments", "ok")}
                    for r in records
                ],
            )
            solution.add_metadata("tool_state", context.state)
            return solution
        finally:
            self._log(records, solution, context)

    def _query(self, messages):
        if self.logger is not None:
            self.logger.log_conversation("client", messages[-1]["content"])
        response = self.llm.query(messages)
        if self.logger is not None:
            self.logger.log_conversation(self.model, response)
        return response

    def _call(self, raw, context, round_):
        start = time.perf_counter()
        context = replace(context, call_id=uuid4().hex[:8])
        try:
            call = json.loads(raw)
            name, arguments = call["name"], call.get("arguments", {})
        except (ValueError, KeyError, TypeError) as e:
            name, arguments = None, raw
            output = {"ok": False, "error": f"Malformed tool call ({e}): {raw}"}
        else:
            tool = self.tools.get(name)
            if tool is None:
                output = {
                    "ok": False,
                    "error": f"Unknown tool {name!r}. "
                    f"Available: {sorted(self.tools)}",
                }
            else:
                output = tool.invoke(arguments, context)
        return {
            "time": f"{datetime.now()}",
            "call_id": context.call_id,
            "round": round_,
            "tool": name,
            "arguments": arguments,
            "ok": output["ok"],
            "output": output,
            "draft": getattr(context.draft, "code", None),
            "script": context.script,
            "duration": time.perf_counter() - start,
        }

    def _format(self, results, round_):
        parts = []
        for r in results:
            text = json.dumps(r["output"], default=str)
            if len(text) > self.max_result_chars:
                text = text[: self.max_result_chars] + " ...[truncated]"
            parts.append(f"### {r['tool']}\n{text}")
        left = self.max_tool_rounds - round_
        return (
            "Tool results:\n\n"
            + "\n\n".join(parts)
            + f"\n\nYou have {left} round(s) of tool calls left."
        )

    def _log(self, records, solution, context):
        if self.logger is None:
            return
        for r in records:
            self.logger.log_tool_call(
                {
                    "agent": self.name,
                    "solution_id": getattr(solution, "id", None),
                    "parent_ids": context.parent_ids,
                    **r,
                }
            )
