import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
from contextlib import contextmanager

import numpy as np
from filelock import FileLock

from llamea import (
    Agent,
    BBOBELATool,
    BBOBOracleTool,
    Claude_LLM,
    DeepSeek_LLM,
    ExperimentTool,
    Gemini_LLM,
    IOHAnalysisTool,
    LLaMEA,
    QuickEvalTool,
)
from llamea.experiment import run_limited
from llamea.ioh_analysis import METRICS, analyse_runs
from llamea.tools import Tool

MODEL = "claude-haiku-5-5"
FID = 3
DIM = 5
IIDS = [1, 2, 3]
SEEDS = [0, 1, 2]
BUDGET = 2000 * DIM
REPETITIONS = 10
MAX_TOOL_ROUNDS = 5
QUICK_PAIRS = [(4, 0), (5, 0)]
TARGET_OFFSET = 1000
MAX_ALGORITHMS = 4
EVAL_TIMEOUT = 600
MAX_MEMORY = 2 * 2**30
SCORER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbob_scoring.py")
SCORING = len(IIDS) * len(SEEDS) * BUDGET
TOOL_EVALS = len(QUICK_PAIRS) * BUDGET
ELA_EVALS = 236_862
FREE = (
    "no_tools",
    "placebo",
    "ioh",
)
COST = (
    "comes from the same pool that pays for scoring algorithms; each answer "
    f"can use no more than {TOOL_EVALS}"
)

TARGET_MODULE = textwrap.dedent("""
    import ioh
    import numpy as np

    _left = {evaluations}
    _used = open("evaluations", "ab", buffering=0)


    def problem(instance, fid=None, dimension={dim}):
        if fid is None:
            fid, instance = {fid}, instance + {offset}
        target = ioh.get_problem(fid, instance, dimension)

        def f(x):
            global _left
            x = np.asarray(x, dtype=float)
            if x.ndim != 1:
                raise ValueError("x must be a single point")
            if _left < 1:
                raise RuntimeError("No evaluations left.")
            _left -= 1
            _used.write(b".")
            return target(x) - target.optimum.y

        f.bounds = target.bounds
        return f
    """)


@contextmanager
def pool(log_dir):
    """The run's evaluation pool; locked for reads and updates."""
    path = os.path.join(log_dir, "evaluations.json")
    with FileLock(path + ".lock"):
        with open(path) as f:
            data = json.load(f)
        yield data
        with open(path, "w") as f:
            json.dump(data, f)


def draw(log_dir, wanted):
    """Withdraw up to `wanted` evaluations from the run's pool; a negative amount returns them."""
    with pool(log_dir) as data:
        granted = min(wanted, data["left"])
        data["left"] -= granted
    return granted


def draw_for_tool(context, wanted):
    """Same as `draw`, but limited to the solution's remaining tool budget."""
    spent = context.state.get("evaluations", 0)
    granted = draw(context.log_dir, min(wanted, TOOL_EVALS - spent))
    context.state["evaluations"] = spent + granted
    return granted


class PooledLLaMEA(LLaMEA):
    """
    Offspring are created only while the run's pool can cover their scoring.
    Each solution reserves `SCORING`; whatever scoring leaves of that reserve is returned.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        total = self.budget * SCORING
        left = total - self.n_parents * SCORING
        with open(os.path.join(self.logger.dirname, "evaluations.json"), "w") as f:
            json.dump({"total": total, "left": left, "charged": {}}, f)
        self.budget *= MAX_ALGORITHMS
        self.settled = 0

    def _select_parents(self, count=None):
        if count is None:
            unused = sum(
                SCORING - solution.metadata.get("evaluations", 0)
                for solution in self.run_history[self.settled :]
            )
            self.settled = len(self.run_history)
            draw(self.logger.dirname, -unused)
            granted = draw(self.logger.dirname, self.n_offspring * SCORING)
            draw(self.logger.dirname, -(granted % SCORING))
            count = granted // SCORING
            if count == 0:
                self.budget = 0
        return super()._select_parents(count)


def run_bbob(solution, pairs, folder, used, timeout):
    """
    AOCC for every (instance, seed) run is computed in a child process by bbob_scoring.py;
    the evaluations from each run go into `used`. Any algorithm that exceeds `timeout`
    seconds or MAX_MEMORY is terminated and charged in full.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        job = {
            "code": solution.code,
            "name": solution.name,
            "fid": FID,
            "dim": DIM,
            "budget": BUDGET,
            "pairs": pairs,
            "folder": os.path.abspath(folder),
            "result": os.path.join(tmp, "result.pkl"),
        }
        _, killed = run_limited(
            [sys.executable, SCORER],
            timeout,
            MAX_MEMORY,
            input=json.dumps(job).encode(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if killed or not os.path.exists(job["result"]):
            used.extend([BUDGET] * len(pairs))
            if killed == "timeout":
                raise TimeoutError(
                    f"Exceeded the evaluation timeout of {timeout:.0f} s"
                )
            if killed == "memory":
                raise MemoryError(
                    f"Exceeded the memory limit of {MAX_MEMORY / 2**30:.0f} GB"
                )
            raise RuntimeError("The algorithm ended the process that runs it.")
        with open(job["result"], "rb") as f:
            result = pickle.load(f)
    used.extend(result["used"])
    if "aucs" not in result:
        reraise(result)
    return result["aucs"]


def reraise(result):
    """Re-raises the child's error at the matching line in the algorithm's code."""
    try:
        error = pickle.loads(result["error"])
    except Exception:
        error = None
    if not isinstance(error, Exception):
        error = type(result["type"], (Exception,), {})(result["message"])
    if result["line"] is None:
        raise error
    code = "\n" * (result["line"] - 1) + "raise error"
    exec(compile(code, "<string>", "exec"), {"error": error})


def evaluate(solution, explogger):
    pairs = [(iid, seed) for iid in IIDS for seed in SEEDS]
    used = []
    try:
        aucs = run_bbob(
            solution,
            pairs,
            os.path.join(explogger.dirname, "ioh", solution.id),
            used,
            timeout=EVAL_TIMEOUT,
        )
    finally:
        solution.add_metadata("evaluations", sum(used))
    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    feedback = (
        f"{solution.name} scored a mean AOCC (area over the convergence curve, 1.0 "
        f"is the best) of {auc_mean:0.2f} with standard deviation {auc_std:0.2f} "
        f"over {len(aucs)} runs."
    )
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)
    return solution


def quick_evaluate(context, runs):
    granted = draw_for_tool(context, runs * BUDGET)
    if granted < runs * BUDGET:
        draw_for_tool(context, -granted)
        raise ValueError(
            f"Only {granted} evaluations are left, enough for "
            f"{granted // BUDGET} run(s)."
        )
    folder = os.path.join(context.log_dir, "quick_eval", context.call_id)
    used = []
    try:
        run_bbob(
            context.draft,
            QUICK_PAIRS[:runs],
            folder,
            used,
            timeout=EVAL_TIMEOUT * runs * BUDGET / SCORING,
        )
    finally:
        draw_for_tool(context, sum(used) - granted)
    result = analyse_runs(folder, BUDGET)
    if "error" in result:
        raise ValueError(f"{context.draft.name} made no function evaluations.")
    return {
        **result,
        "tool_evaluations_left": TOOL_EVALS - context.state["evaluations"],
    }


@contextmanager
def target_problem(context, folder):
    granted = draw_for_tool(context, TOOL_EVALS)
    with open(os.path.join(folder, "target.py"), "w") as f:
        f.write(
            TARGET_MODULE.format(
                evaluations=granted,
                fid=FID,
                dim=DIM,
                offset=TARGET_OFFSET,
            )
        )
    extra = {}
    try:
        yield extra
    finally:
        used = os.path.join(folder, "evaluations")
        used = os.path.getsize(used) if os.path.exists(used) else 0
        draw_for_tool(context, used - granted)
        extra["tool_evaluations_left"] = TOOL_EVALS - context.state["evaluations"]


class PaidOnFirstCall(Tool):
    """The pool is charged for `tool.evaluations` on the run's first call to `tool`."""

    def __init__(self, tool):
        self.tool = tool
        self.name = tool.name
        self.description = (
            f"{tool.description} Charging occurs once per tool: the first call, "
            f"whether from this answer or another, costs {tool.evaluations} "
            "function evaluations from the pool that also funds scoring "
            "algorithms; later calls are free."
        )
        self.parameters = tool.parameters

    def run(self, context, **arguments):
        cost = self.tool.evaluations
        with pool(context.log_dir) as data:
            if self.name not in data["charged"]:
                if data["left"] < cost:
                    raise ValueError(
                        f"Only {data['left']} evaluations are left; the first "
                        f"call costs {cost}."
                    )
                data["left"] -= cost
                data["charged"][self.name] = cost
                context.state["charged"] = context.state.get("charged", 0) + cost
        return self.tool.run(context, **arguments)


def charged_oracle():
    """Oracle priced like ELA, so both consume the same pool."""
    tool = BBOBOracleTool(fid=FID)
    tool.evaluations = ELA_EVALS
    return PaidOnFirstCall(tool)


class PooledAgent(Agent):
    """Agent that reports, beneath the tool list, the run pool's remaining balance."""

    def tool_instructions(self):
        with pool(self.logger.dirname) as data:
            left, total, charged = data["left"], data["total"], data["charged"]
        status = f"\n\nFunction evaluations left in the pool: {left} of {total}."
        for name in charged:
            if name in self.tools:
                status += f" {name} has already been paid for; calling it is free."
        return super().tool_instructions() + status


class PlaceboTool(Tool):
    """Control: adds one extra round but returns no information."""

    name = "extra_round"
    description = (
        "Yields no information, but grants another reply for revising your "
        "code before the final answer."
    )

    def run(self, context):
        return {"status": "round granted"}


task_prompt = textwrap.dedent(f"""
    Write a strong, novel heuristic optimizer in Python that minimizes a single black-box
    function; it is scored on {len(IIDS)} instances of that function. Write a class with
    `__init__(self, budget, dim)` and `__call__(self, func)` that calls `func` at most
    `self.budget` times. The search space is [-5.0, 5.0]^{DIM}.
    """)

example_prompt = textwrap.dedent(f"""
    An example of such code (a simple random search):
    ```python
    import numpy as np

    class RandomSearch:
        def __init__(self, budget={BUDGET}, dim={DIM}):
            self.budget = budget
            self.dim = dim
            self.f_opt = np.inf
            self.x_opt = None

        def __call__(self, func):
            for i in range(self.budget):
                x = np.random.uniform(func.bounds.lb, func.bounds.ub)

                f = func(x)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

            return self.f_opt, self.x_opt
    ```
    """)

output_format_prompt = textwrap.dedent("""
    Give a one-line description of the main idea and the complete Python code, in this
    format:
    # Description: <one-line description>
    # Code:
    ```python
    <code>
    ```
    """)


if __name__ == "__main__":
    tools = {
        "no_tools": lambda: [],
        "placebo": lambda: [PlaceboTool()],
        "ioh": lambda: [IOHAnalysisTool(budget=BUDGET)],
        "oracle": lambda: [charged_oracle()],
        "ela": lambda: [PaidOnFirstCall(BBOBELATool(fid=FID, iids=IIDS, dim=DIM))],
        "quick_eval": lambda: [
            QuickEvalTool(
                quick_evaluate,
                max_runs=len(QUICK_PAIRS),
                notes="A run costs the function evaluations it makes (at most "
                f"{BUDGET}), {COST}. {METRICS}",
            )
        ],
        "run_experiment": lambda: [
            ExperimentTool(
                timeout=60,
                max_memory=MAX_MEMORY,
                max_output_chars=1000,
                setup=target_problem,
                notes="Access test functions only via `from target import problem`: "
                "`f = problem(instance)` is the black-box function, for any "
                "instance number in its own numbering, and `problem(instance, fid)` "
                "is BBOB function `fid` (1-24); both take an optional `dimension`. "
                "`f` is called as `f(x)` for one point, returns the gap f - f_opt and "
                "has `f.bounds`, so your optimizer can run on it. Each call of a "
                "function returned by `problem` costs one function evaluation, "
                f"{COST}; after that, calls raise an error. numpy and scipy are "
                "available.",
            )
        ],
    }
    tools["all"] = lambda: [
        tool
        for condition in ("ioh", "ela", "quick_eval", "run_experiment")
        for tool in tools[condition]()
    ]
    tools["all_unforced"] = tools["all"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=tools, default=list(tools))
    parser.add_argument(
        "--repetitions", nargs="+", type=int, default=range(REPETITIONS)
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--budget", type=int, default=26)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--fid", type=int, default=FID)
    parser.add_argument("--out", default=".", help="directory for the run folders")
    args = parser.parse_args()
    FID = args.fid
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    tools = {condition: tools[condition]() for condition in args.conditions}
    if args.model.startswith("claude-"):
        llm = Claude_LLM(os.getenv("LLAMEA_ANTHROPIC_KEY"), args.model)
    elif args.model.startswith("deepseek-"):
        llm = DeepSeek_LLM(os.getenv("DEEPSEEK_API_KEY"), args.model)
    else:
        llm = Gemini_LLM(os.getenv("GEMINI_API_KEY"), args.model)

    for repetition in args.repetitions:
        for condition in args.conditions:
            name = f"{condition}-f{FID}-{DIM}D-{repetition}"
            if glob.glob(f"exp-*-{args.model}-{name}/done"):
                print(condition, repetition, "already done")
                continue
            es = PooledLLaMEA(
                evaluate,
                n_parents=2,
                n_offspring=5,
                llm=(Agent if condition in FREE else PooledAgent)(
                    llm,
                    tools=tools[condition],
                    max_tool_rounds=MAX_TOOL_ROUNDS,
                    require_tool_call=condition != "all_unforced",
                ),
                task_prompt=task_prompt,
                example_prompt=example_prompt,
                output_format_prompt=output_format_prompt,
                experiment_name=name,
                elitism=True,
                HPO=False,
                budget=args.budget,
                eval_timeout=2 * EVAL_TIMEOUT, 
                max_workers=args.max_workers,
            )
            try:
                best = es.run()
            except Exception as e:
                print(condition, repetition, f"failed: {e!r}")
                continue
            if es.budget:
                print(condition, repetition, "stopped at the algorithm limit")
                continue
            open(os.path.join(es.logger.dirname, "done"), "w").close()
            print(condition, repetition, best.fitness)
