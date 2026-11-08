import argparse
import os
import random
import textwrap

import ioh
import numpy as np

from llamea import (
    Agent,
    BBOBELATool,
    BBOBOracleTool,
    Claude_LLM,
    ExperimentTool,
    Gemini_LLM,
    IOHAnalysisTool,
    LLaMEA,
    QuickEvalTool,
)
from llamea.ioh_analysis import METRICS, analyse_runs
from misc import OverBudgetException, aoc_logger, correct_aoc

MODEL = "claude-haiku-5-5"
FID = 3
DIM = 5
IIDS = [1, 2, 3]
SEEDS = [0, 1, 2]
BUDGET = 2000 * DIM
REPETITIONS = 3
MAX_TOOL_ROUNDS = 5
QUICK_PAIRS = [(4, 0), (5, 0), (4, 1), (5, 1), (4, 2), (5, 2)]


def run_bbob(solution, pairs, folder):
    namespace = {}
    exec(solution.code, namespace)
    root, folder_name = os.path.split(folder)
    analyzer = ioh.logger.Analyzer(
        root=root, folder_name=folder_name, algorithm_name=solution.name
    )
    l2 = aoc_logger(budget=BUDGET, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS])
    loggers = ioh.logger.Combine([l2, analyzer])
    aucs = []

    try:
        for iid, seed in pairs:
            problem = ioh.get_problem(FID, iid, DIM)
            problem.attach_logger(loggers)
            random.seed(seed)
            np.random.seed(seed)
            try:
                namespace[solution.name](budget=BUDGET, dim=DIM)(problem)
            except OverBudgetException:
                pass
            aucs.append(correct_aoc(problem, l2, BUDGET))
            l2.reset(problem)
            problem.reset()
    finally:
        analyzer.close()
    return aucs


def evaluate(solution, explogger):
    pairs = [(iid, seed) for iid in IIDS for seed in SEEDS]
    aucs = run_bbob(
        solution, pairs, os.path.join(explogger.dirname, "ioh", solution.id)
    )
    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    feedback = f"The algorithm {solution.name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)
    return solution


def quick_evaluate(solution, runs, folder):
    run_bbob(solution, QUICK_PAIRS[:runs], folder)
    result = analyse_runs(folder, BUDGET)
    if "error" in result:
        raise ValueError(f"{solution.name} made no function evaluations.")
    return result


task_prompt = textwrap.dedent(f"""
    Write a Python heuristic optimizer for a single black-box function tested on {len(IIDS)}
    instances. Implement `__init__(self, budget, dim)` and `__call__(self, func)`; call `func`
    at most `self.budget` times. Domain: [-5.0, 5.0]; dimension: {DIM}. Provide a strong, novel
    heuristic and a one-line description of its core idea.
    """)


if __name__ == "__main__":
    tools = {
        "no_tools": lambda: [],
        "ioh": lambda: [IOHAnalysisTool(budget=BUDGET)],
        "oracle": lambda: [BBOBOracleTool(fid=FID)],
        "ela": lambda: [BBOBELATool(fid=FID, iids=IIDS, dim=DIM)],
        "quick_eval": lambda: [
            QuickEvalTool(quick_evaluate, max_runs=len(QUICK_PAIRS), metrics=METRICS)
        ],
        "run_experiment": lambda: [ExperimentTool(timeout=60, max_output_chars=1000)],
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=tools, default=list(tools))
    parser.add_argument(
        "--repetitions", nargs="+", type=int, default=range(REPETITIONS)
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--budget", type=int, default=26)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()
    tools = {condition: tools[condition]() for condition in args.conditions}
    if args.model.startswith("claude-"):
        llm = Claude_LLM(os.getenv("LLAMEA_ANTHROPIC_KEY"), args.model)
    else:
        llm = Gemini_LLM(os.getenv("GEMINI_API_KEY"), args.model)

    for repetition in args.repetitions:
        for condition in args.conditions:
            es = LLaMEA(
                evaluate,
                n_parents=2,
                n_offspring=5,
                llm=Agent(llm, tools=tools[condition], max_tool_rounds=MAX_TOOL_ROUNDS),
                task_prompt=task_prompt,
                experiment_name=f"{condition}-f{FID}-{DIM}D-{repetition}",
                elitism=True,
                HPO=False,
                budget=args.budget,
                max_workers=args.max_workers,
            )
            try:
                best = es.run()
            except Exception as e:
                print(condition, repetition, f"failed: {e!r}")
                continue
            print(condition, repetition, best.fitness)
