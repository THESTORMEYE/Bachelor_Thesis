import os
import random
import textwrap

import ioh
import numpy as np

from llamea import Agent, BBOBOracleTool, Gemini_LLM, IOHAnalysisTool, LLaMEA
from misc import OverBudgetException, aoc_logger, correct_aoc

MODEL = "gemini-3.5-flash-lite"
FID = 3
DIM = 5
IIDS = [1, 2, 3]
SEEDS = [0, 1, 2]
BUDGET = 200 * DIM
REPETITIONS = 5


def evaluate(solution, explogger):
    namespace = {}
    exec(solution.code, namespace)
    analyzer = ioh.logger.Analyzer(
        root=os.path.join(explogger.dirname, "ioh"),
        folder_name=solution.id,
        algorithm_name=solution.name,
    )
    l2 = aoc_logger(budget=BUDGET, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS])
    loggers = ioh.logger.Combine([analyzer, l2])
    aucs = []

    try:
        for iid in IIDS:
            problem = ioh.get_problem(FID, iid, DIM)
            problem.attach_logger(loggers)
            for seed in SEEDS:
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
    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    feedback = f"The algorithm {solution.name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)
    return solution


task_prompt = textwrap.dedent(f"""
    Write a Python heuristic optimizer for a single black-box function tested on {len(IIDS)}
    instances. Implement `__init__(self, budget, dim)` and `__call__(self, func)`; call `func`
    at most `self.budget` times. Domain: [-5.0, 5.0]; dimension: {DIM}. Provide a strong, novel
    heuristic and a one-line description of its core idea.
    """)


if __name__ == "__main__":
    llm = Gemini_LLM(os.getenv("GEMINI_API_KEY"), MODEL)
    tools = {
        "no_tools": None,
        "ioh": IOHAnalysisTool(budget=BUDGET),
        "oracle": BBOBOracleTool(fid=FID),
    }

    for repetition in range(REPETITIONS):
        for condition, tool in tools.items():
            es = LLaMEA(
                evaluate,
                n_parents=2,
                n_offspring=5,
                llm=Agent(llm, tools=[tool]) if tool else llm,
                task_prompt=task_prompt,
                experiment_name=f"{condition}-f{FID}-{DIM}D-{repetition}",
                elitism=True,
                HPO=False,
                budget=26,
            )
            best = es.run()
            print(condition, repetition, best.fitness)
