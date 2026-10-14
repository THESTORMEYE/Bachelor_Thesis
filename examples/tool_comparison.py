import os
import random
import textwrap

import ioh
import numpy as np

from llamea import Agent, BBOBELATool, Gemini_LLM, IOHAnalysisTool, LLaMEA

MODEL = "gemini-3.5-flash-lite"
FID = 21
DIM = 5
IIDS = [1, 2, 3]
SEEDS = [0, 1, 2]
BUDGET = 2000 * DIM
REPETITIONS = 3


def evaluate(solution, explogger):
    namespace = {}
    exec(solution.code, namespace)
    analyzer = ioh.logger.Analyzer(
        root=os.path.join(explogger.dirname, "ioh"),
        folder_name=solution.id,
        algorithm_name=solution.name,
    )
    precisions = []

    try:
        for iid in IIDS:
            problem = ioh.get_problem(FID, iid, DIM)
            problem.attach_logger(analyzer)
            for seed in SEEDS:
                random.seed(seed)
                np.random.seed(seed)
                namespace[solution.name](budget=BUDGET, dim=DIM)(problem)
                if problem.state.evaluations > BUDGET:
                    raise ValueError(
                        f"{problem.state.evaluations} evaluations used, "
                        f"with limit of {BUDGET}."
                    )
                precisions.append(problem.state.current_best.y - problem.optimum.y)
                problem.reset()
    finally:
        analyzer.close()
    scores = -np.log10(np.array(precisions) + 1e-8)
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    score_median = float(np.median(scores))
    feedback = f"{solution.name}: n={len(scores)}; mean={score_mean:.2f}, sd={score_std:.2f}, med={score_median:.2f}; score=-log10(f-f_opt), 8=best"
    solution.set_scores(score_mean, feedback)
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
        "ela": BBOBELATool(fid=FID, iids=IIDS, dim=DIM),
        "ioh": IOHAnalysisTool(budget=BUDGET),
    }

    for repetition in range(REPETITIONS):
        for condition, tool in tools.items():
            es = LLaMEA(
                evaluate,
                n_parents=4,
                n_offspring=4,
                llm=Agent(llm, tools=[tool]) if tool else llm,
                task_prompt=task_prompt,
                experiment_name=f"{condition}-f{FID}-{DIM}D-{repetition}",
                budget=20,
            )
            best = es.run()
            print(condition, repetition, best.fitness)