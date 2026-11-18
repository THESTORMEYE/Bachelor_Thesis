"""
Scores an algorithm on BBOB (instance, seed) pairs. Runs as a child process
(killable), reads JSON from stdin, pickles to `job["result"]`.
"""

import json
import os
import pickle
import random
import sys
import traceback

import ioh
import numpy as np

from misc import OverBudgetException, aoc_logger, correct_aoc


def run_bbob(job, used):
    """AOCC for each (instance, seed) run, adding that run's evaluations to `used`."""
    budget = job["budget"]
    namespace = {}
    exec(job["code"], namespace)
    root, folder_name = os.path.split(job["folder"])
    analyzer = ioh.logger.Analyzer(
        root=root, folder_name=folder_name, algorithm_name=job["name"]
    )
    l2 = aoc_logger(budget=budget, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS])
    loggers = ioh.logger.Combine([l2, analyzer])
    aucs = []

    try:
        for iid, seed in job["pairs"]:
            problem = ioh.get_problem(job["fid"], iid, job["dim"])
            problem.attach_logger(loggers)
            random.seed(seed)
            np.random.seed(seed)
            try:
                namespace[job["name"]](budget=budget, dim=job["dim"])(problem)
            except OverBudgetException:
                pass
            finally:
                used.append(min(problem.state.evaluations, budget))
            aucs.append(correct_aoc(problem, l2, budget))
            l2.reset(problem)
            problem.reset()
    finally:
        analyzer.close()
    return aucs


if __name__ == "__main__":
    job = json.load(sys.stdin)
    used = []
    try:
        result = {"aucs": run_bbob(job, used)}
    except BaseException as e:
        frames = [
            f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<string>"
        ]
        try:
            error = pickle.dumps(e)
        except Exception:
            error = None
        result = {
            "error": error,
            "type": type(e).__name__,
            "message": str(e),
            "line": frames[-1].lineno if frames else None,
        }
    with open(job["result"], "wb") as f:
        pickle.dump({**result, "used": used}, f)
