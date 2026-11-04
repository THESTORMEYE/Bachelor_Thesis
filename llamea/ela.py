"""
ELA toolkit for BBOB problems.
"""

import os

try:
    import ioh
    import pandas as pd
    from pflacco.classical_ela_features import (
        calculate_dispersion,
        calculate_ela_distribution,
        calculate_ela_meta,
        calculate_information_content,
        calculate_nbc,
        calculate_pca,
    )
    from pflacco.deep_ela import load_large_50d_v1
    from pflacco.sampling import create_initial_sample

    FEATURE_SETS = (
        calculate_ela_meta,
        calculate_ela_distribution,
        calculate_nbc,
        calculate_dispersion,
        calculate_pca,
        calculate_information_content,
    )
except ModuleNotFoundError:
    ioh = None

import numpy as np
import xgboost as xgb

from .tools import Tool, ToolContext

SAMPLE_SIZE_PER_DIM = 250
SEEDS = range(5)
LOWER, UPPER = -5.0, 5.0
PROPERTIES = ("Basins", "GlobalLocal", "Homogeneous", "Multimodality", "Separable")
LEARNED = PROPERTIES[:-1]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ela_models")
SEPARABILITY_SAMPLES = 1024


def separability(problem, dim, samples=SEPARABILITY_SAMPLES, tol=1e-6, seed=0):
    h = 1e-3 * (UPPER - LOWER)
    eps = 1e-12
    rng = np.random.default_rng(seed)
    X = rng.uniform(LOWER + 2 * h, UPPER - 2 * h, size=(samples, dim))
    e = np.eye(dim) * h

    def f(A):
        return np.asarray(problem(A), dtype=float)

    fx = f(X)
    up = [f(X + e[i]) for i in range(dim)]
    hii = np.stack([(up[i] - 2 * fx + f(X - e[i])) / h**2 for i in range(dim)], 1)
    scale = np.median(np.abs(hii), axis=0) + eps

    hess = []
    for i in range(dim):
        for j in range(i + 1, dim):
            hij = (
                f(X + e[i] + e[j])
                - f(X + e[i] - e[j])
                - f(X - e[i] + e[j])
                + f(X - e[i] - e[j])
            ) / (4 * h**2)
            hess.append(np.abs(hij) / np.sqrt(scale[i] * scale[j]) > tol)

    sup = []
    for i in range(dim):
        d_i = up[i] - fx
        for j in range(dim):
            if j != i:
                d_ij = f(X + e[j] + e[i]) - up[j]
                sup.append(
                    np.abs(d_ij - d_i) / (np.abs(d_i) + np.abs(d_ij) + eps) > tol
                )

    frac_hess = np.mean(hess) if hess else 0.0
    frac_sup = np.mean(sup) if sup else 0.0
    return float(1 - (0.5 * frac_hess + 0.5 * frac_sup))


class BBOBELATool(Tool):
    """
    ELA predictions are averaged over BBOB instances and precomputed at 
    construction time. Separability is detected via a finite-difference test.

    Args:
        fid: BBOB function id.
        iids: Instance ids to sample.
        dim: Problem dimension.
    """

    name = "landscape_analysis"

    description = (
        "Reveals the target function's structure via [0,1] scores "
        "(0 none, 1 high) for Basins (uniform sizes), "
        "GlobalLocal (sharp global-local contrast), Homogeneous (no phase transitions), "
        "Multimodality (many local optima), and Separable (dimension-wise decomposable)."
        )


    parameters = {"type": "object", "properties": {}}

    def __init__(self, fid: int, iids: list[int], dim: int):
        if ioh is None:
            raise ImportError(
                "BBOBELATool needs ioh and pflacco with Deep-ELA. "
                "Install via `uv sync --dev --group ela`."
            )
        deep_ela = load_large_50d_v1()
        models = {}
        for prop in LEARNED:
            models[prop] = xgb.Booster()
            models[prop].load_model(
                os.path.join(
                    MODEL_DIR,
                    f"model_combined_Groups_{prop}_feature_selection_0.5.json",
                )
            )
        scores = []
        for iid in iids:
            problem = ioh.get_problem(fid, iid, dim)
            scores.append(
                {
                    **self._predict(problem, iid, dim, deep_ela, models),
                    "Separable": separability(problem, dim),
                }
            )
        self.properties = {
            prop: float(np.mean([s[prop] for s in scores])) for prop in PROPERTIES
        }

    def run(self, context: ToolContext) -> dict:
        return dict(self.properties)

    @staticmethod
    def _predict(problem, iid, dim, deep_ela, models) -> dict:
        """Property probabilities for one problem instance."""
        deep, classical = [], []
        for seed in SEEDS:
            X = create_initial_sample(
                dim,
                n=SAMPLE_SIZE_PER_DIM * dim,
                lower_bound=LOWER,
                upper_bound=UPPER,
                seed=seed,
            )
            y = pd.Series(problem(X.values), index=X.index, dtype=float)
            deep.append(deep_ela(X, y, include_costs=False))

            y[y == 0] = 0.1**100
            X_scaled = (X - X.min()) / (X.max() - X.min())
            y_scaled = (
                0 * y if y.max() == y.min() else (y - y.min()) / (y.max() - y.min())
            )
            features = {}
            for feature_set in FEATURE_SETS:
                features.update(feature_set(X_scaled, y_scaled))
            classical.append(features)

        classical = (
            pd.DataFrame(classical).dropna().filter(regex="^(?!.*costs_runtime)")
        )
        row = {
            **pd.DataFrame(deep).mean(),
            "dim": dim,
            **classical.mean(),
            "instance": iid,
        }
        return {
            prop: float(
                model.predict(xgb.DMatrix(pd.DataFrame([row])[model.feature_names]))[0]
            )
            for prop, model in models.items()
        }


LEVELS = {"none": 0.0, "low": 1 / 3, "medium": 2 / 3, "high": 1.0}

BBOB_LABELS = {
    1: ("none", "none", "high", "none", "high"),
    2: ("none", "none", "high", "none", "high"),
    3: ("low", "low", "high", "high", "high"),
    4: ("medium", "low", "high", "high", "high"),
    5: ("none", "none", "high", "none", "high"),
    6: ("none", "none", "medium", "none", "none"),
    7: ("none", "none", "high", "none", "none"),
    8: ("low", "low", "medium", "low", "none"),
    9: ("low", "low", "medium", "low", "none"),
    10: ("none", "none", "high", "none", "none"),
    11: ("none", "none", "high", "none", "none"),
    12: ("none", "none", "high", "none", "none"),
    13: ("none", "none", "medium", "none", "none"),
    14: ("none", "none", "medium", "none", "none"),
    15: ("low", "low", "high", "high", "none"),
    16: ("medium", "low", "high", "high", "none"),
    17: ("medium", "high", "medium", "high", "none"),
    18: ("medium", "high", "medium", "high", "none"),
    19: ("low", "low", "high", "high", "none"),
    20: ("low", "high", "high", "medium", "none"),
    21: ("medium", "low", "high", "medium", "none"),
    22: ("medium", "high", "high", "low", "none"),
    23: ("low", "low", "high", "high", "none"),
    24: ("low", "low", "high", "high", "none"),
}


class BBOBOracleTool(BBOBELATool):
    """
    Expert-rated BBOB problem properties, mapped to evenly spaced degrees.
    Exposed like :class:`BBOBELATool`.
    Source: Tanabe 2021, Tbl. 2 (revises Mersmann 2010),
    https://arxiv.org/abs/2104.10301.

    Args:
        fid: BBOB function ID.
    """

    description = (
        "Reveals the target function's structure via [0,1] scores "
        "(0 none, 1 high) for Basins (uniform sizes), "
        "GlobalLocal (sharp global-local contrast), Homogeneous (no phase transitions), "
        "Multimodality (many local optima), and Separable (dimension-wise decomposable)."
        )


    def __init__(self, fid: int):
        self.properties = {
            prop: LEVELS[label] for prop, label in zip(PROPERTIES, BBOB_LABELS[fid])
        }
