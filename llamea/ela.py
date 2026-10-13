"""
ELA toolkit for BBOB problems.
"""

import os

try:
    import ioh
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
import pandas as pd
import xgboost as xgb

from .tools import Tool, ToolContext

SAMPLE_SIZE_PER_DIM = 250
SEEDS = range(5)
LOWER, UPPER = -5.0, 5.0
PROPERTIES = ("Basins", "GlobalLocal", "Homogeneous", "Multimodality", "Separable")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ela_models")


class BBOBELATool(Tool):
    """
    Mean of ELA features across BBOB instances.

    Args:
        fid: BBOB function id.
        iids: Instance ids to sample.
        dim: Problem dimension.
    """

    name = "landscape_analysis"

    description = (
        "Estimated properties of the target problem's landscape, each expressed "
        "as a probability in [0, 1]: Basins (uniform basin sizes), GlobalLocal "
        "(pronounced contrast between global and local optima), Homogeneous "
        "(absence of phase transitions), Multimodality (numerous local optima), "
        "Separable (decomposable along each dimension)."
    )

    parameters = {"type": "object", "properties": {}}

    def __init__(self, fid: int, iids: list[int], dim: int):
        if ioh is None:
            raise ImportError(
                "BBOBELATool needs ioh and pflacco with Deep-ELA. "
                "Install via `uv sync --dev --group ela`."
            )
        self.fid = fid
        self.iids = list(iids)
        self.dim = dim
        self.deep_ela = load_large_50d_v1()
        self.models = {}
        for prop in PROPERTIES:
            self.models[prop] = xgb.Booster()
            self.models[prop].load_model(
                os.path.join(MODEL_DIR, f"model_Groups_{prop}_llamea_fixed_labels.json")
            )

    def run(self, context: ToolContext) -> dict:
        scores = [
            self._predict(ioh.get_problem(self.fid, iid, self.dim)) for iid in self.iids
        ]
        return {prop: float(np.mean([s[prop] for s in scores])) for prop in PROPERTIES}

    def _predict(self, problem) -> dict:
        """Property probabilities for one problem instance."""
        deep, classical = [], []
        for seed in SEEDS:
            X = create_initial_sample(
                self.dim,
                n=SAMPLE_SIZE_PER_DIM * self.dim,
                lower_bound=LOWER,
                upper_bound=UPPER,
                seed=seed,
            )
            y = pd.Series(problem(X.values), index=X.index, dtype=float)
            deep.append(self.deep_ela(X, y, include_costs=False))

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
        row = {**pd.DataFrame(deep).mean(), "dim": self.dim, **classical.mean()}
        return {
            prop: float(
                model.predict(xgb.DMatrix(pd.DataFrame([row])[model.feature_names]))[0]
            )
            for prop, model in self.models.items()
        }
