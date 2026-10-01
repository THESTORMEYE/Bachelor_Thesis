"""
# ELA toolkit for BBOB problems.
"""

import math

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

from .tools import Tool, ToolContext

SAMPLE_SIZE_PER_DIM = 250
SEED = 0
LOWER, UPPER = -5.0, 5.0


class BBOBELATool(Tool):
    """
    Mean of ELA features across BBOB instances.

    Args:
        fid: BBOB function id.
        iids: Instance ids to sample.
        dim: Problem dimension.
    """

    name = "landscape_analysis"

    description = """..."""
    
    parameters = {"type": "object", "properties": {}}

    def __init__(self, fid: int, iids: list[int], dim: int):
        if ioh is None:
            raise ImportError(
                "BBOBELATool needs ioh and pflacco. Install via `uv sync --dev --group ela`."
            )
        self.fid = fid
        self.iids = list(iids)
        self.dim = dim

    def run(self, context: ToolContext) -> dict:
        X = create_initial_sample(
            self.dim,
            n=SAMPLE_SIZE_PER_DIM * self.dim,
            lower_bound=LOWER,
            upper_bound=UPPER,
            seed=SEED,
        )
        X_scaled = (X - X.min()) / (X.max() - X.min())
        per_instance = []
        for iid in self.iids:
            problem = ioh.get_problem(self.fid, iid, self.dim)
            y = np.asarray(problem(X.values), dtype=float)
            y_scaled = (y - y.min()) / (y.max() - y.min())
            features = {}
            for feature_set in FEATURE_SETS:
                features.update(feature_set(X_scaled, y_scaled))
            per_instance.append(features)

        aggregated = {}
        for key in per_instance[0]:
            # *_x features depend on the fixed sample only.
            if key.endswith("costs_runtime") or key.endswith("_x"):
                continue
            mean = float(np.mean([f[key] for f in per_instance]))
            aggregated[key] = mean if math.isfinite(mean) else None
        return aggregated
