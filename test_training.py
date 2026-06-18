import unittest

import numpy as np
from models import model_from_params
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from train import grouped_holdout_indices


class TrainingPipelineTest(unittest.TestCase):
    def test_grouped_holdout_never_leaks_users(self) -> None:
        y = []
        groups = []
        for label in range(3):
            for group in range(4):
                for _ in range(2):
                    y.append(label)
                    groups.append(f"user-{label}-{group}")

        train_indices, holdout_indices = grouped_holdout_indices(
            np.array(y),
            np.array(groups),
            splits=4,
            random_seed=42,
        )

        self.assertTrue(
            set(np.array(groups)[train_indices]).isdisjoint(
                set(np.array(groups)[holdout_indices])
            )
        )

    def test_model_families_rebuild_from_params(self) -> None:
        random_forest = model_from_params(
            {
                "family": "random_forest",
                "random_forest_max_depth": None,
                "random_forest_max_features": "sqrt",
                "random_forest_min_samples_leaf": 1,
                "random_forest_n_estimators": 100,
            },
            random_seed=42,
            n_jobs=1,
        )
        mlp = model_from_params(
            {
                "family": "mlp",
                "mlp_alpha": 0.001,
                "mlp_batch_size": 64,
                "mlp_hidden_layers": "64",
                "mlp_learning_rate_init": 0.001,
            },
            random_seed=42,
            n_jobs=1,
        )

        self.assertIsInstance(random_forest, RandomForestClassifier)
        self.assertIsInstance(mlp, Pipeline)


if __name__ == "__main__":
    unittest.main()
