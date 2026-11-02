import itertools
import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score

from src.hoi_anesthesia import diagnostics as dg
from src.hoi_anesthesia.states import NONRESPONSIVE_STATES, RESPONSIVE_STATES
from src.hoi_anesthesia.utils import build_X_for_dataset


class SharedStateDefinitionTests(unittest.TestCase):
    def test_diagnostics_uses_shared_awake_only_responsive_states(self):
        self.assertEqual(RESPONSIVE_STATES["MA"], ("MA_awake",))
        self.assertEqual(RESPONSIVE_STATES["DBS"], ("DBS_awake",))
        self.assertEqual(
            NONRESPONSIVE_STATES["DBS"],
            ("ts_off", "ts_on_3V_control", "ts_on_5V_control"),
        )
        self.assertEqual(dg.state_class("DBS", "DBS_awake"), "C")
        self.assertEqual(dg.state_class("DBS", "ts_on_5V"), "descriptive")

    def test_dataset_builder_excludes_held_out_dbs_stimulation_states(self):
        all_covs = {
            "DBS": {
                "DBS_awake": np.full((2, 2, 2), 1.0),
                "ts_off": np.full((1, 2, 2), 2.0),
                "ts_on_3V_control": np.full((1, 2, 2), 3.0),
                "ts_on_5V_control": np.full((1, 2, 2), 4.0),
                "ts_on_5V": np.full((2, 2, 2), 5.0),
                "ts_on_3V": np.full((2, 2, 2), 6.0),
            }
        }

        X, n_responsive = build_X_for_dataset("DBS", all_covs)

        self.assertEqual(n_responsive, 2)
        self.assertEqual(X.shape[0], 5)
        np.testing.assert_array_equal(X[:, 0, 0].numpy(), [1, 1, 2, 3, 4])


class OmegaSummaryTests(unittest.TestCase):
    def test_average_precision_matches_sklearn_with_ties(self):
        y = np.array([1, 0, 1, 0, 0])
        scores = np.array([
            [1, 0, 2, 1],
            [1, 0, 0, 4],
            [0, 0, 2, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
        ], dtype=float)
        for labels in (y, 1 - y):
            expected = [average_precision_score(labels, column) for column in scores.T]
            np.testing.assert_allclose(dg.average_precision_columns(labels, scores), expected)

    def test_two_positive_classes_use_same_omega_score(self):
        omega = np.array([[4, 1], [3, 0], [1, 4], [0, 3], [1000, -1000]], dtype=float)
        y = np.array([1, 1, 0, 0, -1])
        scores = dg.omega_separation_scores(omega, y)
        np.testing.assert_allclose(scores["pr_auc_C"], [1, 5 / 12])
        np.testing.assert_allclose(scores["pr_auc_NR"], [5 / 12, 1])
        np.testing.assert_allclose(scores["delta"], [3, -3])

    def test_omega_matrix_batching_preserves_all_values(self):
        rng = np.random.default_rng(2026)
        covs = np.stack([
            np.cov(rng.normal(size=(60, 6)), rowvar=False) for _ in range(3)
        ])
        for order in (3, 4, 5):
            with self.subTest(order=order):
                nplets = np.array(list(itertools.combinations(range(6), order)))
                unbatched = dg.omega_matrix(covs, nplets, batch_size=len(nplets))
                batched = dg.omega_matrix(covs, nplets, batch_size=3)
                np.testing.assert_allclose(batched, unbatched, rtol=1e-10, atol=1e-12)

    def test_class_summary_uses_scan_medians_and_mean_effect(self):
        omega = np.array([
            [0.0, -4.0], [1.0, -2.0], [5.0, 0.0], [10.0, 2.0],
            [2.0, 1.0], [4.0, 3.0],
        ])
        y = np.array([1, 1, 1, 1, 0, 0])
        summary = dg.omega_class_summary(omega, y)
        np.testing.assert_array_equal(summary["C"], [3.0, -1.0])
        np.testing.assert_array_equal(summary["NR"], [3.0, 2.0])
        np.testing.assert_array_equal(summary["delta"], [1.0, -3.0])

    def test_nonfinite_omega_is_reported(self):
        with self.assertRaisesRegex(FloatingPointError, "Non-finite Omega"):
            dg.omega_class_summary(
                np.array([[np.nan], [0.0]]), np.array([1, 0]),
            )

    def test_exhaustive_summaries_match_direct_thoi_in_combination_order(self):
        rng = np.random.default_rng(2026)
        covs = np.stack([
            np.cov(rng.normal(size=(60, 6)), rowvar=False) for _ in range(6)
        ])
        y = np.array([1, 1, 1, 1, 0, 0])
        for order in (3, 4):
            with self.subTest(order=order):
                nplets = np.array(list(itertools.combinations(range(6), order)))
                omega = dg.omega_matrix(covs, nplets)
                expected = {
                    "C": np.median(omega[y == 1], axis=0),
                    "NR": np.median(omega[y == 0], axis=0),
                    "delta": omega[y == 1].mean(axis=0) - omega[y == 0].mean(axis=0),
                }
                actual = dg.exhaustive_omega_summary(covs, y, order, batch_size=7)
                for key in expected:
                    self.assertEqual(actual[key].shape, (math.comb(6, order),))
                    np.testing.assert_allclose(
                        actual[key], expected[key], rtol=1e-10, atol=1e-12,
                    )

    def test_every_scan_value_and_nplet_is_persisted_including_descriptive(self):
        rng = np.random.default_rng(42)
        covs = np.stack([
            np.cov(rng.normal(size=(60, 6)), rowvar=False) for _ in range(7)
        ])
        y = np.array([1, 1, 1, 0, 0, -1, -1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "omega.h5"
            for order in (3, 4):
                nplets = np.array(list(itertools.combinations(range(6), order)))
                expected = dg.omega_matrix(covs, nplets)
                with h5py.File(path, "w") as h5:
                    summary = dg.exhaustive_omega_summary(
                        covs, y, order, batch_size=7, output_group=h5.create_group("values"),
                    )
                with h5py.File(path, "r") as h5:
                    group = h5["values"]
                    self.assertTrue(group.attrs["complete"])
                    self.assertEqual(group["omega"].dtype, np.dtype("float64"))
                    self.assertEqual(group["omega"].shape, (7, math.comb(6, order)))
                    np.testing.assert_array_equal(group["nplets"][:], nplets)
                    np.testing.assert_allclose(group["omega"][:], expected, atol=1e-12)
                    np.testing.assert_allclose(
                        summary["C"], np.median(expected[y == 1], axis=0), atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        summary["NR"], np.median(expected[y == 0], axis=0), atol=1e-12,
                    )

    def test_streamed_dataset_and_pooled_scores_match_full_values(self):
        rng = np.random.default_rng(2026)
        covs = {
            ds: np.stack([
                np.cov(rng.normal(size=(60, 5)), rowvar=False) for _ in range(count)
            ])
            for ds, count in (("MA", 5), ("DBS", 4))
        }
        labels = {"MA": np.array([1, 1, 1, 0, -1]), "DBS": np.array([1, 0, 0, -1])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pooled.h5"
            for order in (3, 4):
                nplets = np.array(list(itertools.combinations(range(5), order)))
                raw = {ds: dg.omega_matrix(X, nplets) for ds, X in covs.items()}
                raw["combined"] = np.concatenate([raw["MA"], raw["DBS"]])
                ys = {**labels, "combined": np.concatenate(list(labels.values()))}
                with h5py.File(path, "w") as h5:
                    for variant in ("V0", "V1"):
                        dg.save_exhaustive_comparison(
                            covs, labels, order, h5.create_group(variant), batch_size=3,
                        )
                with h5py.File(path, "r") as h5:
                    for variant in ("V0", "V1"):
                        for ds in ("MA", "DBS", "combined"):
                            group = h5[f"{variant}/{ds}/{order}"]
                            np.testing.assert_array_equal(group["nplets"][:], nplets)
                            if ds != "combined":
                                np.testing.assert_allclose(group["omega"][:], raw[ds], atol=1e-12)
                            else:
                                self.assertNotIn("omega", group)
                            y, omega = ys[ds], raw[ds]
                            included = y >= 0
                            np.testing.assert_allclose(
                                group["delta"][:],
                                omega[y == 1].mean(axis=0) - omega[y == 0].mean(axis=0),
                                atol=1e-12,
                            )
                            for positive, binary in (("C", y[included]), ("NR", 1 - y[included])):
                                expected = [
                                    average_precision_score(binary, column)
                                    for column in omega[included].T
                                ]
                                np.testing.assert_allclose(group[f"pr_auc_{positive}"][:], expected)
                                self.assertEqual(
                                    group.attrs[f"chance_pr_auc_{positive}"], binary.mean(),
                                )
                    average_delta = (
                        h5[f"V0/MA/{order}/delta"][:] + h5[f"V0/DBS/{order}/delta"][:]
                    ) / 2
                    self.assertFalse(np.allclose(
                        h5[f"V0/combined/{order}/delta"][:], average_delta,
                    ))

    def test_density_uses_every_selected_scan_value_including_tails(self):
        values = np.array([
            [-100, 0, 0.1, 1, 100], [1000, 2, 3, 4, 5], [-30, -2, 0, 2, 30],
        ], dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "density.h5", "w") as h5:
                omega = h5.create_dataset("omega", data=values)
                density = dg.saved_omega_density(omega, np.array([0, 2]), batch_size=2)
                self.assertEqual(density["count"], 10)
                self.assertEqual(density["histogram_count"], 10)
                self.assertLess(density["coords"][0], -100)
                self.assertGreater(density["coords"][-1], 100)
                spacing = density["coords"][1] - density["coords"][0]
                self.assertAlmostEqual(density["density"].sum() * spacing, 1, places=4)
                zeros = h5.create_dataset("zeros", data=np.zeros((2, 7)))
                constant = dg.saved_omega_density(zeros, np.array([0, 1]), batch_size=3)
                self.assertEqual(constant["histogram_count"], 14)
                self.assertTrue(np.isfinite(constant["density"]).all())


if __name__ == "__main__":
    unittest.main()
