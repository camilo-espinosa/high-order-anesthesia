import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from src.hoi_anesthesia import validation as v
from src.hoi_anesthesia.thoi_utils import simulated_annealing_parallel
from src.hoi_anesthesia.utils import pairwise_fisher_z


class WeightedObjectiveTests(unittest.TestCase):
    def test_equal_animal_contributions_within_each_class(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0])
        animals = np.array(["A", "A", "A", "B", "A", "C", "C"])
        weights = v.animal_balanced_weights(labels, animals)
        np.testing.assert_allclose(weights, [1 / 6, 1 / 6, 1 / 6, 1 / 2, 1 / 2, 1 / 4, 1 / 4])
        for label in (0, 1):
            self.assertAlmostEqual(weights[labels == label].sum(), 1)

    def test_rank_auc_matches_sklearn_with_ties_and_both_polarities(self):
        y = np.array([1, 1, 0, 1, 0, 0])
        weights = v.animal_balanced_weights(y, ["A", "A", "B", "C", "B", "D"])
        values = np.array([
            [2, 2, 2, 4, 1, 4], [1, 1, 1, 1, 1, 1], [6, 5, 4, 3, 2, 1],
        ], dtype=float)
        for sign in (-1, 1):
            actual = v.weighted_roc_auc(
                torch.tensor(sign * values), torch.tensor(y), torch.tensor(weights),
            ).numpy()
            expected = [roc_auc_score(y, sign * row, sample_weight=weights) for row in values]
            np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertEqual(actual[1], 0.5)

    def test_invalid_auc_inputs_fail_explicitly(self):
        for labels in ([1, 1], [1, 2]):
            with self.assertRaises(ValueError):
                v.weighted_roc_auc(
                    torch.zeros((1, 2)), torch.tensor(labels), torch.ones(2),
                )
        with self.assertRaises(ValueError):
            v.animal_balanced_weights(np.array([1, 1]), ["A", "B"])
        with self.assertRaises(ValueError):
            v.weighted_roc_auc(
                torch.tensor([[float("nan"), 1]]), torch.tensor([0, 1]), torch.ones(2),
            )

    def test_one_class_test_folds_are_valid_but_one_class_training_is_not(self):
        frame = pd.DataFrame({"animal": ["A", "B", "C", "C"], "responsive": [1, 0, 1, 0]})
        folds = list(v.loao_folds(frame))
        self.assertEqual(len(folds), 3)
        self.assertEqual(len(frame.iloc[folds[0][2]]["responsive"].unique()), 1)
        for animal, train, test in folds:
            self.assertNotIn(animal, frame.iloc[train]["animal"].tolist())
            self.assertEqual(set(frame.iloc[test]["animal"]), {animal})
        with self.assertRaisesRegex(ValueError, "one-class training"):
            list(v.loao_folds(pd.DataFrame({"animal": ["A", "B"], "responsive": [1, 0]})))


class MetadataTests(unittest.TestCase):
    def setUp(self):
        states = ("MA_awake", "deep_propofol", "ketamine", "ts_selv4")
        self.covs = {"MA": {state: np.eye(4)[None] for state in states}}
        self.metadata = pd.DataFrame([
            dict(dataset="MA", state=state, scan_index=0, animal=str(index), session="1", run="1")
            for index, state in enumerate(states)
        ])

    def test_metadata_is_aligned_by_explicit_scan_index(self):
        X, frame = v.load_classification_data(self.covs, self.metadata.iloc[::-1], "MA")
        self.assertEqual(X.shape, (4, 4, 4))
        self.assertEqual(frame["state"].tolist(), list(self.covs["MA"]))
        self.assertEqual(frame["responsive"].tolist(), [1, 0, 0, 0])

    def test_missing_duplicate_and_bad_indices_fail(self):
        for frame in (
            self.metadata.iloc[1:],
            pd.concat([self.metadata, self.metadata.iloc[:1]]),
            self.metadata.assign(scan_index=0.5),
            self.metadata.assign(animal=""),
        ):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                v.load_classification_data(self.covs, frame, "MA")

    def test_nonpositive_covariances_fail_before_search(self):
        self.covs["MA"]["MA_awake"][0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "positive definite"):
            v.load_classification_data(self.covs, self.metadata, "MA")


class SearchAndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([1, 1, 0, 0])
        self.animals = ["A", "B", "C", "D"]
        self.covs = np.repeat(np.eye(6)[None], 4, axis=0)
        self.settings = v.SearchSettings(restarts=2, iterations=4, early_stop=3, batch_size=2)

    def test_pairwise_vectors_match_direct_calculation_and_legacy_mean(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(3, 6, 12))
        covs = samples @ samples.swapaxes(-1, -2) + np.eye(6)[None]
        nplets = torch.tensor([[0, 2, 4], [1, 3, 5]])
        actual = pairwise_fisher_z(torch.tensor(covs), nplets).numpy()
        self.assertEqual(actual.shape, (2, 3, 3))
        for index, nplet in enumerate(nplets.numpy()):
            sub = covs[:, nplet[:, None], nplet[None, :]]
            std = np.sqrt(np.diagonal(sub, axis1=-2, axis2=-1))
            corr = sub / (std[:, :, None] * std[:, None, :])
            edges = np.triu_indices(3, 1)
            expected = np.arctanh(np.clip(corr[:, edges[0], edges[1]], -0.999999, 0.999999))
            np.testing.assert_allclose(actual[index], expected)
        objective = v.TrainingObjective(covs, self.labels[:3], self.animals[:3], "fc", "cpu", 2)
        np.testing.assert_allclose(objective.values(nplets).numpy(), actual.mean(axis=-1))

    def test_fc_never_computes_hoi_and_exhaustive_ties_are_lexicographic(self):
        objective = v.TrainingObjective(self.covs, self.labels, self.animals, "fc", "cpu", 2)
        with patch.object(v, "nplets_measures", side_effect=AssertionError("FC must not call THOI")):
            result = v.exhaustive_search(objective, 3, self.settings.batch_size, self.settings.seed)
        for selection in result.values():
            self.assertEqual(selection.nplet, (0, 1, 2))
            self.assertEqual(selection.training_auc, 0.5)
        batches = list(v.candidate_batches(6, 3, 2))
        self.assertEqual([row for batch in batches for row in batch], list(itertools.combinations(range(6), 3)))

    def test_shared_annealer_callback_skips_hoi_and_preserves_input(self):
        X = torch.tensor(self.covs)
        original = X.clone()
        seen = []

        def score(candidates):
            seen.append(candidates.clone())
            return candidates.sum(dim=1).to(torch.float64) / 15

        with patch("src.hoi_anesthesia.thoi_utils._evaluate_nplets", side_effect=AssertionError("HOI path")):
            solutions, scores = simulated_annealing_parallel(
                X, order=3, initial_solution=torch.tensor([[0, 1, 2], [3, 4, 5]]),
                repeat=2, batch_size=1, max_iterations=4, early_stop=3,
                initial_temp=0.05, largest=True, candidate_score=score, verbose=40,
            )
        self.assertEqual(solutions.shape, (2, 1, 3))
        self.assertEqual(scores.shape, (2, 1))
        self.assertTrue(all(batch.shape == (1, 3) for batch in seen))
        self.assertTrue(torch.equal(X, original))
        self.assertTrue(torch.all(scores[:, 0] >= torch.tensor([3 / 15, 12 / 15], dtype=torch.float64)))
        for solution in solutions[:, 0].tolist():
            self.assertEqual(len(set(solution)), 3)

    def test_exhaustive_fc_matches_brute_force_weighted_auc_in_both_directions(self):
        rng = np.random.default_rng(2026)
        samples = rng.normal(size=(4, 6, 12))
        covs = samples @ samples.swapaxes(-1, -2) + np.eye(6)[None]
        objective = v.TrainingObjective(covs, self.labels, self.animals, "fc", "cpu", 2)
        candidates = list(itertools.combinations(range(6), 3))
        features = pairwise_fisher_z(torch.tensor(covs), torch.tensor(candidates)).mean(dim=-1).numpy()
        weights = v.animal_balanced_weights(self.labels, self.animals)
        actual = v.exhaustive_search(objective, 3, self.settings.batch_size, self.settings.seed)
        for polarity in (1, -1):
            scores = [roc_auc_score(self.labels, polarity * row, sample_weight=weights) for row in features]
            maximum = max(scores)
            expected = min(nplet for nplet, score in zip(candidates, scores) if score == maximum)
            self.assertEqual(actual[polarity].nplet, expected)
            self.assertAlmostEqual(actual[polarity].training_auc, maximum)

    def test_legacy_pair_based_annealer_path_still_uses_measure_evaluator(self):
        pairs = torch.tensor(self.covs[:2]).unsqueeze(0)

        def normalize(covariances, *_):
            return covariances, 2, 6, None

        def evaluate(_covariances, _sample_counts, candidates, _metric, **_kwargs):
            return candidates.sum(dim=-1).to(torch.float64)

        with patch("src.hoi_anesthesia.thoi_utils._normalize_input_data", side_effect=normalize), patch(
            "src.hoi_anesthesia.thoi_utils._evaluate_nplets", side_effect=evaluate,
        ) as evaluator:
            solutions, scores = simulated_annealing_parallel(
                pairs, order=3, covmat_precomputed=True, repeat=2, max_iterations=2,
                early_stop=2, initial_temp=0.05, largest=True, verbose=40,
            )
        self.assertGreaterEqual(evaluator.call_count, 2)
        self.assertEqual(solutions.shape, (2, 1, 3))
        self.assertEqual(scores.shape, (2, 1))

    def test_sa_is_reproducible_without_consuming_callers_rng(self):
        objective = v.TrainingObjective(self.covs, self.labels, self.animals, "fc", "cpu", 2)
        before = torch.random.get_rng_state()
        first = v.simulated_annealing_search(objective, 3, self.settings, self.settings.seed)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = v.simulated_annealing_search(objective, 3, self.settings, self.settings.seed)
        self.assertEqual(first, second)

    def test_scalar_fit_uses_only_training_scaling(self):
        train = np.array([[0], [1], [4], [5]], dtype=float)
        test = np.array([[1000]], dtype=float)
        weights = np.array([1 / 2] * 4)
        probability, scaler, model = v.fit_probabilities(train, test, self.labels, weights)
        np.testing.assert_allclose(scaler.mean_, [2.5])
        self.assertEqual(model.C, 1)
        self.assertEqual(probability.shape, (1,))
        self.assertTrue(0 <= probability[0] <= 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame([
            dict(dataset="MA", state=state, scan_index=index, animal=animal,
                 session=f"s{index}", run="1", responsive=label)
            for index, (animal, label, state) in enumerate([
                ("A", 1, "MA_awake"), ("B", 0, "deep_propofol"),
                ("C", 1, "MA_awake"), ("C", 0, "ketamine"),
                ("D", 1, "MA_awake"), ("D", 0, "ts_selv4"),
            ])
        ])
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(6, 5, 10))
        self.covs = samples @ samples.swapaxes(-1, -2) + 2 * np.eye(5)[None]

    def test_end_to_end_shapes_training_isolation_metrics_and_persistence(self):
        selected_training = []
        original = v.exhaustive_search

        def record(objective, order, batch_size, seed):
            selected_training.append(objective.covariances.cpu().numpy().copy())
            return original(objective, order, batch_size, seed)

        settings = v.SearchSettings(batch_size=4)
        with patch.object(v, "exhaustive_search", side_effect=record):
            result = v.run_loao_exhaustive(self.covs, self.metadata, settings=settings)
        self.assertEqual(len(result.selections), 4 * 3 * 2 * 2)
        self.assertEqual(len(result.predictions), 6 * 3 * 3 * 2)
        self.assertEqual(len(result.metrics), 3 * 3 * 2)
        self.assertEqual(len(result.fits), 4 * 3 * 3 * 2)
        self.assertFalse((
            result.predictions["selection_measure"].eq("omega")
            & result.predictions["model"].eq("pairwise_vector")
        ).any())
        self.assertEqual(
            set(result.predictions.loc[
                result.predictions["selection_measure"] == "fc", "model",
            ]),
            {"scalar", "pairwise_vector"},
        )
        for fold_index, (_, train, _) in enumerate(v.loao_folds(self.metadata)):
            for order_index in range(3):
                for measure_index in range(2):
                    position = fold_index * 6 + order_index * 2 + measure_index
                    np.testing.assert_array_equal(selected_training[position], self.covs[train])
        for _, group in result.predictions.groupby(["selection_measure", "polarity", "model"]):
            self.assertEqual(sorted(group["scan_index"]), list(range(6)))
            self.assertTrue((group["animal"] == group["held_out_animal"]).all())
        np.testing.assert_allclose(result.metrics["weighted_prevalence"], 0.5)
        for row in result.metrics.itertuples():
            group = result.predictions.loc[
                (result.predictions["selection_measure"] == row.selection_measure)
                & (result.predictions["polarity"] == row.polarity)
                & (result.predictions["model"] == row.model)
            ]
            weights = v.animal_balanced_weights(group["responsive"].to_numpy(), group["animal"])
            self.assertAlmostEqual(row.roc_auc, roc_auc_score(group["responsive"], group["probability"], sample_weight=weights))
        legacy_predictions = pd.concat([
            result.predictions,
            result.predictions.loc[
                result.predictions["selection_measure"].eq("omega")
                & result.predictions["model"].eq("scalar")
            ].assign(model="pairwise_vector"),
        ], ignore_index=True)
        legacy_fits = pd.concat([
            result.fits,
            result.fits.loc[
                result.fits["selection_measure"].eq("omega")
                & result.fits["model"].eq("scalar")
            ].assign(model="pairwise_vector"),
        ], ignore_index=True)
        rebuilt = v._assemble_loao_result(
            self.covs, self.metadata, legacy_predictions.to_dict("records"),
            result.selections.to_dict("records"), legacy_fits.to_dict("records"),
        )
        self.assertFalse((
            rebuilt.predictions["selection_measure"].eq("omega")
            & rebuilt.predictions["model"].eq("pairwise_vector")
        ).any())
        self.assertTrue((
            rebuilt.predictions["selection_measure"].eq("fc")
            & rebuilt.predictions["model"].eq("pairwise_vector")
        ).any())
        with tempfile.TemporaryDirectory() as directory:
            v.save_loao_result(result, directory, "MA")
            saved = pd.read_csv(Path(directory) / "MA_selections.csv")
            self.assertTrue(all(len(json.loads(value)) == 3 for value in saved["nplet"]))

    def test_duplicate_oof_scans_fail_instead_of_inflating_metrics(self):
        predictions = pd.DataFrame([
            dict(dataset="MA", order=3, selection_measure="fc", polarity=1, model="scalar",
                 state="awake", scan_index=0, responsive=1, animal="A", probability=0.8),
        ] * 2)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            v.pooled_metrics(predictions)

    def test_normalized_omega_is_o_over_s_and_runs_alone(self):
        def fake_measures(covariances, nplets, **_kwargs):
            nplets = torch.as_tensor(nplets, device=covariances.device)
            index = nplets.sum(dim=-1, dtype=torch.float64)[:, None]
            trace = covariances.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[None, :]
            o = torch.sin(index + trace)
            s = 2 + torch.cos(index * trace)
            return torch.stack((s - o, s + o, o, s), dim=-1)

        nplets = torch.tensor([[0, 1, 2], [1, 3, 4]])
        with patch.object(v, "nplets_measures", side_effect=fake_measures):
            raw = v.measure_values(torch.tensor(self.covs), nplets, "omega", 2)
            norm = v.measure_values(torch.tensor(self.covs), nplets, "omega_norm", 2)
            expected = fake_measures(torch.tensor(self.covs), nplets)
            torch.testing.assert_close(raw, expected[..., 2])
            torch.testing.assert_close(norm, expected[..., 2] / expected[..., 3])
            result = v.run_loao_exhaustive(
                self.covs, self.metadata, settings=v.SearchSettings(batch_size=4),
                measures=("omega_norm",),
            )
        self.assertEqual(set(result.selections["selection_measure"]), {"omega_norm"})
        self.assertEqual(set(result.predictions["model"]), {"scalar"})
        self.assertEqual(len(result.predictions), 6 * 3 * 2)
        self.assertTrue(result.predictions["raw_omega"].isna().all())
        self.assertTrue(result.predictions["raw_value"].between(-1, 1).all())
        for measures in ((), ("omega", "omega"), ("pairwise",)):
            with self.assertRaises(ValueError):
                v.run_loao_exhaustive(self.covs, self.metadata, measures=measures)

    @staticmethod
    def _fake_measures(covariances, nplets, **_kwargs):
        nplets = torch.as_tensor(nplets, device=covariances.device)
        index = nplets.sum(dim=-1, dtype=torch.float64)[:, None]
        trace = covariances.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[None, :]
        tc = 2 + torch.sin(index + trace)
        dtc = 1.5 + torch.cos(index * trace)
        return torch.stack((tc, dtc, tc - dtc, tc + dtc), dim=-1)

    def test_feature_sets_match_their_definitions_and_never_cross_families(self):
        nplet = (0, 2, 3)
        covs = torch.tensor(self.covs)
        with patch.object(v, "nplets_measures", side_effect=self._fake_measures):
            raw = v.nplet_feature_sets(covs, nplet, "omega", 4)
            norm = v.nplet_feature_sets(covs, nplet, "omega_norm", 4)
        with patch.object(v, "nplets_measures", side_effect=AssertionError("FC must not call THOI")):
            fc = v.nplet_feature_sets(covs, nplet, "fc", 4)
        hoi = self._fake_measures(covs, [nplet])[0].numpy()
        pairs = pairwise_fisher_z(covs, torch.tensor([nplet]))[0].numpy()
        self.assertEqual(set(raw), {"omega", "hoi_vector", "omega_gradient"})
        self.assertEqual(set(norm), {"omega", "hoi_vector", "omega_gradient"})
        self.assertEqual(set(fc), {"fc_mean", "fc_vector"})
        np.testing.assert_allclose(raw["omega"][:, 0], hoi[:, 2])
        np.testing.assert_allclose(norm["omega"][:, 0], hoi[:, 2] / hoi[:, 3])
        np.testing.assert_allclose(raw["hoi_vector"], hoi[:, :2])
        np.testing.assert_allclose(fc["fc_mean"][:, 0], pairs.mean(axis=1))
        np.testing.assert_allclose(fc["fc_vector"], pairs)
        self.assertEqual(fc["fc_vector"].shape, (6, 3))
        with self.assertRaises(ValueError):
            v.nplet_feature_sets(covs, nplet, "pairwise", 4)

    def test_omega_gradient_is_full_minus_leave_one_region_out(self):
        covs = torch.tensor(self.covs)
        for nplet in ((0, 2, 3), (0, 2, 3, 5)):
            for measure in ("omega", "omega_norm"):
                with patch.object(v, "nplets_measures", side_effect=self._fake_measures):
                    gradient = v.nplet_feature_sets(covs, nplet, measure, 4)["omega_gradient"]
                full = self._fake_measures(covs, [nplet])[0].numpy()
                full = full[:, 2] if measure == "omega" else full[:, 2] / full[:, 3]
                expected = []
                for removed in range(len(nplet)):
                    reduced = tuple(region for j, region in enumerate(nplet) if j != removed)
                    if len(reduced) < 3:
                        expected.append(full)
                        continue
                    sub = self._fake_measures(covs, [reduced])[0].numpy()
                    sub = sub[:, 2] if measure == "omega" else sub[:, 2] / sub[:, 3]
                    expected.append(full - sub)
                self.assertEqual(gradient.shape, (6, len(nplet)))
                np.testing.assert_allclose(gradient, np.column_stack(expected))

    def test_feature_evaluation_is_out_of_fold_and_training_scaled(self):
        with patch.object(v, "nplets_measures", side_effect=self._fake_measures):
            search = v.run_loao_exhaustive(
                self.covs, self.metadata, settings=v.SearchSettings(batch_size=4),
                measures=("omega", "omega_norm", "fc"),
            )
            result = v.evaluate_feature_sets(self.covs, self.metadata, search.selections, batch_size=4)
        n_groups = 3 * 2
        self.assertEqual(
            set(map(tuple, result.predictions[["selection_measure", "feature_set"]].drop_duplicates().to_numpy())),
            {(measure, feature) for measure, features in v.FEATURE_SETS.items() for feature in features},
        )
        n_models = sum(len(features) for features in v.FEATURE_SETS.values())
        self.assertEqual(len(result.predictions), 6 * n_groups * n_models)
        self.assertEqual(len(result.pooled), n_groups * n_models)
        self.assertEqual(len(result.animal_summary), n_groups * n_models)
        self.assertTrue((result.predictions["animal"] == result.predictions["held_out_animal"]).all())
        selected = search.selections.set_index(
            ["selection_measure", "order", "polarity", "held_out_animal"],
        )["nplet"]
        for fit in result.fits.itertuples():
            key = (fit.selection_measure, fit.order, fit.polarity, fit.held_out_animal)
            self.assertEqual(tuple(fit.nplet), tuple(selected.loc[key]))
        expected_width = {
            "omega": lambda k: 1, "hoi_vector": lambda k: 2, "omega_gradient": lambda k: k,
            "fc_mean": lambda k: 1, "fc_vector": lambda k: k * (k - 1) // 2,
        }
        for fit in result.fits.itertuples():
            self.assertEqual(fit.n_features, expected_width[fit.feature_set](fit.order))
            self.assertEqual(len(fit.nplet), fit.order)
        folds = {animal: train for animal, train, _ in v.loao_folds(self.metadata)}
        for fit in (result.fits.iloc[0], result.fits.loc[result.fits["feature_set"] == "fc_vector"].iloc[0]):
            train = folds[fit["held_out_animal"]]
            with patch.object(v, "nplets_measures", side_effect=self._fake_measures):
                features = v.nplet_feature_sets(
                    torch.tensor(self.covs), fit["nplet"], fit["selection_measure"], 4,
                )[fit["feature_set"]]
            weights = v.animal_balanced_weights(
                self.metadata["responsive"].to_numpy()[train],
                self.metadata["animal"].to_numpy()[train],
            )
            np.testing.assert_allclose(fit["mean"], np.average(features[train], axis=0, weights=weights))
        with tempfile.TemporaryDirectory() as directory:
            v.save_feature_evaluation(result, directory, "MA_features")
            saved = pd.read_csv(Path(directory) / "MA_features_fits.csv")
            self.assertTrue(all(
                len(json.loads(nplet)) == order for nplet, order in zip(saved["nplet"], saved["order"])
            ))
        incomplete = search.selections.loc[search.selections["held_out_animal"] != "A"]
        with self.assertRaisesRegex(ValueError, "every fold"):
            v.evaluate_feature_sets(self.covs, self.metadata, incomplete, batch_size=4)

    def test_animal_metrics_match_animal_balanced_accuracy(self):
        predictions = pd.DataFrame([
            dict(state=state, scan_index=index, animal=animal, responsive=label, probability=p)
            for index, (state, animal, label, p) in enumerate([
                ("awake", "A", 1, 0.9), ("awake", "A", 1, 0.4), ("awake", "A", 1, 0.8),
                ("awake", "B", 1, 0.3),
                ("drug", "B", 0, 0.2), ("drug", "C", 0, 0.6), ("drug", "C", 0, 0.5),
            ])
        ]).assign(dataset="MA", order=3, selection_measure="omega", polarity=-1, feature_set="omega")
        animal, summary = v.animal_level_metrics(predictions)
        self.assertEqual(len(animal), 4)
        row = summary.iloc[0]
        np.testing.assert_allclose(row["accuracy_responsive"], (2 / 3 + 0) / 2)
        np.testing.assert_allclose(row["accuracy_nonresponsive"], (1 + 0.25) / 2)
        y = predictions["responsive"].to_numpy()
        p = predictions["probability"].to_numpy()
        correct = np.where(y == 1, p, 1 - p)
        correct = np.where(correct > 0.5, 1.0, np.where(correct < 0.5, 0.0, 0.5))
        weights = v.animal_balanced_weights(y, predictions["animal"])
        np.testing.assert_allclose(row["balanced_accuracy"], np.average(correct, weights=weights))
        self.assertEqual((row["units_correct"], row["units"]), (2, 4))
        with self.assertRaisesRegex(ValueError, "both classes"):
            v.animal_level_metrics(predictions.loc[predictions["responsive"] == 1])

    def test_interrupted_search_logs_traceback_and_resumes_completed_units(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoints = v.LOAOCheckpointStore(checkpoint_dir, {"fixture": "resume"})
            logger = v.configure_loao_logger(checkpoint_dir / "run.log")
            original = v.exhaustive_search
            calls = 0

            def fail_second_unit(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated search failure")
                return original(*args)

            with patch.object(v, "exhaustive_search", side_effect=fail_second_unit):
                with self.assertRaisesRegex(RuntimeError, "simulated search failure"):
                    v.run_loao_exhaustive(
                        self.covs, self.metadata, checkpoint=checkpoints, logger=logger,
                    )

            self.assertEqual(len(list(checkpoint_dir.glob("task-*.json"))), 1)
            log_text = (checkpoint_dir / "run.log").read_text(encoding="utf-8")
            self.assertIn("Traceback (most recent call last):", log_text)
            self.assertIn("simulated search failure", log_text)
            self.assertIn("measure=fc", log_text)

            searched_orders = []

            def record_search(*args):
                searched_orders.append(args[1])
                return original(*args)

            with patch.object(v, "exhaustive_search", side_effect=record_search):
                resumed = v.run_loao_exhaustive(
                    self.covs, self.metadata, checkpoint=checkpoints, logger=logger,
                )

            self.assertEqual(len(searched_orders), 23)
            self.assertEqual(len(resumed.selections), 4 * 3 * 2 * 2)

    def test_manifest_tracks_input_bytes_and_search_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            covariance_path = Path(directory) / "covariances.h5"
            metadata_path = Path(directory) / "metadata.csv"
            covariance_path.write_bytes(b"covariances-v1")
            metadata_path.write_bytes(b"metadata-v1")
            settings = v.SearchSettings(batch_size=4)
            original = v.build_loao_manifest(
                covariance_path, metadata_path, settings, ("MA", "DBS"), "cpu",
            )
            synergy = v.build_synergy_manifest(
                covariance_path, metadata_path, settings, ("MA", "DBS"), "cpu",
            )
            self.assertNotEqual(v.loao_run_id(original), v.loao_run_id(synergy))
            self.assertEqual(synergy.get("analysis"), "R1_D_synergy_loao")

            covariance_path.write_bytes(b"covariances-v2")
            changed_input = v.build_loao_manifest(
                covariance_path, metadata_path, settings, ("MA", "DBS"), "cpu",
            )
            metadata_path.write_bytes(b"metadata-v2")
            changed_metadata = v.build_loao_manifest(
                covariance_path, metadata_path, settings, ("MA", "DBS"), "cpu",
            )
            changed_settings = v.build_loao_manifest(
                covariance_path, metadata_path,
                v.SearchSettings(batch_size=8), ("MA", "DBS"), "cpu",
            )
            self.assertNotEqual(v.loao_run_id(original), v.loao_run_id(changed_input))
            self.assertNotEqual(v.loao_run_id(changed_input), v.loao_run_id(changed_metadata))
            self.assertNotEqual(v.loao_run_id(changed_metadata), v.loao_run_id(changed_settings))

    def test_checkpoint_directory_rejects_a_different_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            v.LOAOCheckpointStore(directory, {"input": "first"})
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                v.LOAOCheckpointStore(directory, {"input": "changed"})

    def test_checkpoint_round_trip_preserves_float_precision(self):
        precision = 0.12345678901234566
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = v.LOAOCheckpointStore(directory, {"fixture": "precision"})
            checkpoints.save(
                "exhaustive", "MA", "A", 3, "omega",
                [{"probability": precision, "raw_omega": np.nan}],
                [{"nplet": (1, 2, 3), "training_auc": precision}],
                [{"mean": [precision]}],
            )
            loaded = checkpoints.load("exhaustive", "MA", "A", 3, "omega")

        if loaded is None:
            self.fail("completed checkpoint was not loaded")
        self.assertEqual(loaded[0][0]["probability"], precision)
        self.assertIsNone(loaded[0][0]["raw_omega"])
        self.assertEqual(loaded[1][0]["nplet"], [1, 2, 3])


class SynergyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([1, 1, 0, 0])
        self.animals = ["A", "B", "C", "D"]
        self.covariances = np.repeat(np.eye(4)[None], 4, axis=0)

    def _objective(self, synergy_rule=True):
        return v.TrainingObjective(
            self.covariances, self.labels, self.animals, "omega", "cpu", 2,
            synergy_rule=synergy_rule,
        )

    @staticmethod
    def _candidate_values(candidates):
        values_by_nplet = {
            (0, 1, 2): [-1, -1, 1, 1],
            (0, 1, 3): [1, 1, -1, -1],
            (0, 2, 3): [-2, -2, 0, 0],
            (1, 2, 3): [-1.5, -1.5, 0.5, 0.5],
        }
        return torch.tensor(
            [values_by_nplet[tuple(row)] for row in candidates.cpu().tolist()],
            dtype=torch.float64,
            device=candidates.device,
        )

    def test_weighted_median_uses_first_cumulative_half_weight_and_validates(self):
        values = torch.tensor([[0, 2, 4, 8], [1, 1, 1, 10]], dtype=torch.float64)
        weights = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
        actual = v.weighted_median_by_candidate(values, weights)
        torch.testing.assert_close(actual, torch.tensor([2, 1], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            v.weighted_median_by_candidate(values, torch.tensor([1, 0, 1, 1.]))

    def test_exhaustive_synergy_masks_candidates_and_breaks_ties_by_awake_median(self):
        objective = self._objective()
        with patch.object(objective, "values", side_effect=self._candidate_values):
            selection = v.exhaustive_synergy_search(objective, 3, batch_size=2)
        self.assertEqual(selection.nplet, (0, 2, 3))
        self.assertAlmostEqual(selection.training_auc, 1)
        self.assertEqual(selection.n_admissible, 3)
        self.assertEqual(selection.n_candidates, 4)
        self.assertEqual(selection.n_tied, 3)
        self.assertAlmostEqual(selection.train_median_R, -2)

    def test_exhaustive_loao_traverses_each_order_once_for_all_folds(self):
        metadata = pd.DataFrame([
            dict(dataset="MA", state=state, scan_index=index, animal=animal,
                 session=f"s{index}", run="1", responsive=label)
            for index, (animal, label, state) in enumerate([
                ("A", 1, "MA_awake"), ("B", 0, "deep_propofol"),
                ("C", 1, "MA_awake"), ("C", 0, "ketamine"),
                ("D", 1, "MA_awake"), ("D", 0, "ts_selv4"),
            ])
        ])
        covariances = np.repeat(np.eye(5)[None], len(metadata), axis=0)
        responsive = metadata["responsive"].to_numpy()
        for strategy, measure in (
            ("synergy_exhaustive", "omega"),
            ("pairwise_exhaustive", "fc"),
        ):
            calls = []

            def measure_batch(scan_covariances, candidates, actual_measure, _batch_size):
                calls.append(actual_measure)
                self.assertEqual(actual_measure, measure)
                values = np.where(responsive == 1, -1.0, 1.0)
                if actual_measure == "fc":
                    values = -values
                matrix = np.tile(values, (len(candidates), 1))
                return torch.tensor(matrix, dtype=torch.float64, device=candidates.device)

            with patch.object(v, "measure_values", side_effect=measure_batch):
                selections = v.run_synergy_selection_stage(
                    covariances, metadata, (3,), strategy=strategy,
                    settings=v.SearchSettings(batch_size=4),
                )
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(selections), metadata["animal"].nunique())
            self.assertEqual(set(selections["nplet"]), {(0, 1, 2)})
            self.assertTrue((selections["n_tied"] == 10).all())

    def test_empty_admissible_set_is_preserved_in_exhaustive_and_annealing(self):
        objective = self._objective()

        def all_positive_awake(candidates):
            values = torch.ones(
                (len(candidates), len(self.labels)), dtype=torch.float64,
                device=candidates.device,
            )
            values[:, self.labels == 0] = 2
            return values

        with patch.object(objective, "values", side_effect=all_positive_awake):
            exhaustive = v.exhaustive_synergy_search(objective, 3, batch_size=2)
            self.assertEqual(exhaustive.nplet, ())
            self.assertEqual(exhaustive.n_admissible, 0)
            with patch.object(
                v, "simulated_annealing_parallel",
                return_value=(
                    torch.tensor([[[0, 1, 2]], [[0, 1, 3]]]),
                    torch.zeros((2, 1), dtype=torch.float64),
                ),
            ):
                annealed = v.simulated_annealing_synergy_search(
                    objective, 3, v.SearchSettings(restarts=2, iterations=1, early_stop=1),
                    seed=10,
                )
        self.assertEqual(annealed.nplet, ())
        self.assertEqual(annealed.admissible_restarts, 0)

    def test_annealing_penalizes_and_excludes_inadmissible_endpoints(self):
        objective = self._objective()

        def candidate_values(candidates):
            values_by_nplet = {
                (0, 1, 2): [1, 1, 0, 0],
                (0, 1, 3): [-1, -1, 1, 1],
            }
            return torch.tensor(
                [values_by_nplet[tuple(row)] for row in candidates.cpu().tolist()],
                dtype=torch.float64,
                device=candidates.device,
            )

        endpoints = torch.tensor([[0, 1, 2], [0, 1, 3]])

        def anneal(_covariances, *, candidate_score, **_kwargs):
            scores = candidate_score(endpoints)
            torch.testing.assert_close(scores, torch.tensor([-2, 1], dtype=torch.float64))
            return endpoints[:, None, :], scores[:, None]

        with patch.object(objective, "values", side_effect=candidate_values), patch.object(
            v, "simulated_annealing_parallel", side_effect=anneal,
        ):
            selection = v.simulated_annealing_synergy_search(
                objective, 3, v.SearchSettings(restarts=2, iterations=1, early_stop=1),
                seed=10,
            )
        self.assertEqual(selection.nplet, (0, 1, 3))
        self.assertEqual(selection.admissible_restarts, 1)

    def test_direction_enforced_fit_never_reverses_and_uses_constant_prior_at_bound(self):
        labels = np.array([0, 0, 1, 1])
        weights = np.full(4, 0.5)
        increasing = np.array([[0], [1], [2], [3]], dtype=float)
        positive, _, coefficient, _, bound_active = v.fit_direction_enforced_probabilities(
            increasing, increasing, labels, weights,
        )
        self.assertGreaterEqual(coefficient, 0)
        self.assertGreater(coefficient, 0)
        self.assertFalse(bound_active)
        self.assertTrue(np.all((positive >= 0) & (positive <= 1)))

        decreasing = increasing[::-1].copy()
        constant, _, coefficient, _, bound_active = v.fit_direction_enforced_probabilities(
            decreasing, decreasing, labels, weights,
        )
        self.assertEqual(coefficient, 0)
        self.assertTrue(bound_active)
        np.testing.assert_allclose(constant, 0.5)

    def test_free_direction_fc_score_is_maximum_of_both_polarities(self):
        objective = v.TrainingObjective(
            self.covariances, self.labels, self.animals, "fc", "cpu", 2,
        )
        candidates = torch.tensor([[0, 1, 2], [0, 1, 3]])
        values = objective.values(candidates)
        expected = torch.maximum(
            v.weighted_roc_auc(values, objective.labels, objective.weights),
            v.weighted_roc_auc(-values, objective.labels, objective.weights),
        )
        torch.testing.assert_close(objective.free_direction_fc_score(candidates), expected)

    def test_matched_region_features_use_only_selected_regions(self):
        def fake_measures(covariances, nplets, **_kwargs):
            nplets = torch.as_tensor(nplets, device=covariances.device)
            signal = nplets.sum(dim=-1, dtype=torch.float64)[:, None]
            omega = signal + covariances.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[None, :]
            zeros = torch.zeros_like(omega)
            return torch.stack((zeros, zeros, omega, torch.ones_like(omega)), dim=-1)

        with patch.object(v, "nplets_measures", side_effect=fake_measures):
            features = v.matched_region_features(self.covariances, (0, 1, 3))
        expected_omega = 4 + np.trace(self.covariances, axis1=-2, axis2=-1)
        expected_fc = pairwise_fisher_z(
            torch.tensor(self.covariances), torch.tensor([[0, 1, 3]]),
        )[0].numpy()
        np.testing.assert_allclose(features["omega"], expected_omega)
        np.testing.assert_allclose(features["fc_vector"], expected_fc)
        np.testing.assert_allclose(features["mean_fc"], expected_fc.mean(axis=1))
        np.testing.assert_allclose(
            features["fc_plus_omega"], np.column_stack((expected_fc, expected_omega)),
        )

    def test_synergy_evaluation_scales_matched_features_on_training_scans_only(self):
        metadata = pd.DataFrame([
            dict(dataset="MA", state=state, scan_index=index, animal=animal,
                 session=f"s{index}", run="1", responsive=label)
            for index, (animal, label, state) in enumerate([
                ("A", 1, "MA_awake"), ("B", 0, "deep_propofol"),
                ("C", 1, "MA_awake"), ("C", 0, "ketamine"),
                ("D", 1, "MA_awake"), ("D", 0, "ts_selv4"),
            ])
        ])
        rng = np.random.default_rng(23)
        samples = rng.normal(size=(6, 5, 10))
        covariances = samples @ samples.swapaxes(-1, -2) + 2 * np.eye(5)[None]
        selections = []
        for animal, _, _ in v.loao_folds(metadata):
            selections.extend([
                dict(
                    dataset="MA", order=3, held_out_animal=animal,
                    selection_measure="omega", polarity=-1, nplet=(0, 1, 2),
                    train_median_NR=-1.0, empty=False,
                ),
                dict(
                    dataset="MA", order=3, held_out_animal=animal,
                    selection_measure="fc_reference", polarity=1, nplet=(1, 2, 3),
                    train_median_NR=None, empty=False,
                ),
            ])
        with patch.object(v, "nplets_measures", side_effect=PipelineTests._fake_measures):
            result = v.evaluate_synergy_models(
                covariances, metadata, pd.DataFrame(selections), batch_size=4,
            )
        self.assertEqual(
            set(result.predictions["model"]),
            {
                "omega_direction", "mean_fc_same_regions", "fc_vector_same_regions",
                "fc_vector_plus_omega", "best_pairwise_anywhere",
            },
        )
        fit = result.fits.loc[
            (result.fits["held_out_animal"] == "A")
            & (result.fits["model"] == "fc_vector_same_regions")
        ].iloc[0]
        _, training, _ = next(
            fold for fold in v.loao_folds(metadata) if fold[0] == "A"
        )
        features = pairwise_fisher_z(
            torch.tensor(covariances[training]), torch.tensor([[0, 1, 2]]),
        )[0].numpy()
        weights = v.animal_balanced_weights(
            metadata["responsive"].to_numpy()[training],
            metadata["animal"].to_numpy()[training],
        )
        np.testing.assert_allclose(
            fit["mean"], np.average(features, axis=0, weights=weights),
        )
        self.assertEqual(len(result.predictions), 6 * 5)


if __name__ == "__main__":
    unittest.main()
