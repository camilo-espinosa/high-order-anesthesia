import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.hoi_anesthesia import diagnostics as dg


class PrimaryPreprocessingTests(unittest.TestCase):
    def setUp(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks" / "00_preprocessing.ipynb"
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        self.source = "".join(next(cell["source"] for cell in cells if cell["id"] == "3a4419d3"))
        rng = np.random.default_rng(2026)
        self.states = {
            dataset: {state: rng.normal(size=(2, 500, 82))}
            for dataset, state in (("MA", "MA_awake"), ("DBS", "DBS_awake"))
        }
        self.metadata = pd.DataFrame([
            {"dataset": dataset, "state": state, "scan_index": index}
            for dataset, states in self.states.items()
            for state, scans in states.items()
            for index in range(len(scans))
        ])
        self.writer = Mock()

    def run_preprocessing(self):
        namespace = {
            "np": np, "pd": pd, "Path": Path, "dg": dg,
            "generate_covmats": self.writer,
        }
        with patch.object(pd, "read_csv", return_value=self.metadata), patch.object(
            dg, "load_timeseries", side_effect=lambda path, dataset: self.states[dataset],
        ):
            exec(compile(self.source, "00_preprocessing.ipynb", "exec"), namespace)
        return namespace["all_states"]

    def test_full_scan_regional_gsr_precedes_exact_250_point_matching(self):
        actual = self.run_preprocessing()
        for dataset, states in self.states.items():
            for state, scans in states.items():
                self.assertEqual(actual[dataset][state].shape, (2, 250, 82))
                for index, scan in enumerate(scans):
                    z = (scan - scan.mean(axis=0)) / scan.std(axis=0)
                    design = np.column_stack([np.ones(500), z.mean(axis=1)])
                    residual = scan - design @ np.linalg.lstsq(design, scan, rcond=None)[0]
                    expected = residual[:250] if dataset == "MA" else residual[::2]
                    np.testing.assert_allclose(actual[dataset][state][index], expected, atol=1e-12)
        self.writer.assert_called_once_with(actual, "data")

    def test_metadata_mismatch_fails_before_writing(self):
        self.metadata.loc[0, "scan_index"] = 99
        with self.assertRaisesRegex(ValueError, "scan counts/indices"):
            self.run_preprocessing()
        self.writer.assert_not_called()

    def test_invalid_timeseries_fail_before_writing(self):
        for invalid in ("short", "nonfinite", "constant"):
            with self.subTest(invalid=invalid):
                scans = self.states["MA"]["MA_awake"].copy()
                if invalid == "short":
                    scans = scans[:, :249]
                elif invalid == "nonfinite":
                    scans[0, 0, 0] = np.nan
                else:
                    scans[:, :, 0] = 1
                original = self.states["MA"]["MA_awake"]
                self.states["MA"]["MA_awake"] = scans
                with self.assertRaises(ValueError):
                    self.run_preprocessing()
                self.writer.assert_not_called()
                self.states["MA"]["MA_awake"] = original
