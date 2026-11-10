"""Training-only animal-held-out selection and symmetric pairwise baselines."""

from dataclasses import asdict, dataclass
from itertools import combinations, islice
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from thoi.measures.gaussian_copula import nplets_measures
from tqdm.auto import tqdm

from .states import NONRESPONSIVE_STATES, RESPONSIVE_STATES
from .thoi_utils import simulated_annealing_parallel
from .utils import pairwise_fisher_z


@dataclass(frozen=True)
class SearchSettings:
    """LOPO budget, with an explicitly approved AUC-scale temperature."""

    restarts: int = 100
    iterations: int = 10000
    early_stop: int = 1000
    temperature: float = 0.05
    cooling: float = 0.99
    batch_size: int = 2048
    seed: int = 2026

    def __post_init__(self):
        counts = (self.restarts, self.iterations, self.early_stop, self.batch_size)
        if any(not isinstance(value, int) or value < 1 for value in counts):
            raise ValueError("Search counts must be positive integers.")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")
        if not np.isfinite(self.cooling) or not 0 < self.cooling <= 1:
            raise ValueError("Cooling must be in (0, 1].")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")


@dataclass(frozen=True)
class Selection:
    nplet: tuple[int, ...]
    training_auc: float
    method: str
    seed: int
    n_admissible: int | None = None
    n_candidates: int | None = None
    n_tied: int = 1
    train_median_R: float = math.nan
    train_median_NR: float = math.nan
    admissible_restarts: int | None = None


@dataclass
class LOAOResult:
    predictions: pd.DataFrame
    selections: pd.DataFrame
    metrics: pd.DataFrame
    stability: pd.DataFrame
    region_frequency: pd.DataFrame
    sessions: pd.DataFrame
    fits: pd.DataFrame


@dataclass
class SynergyLOAOResult:
    selections: pd.DataFrame
    predictions: pd.DataFrame
    fits: pd.DataFrame
    pooled: pd.DataFrame
    animal: pd.DataFrame
    animal_summary: pd.DataFrame


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _json_compatible(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_loao_manifest(
    covariance_path: str | Path,
    metadata_path: str | Path,
    settings: SearchSettings,
    datasets: Sequence[str],
    device: str,
) -> dict[str, object]:
    """Fingerprint inputs and search configuration to prevent stale checkpoint reuse."""
    return {
        "checkpoint_schema": 1,
        "algorithm_version": 1,
        "inputs": {
            "covariance_sha256": _sha256_file(covariance_path),
            "metadata_sha256": _sha256_file(metadata_path),
        },
        "settings": asdict(settings),
        "datasets": list(datasets),
        "device": str(device),
        "orders": {"exhaustive": [3, 4, 5], "annealing": [6, 7, 8, 9]},
    }


def build_synergy_manifest(
    covariance_path: str | Path,
    metadata_path: str | Path,
    settings: SearchSettings,
    datasets: Sequence[str],
    device: str,
) -> dict[str, object]:
    """Fingerprint the distinct R1_D constrained-selection analysis."""
    manifest = build_loao_manifest(
        covariance_path, metadata_path, settings, datasets, device,
    )
    manifest["analysis"] = "R1_D_synergy_loao"
    manifest["algorithm_version"] = 1
    manifest["selection_rule"] = {
        "measure": "raw_omega",
        "polarity": -1,
        "training_responsive_weighted_median_below": 0,
        "tie_break": ["most_negative_training_responsive_median", "lexicographic_nplet"],
        "pairwise_reference": "best_of_both_mean_fc_auc_directions",
    }
    manifest["criterion_3_near_chance_roc_auc"] = 0.55
    return manifest


def loao_run_id(manifest: Mapping[str, object]) -> str:
    """Return a stable run directory name for an exact manifest."""
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def configure_loao_logger(log_path: str | Path) -> logging.Logger:
    """Log progress to both the notebook console and a persistent run log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    logger = logging.getLogger(f"hoi_anesthesia.loao.{logger_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    resolved_path = path.resolve()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved_path
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


class LOAOCheckpointStore:
    """Persist completed fold/order/measure units for safe resume."""

    def __init__(self, directory: str | Path, manifest: Mapping[str, object]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = dict(manifest)
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved_manifest != self.manifest:
                raise ValueError(
                    f"Checkpoint manifest mismatch in {self.directory}; "
                    "use a directory for the matching inputs and settings."
                )
        else:
            _atomic_write_json(manifest_path, self.manifest)

    @staticmethod
    def _unit(
        strategy: str, dataset: str, animal: str, order: int, measure: str,
    ) -> dict[str, object]:
        return {
            "strategy": strategy,
            "dataset": dataset,
            "held_out_animal": str(animal),
            "order": order,
            "measure": measure,
        }

    def _path_for(self, unit: Mapping[str, object]) -> Path:
        encoded = json.dumps(unit, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return self.directory / f"task-{digest}.json"

    def load(
        self, strategy: str, dataset: str, animal: str, order: int, measure: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
        unit = self._unit(strategy, dataset, animal, order, measure)
        path = self._path_for(unit)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("status") != "complete":
            raise ValueError(f"Incomplete or invalid LOAO checkpoint: {path}")
        if payload.get("unit") != unit:
            raise ValueError(f"LOAO checkpoint identity mismatch: {path}")
        records = []
        for name in ("predictions", "selections", "fits"):
            table = payload.get(name)
            if not isinstance(table, list) or any(not isinstance(row, dict) for row in table):
                raise ValueError(f"Invalid {name} table in LOAO checkpoint: {path}")
            records.append([dict(row) for row in table])
        return records[0], records[1], records[2]

    def save(
        self,
        strategy: str,
        dataset: str,
        animal: str,
        order: int,
        measure: str,
        predictions: Sequence[Mapping[str, object]],
        selections: Sequence[Mapping[str, object]],
        fits: Sequence[Mapping[str, object]],
    ) -> Path:
        unit = self._unit(strategy, dataset, animal, order, measure)
        payload = {
            "status": "complete",
            "unit": unit,
            "predictions": _json_compatible(pd.DataFrame(predictions).to_dict(orient="records")),
            "selections": _json_compatible(pd.DataFrame(selections).to_dict(orient="records")),
            "fits": _json_compatible(pd.DataFrame(fits).to_dict(orient="records")),
        }
        path = self._path_for(unit)
        _atomic_write_json(path, payload)
        return path


def load_classification_data(
    covariances: Mapping[str, Mapping[str, np.ndarray]],
    metadata: pd.DataFrame,
    dataset: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Align classification scans to metadata; never infer animal from scan order."""
    if dataset not in RESPONSIVE_STATES:
        raise ValueError(f"Unknown dataset: {dataset}")
    required = {"dataset", "state", "scan_index", "animal", "session", "run"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata requires columns {sorted(required)}.")
    states = RESPONSIVE_STATES[dataset] + NONRESPONSIVE_STATES[dataset]
    frame = metadata.loc[
        (metadata["dataset"] == dataset) & metadata["state"].isin(states)
    ].copy()
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(f"{dataset}: missing scan metadata or identities.")
    for column in ("animal", "session", "run"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{dataset}: blank {column}.")
    indices = pd.to_numeric(frame["scan_index"], errors="raise")
    if not np.isfinite(indices).all() or (indices < 0).any() or (indices % 1 != 0).any():
        raise ValueError("Scan indices must be nonnegative integers.")
    frame["scan_index"] = indices.astype(int)
    if frame.duplicated(["state", "scan_index"]).any():
        raise ValueError(f"{dataset}: duplicate state/scan indices.")
    arrays, rows = [], []
    for state in states:
        scans = np.asarray(covariances[dataset][state], dtype=np.float64)
        state_rows = frame.loc[frame["state"] == state].sort_values("scan_index")
        if not np.array_equal(state_rows["scan_index"], np.arange(len(scans))):
            raise ValueError(f"{dataset}/{state}: metadata scan counts/indices mismatch.")
        arrays.append(scans)
        rows.append(state_rows)
    X = np.concatenate(arrays)
    if X.ndim != 3 or X.shape[1] != X.shape[2] or not np.isfinite(X).all():
        raise ValueError("Covariances must be finite (scans, regions, regions) arrays.")
    if not np.allclose(X, X.swapaxes(-1, -2), rtol=1e-8, atol=1e-10):
        raise ValueError("Covariance matrices must be symmetric.")
    if np.any(np.linalg.eigvalsh(X) <= 0):
        raise ValueError("Covariance matrices must be positive definite.")
    frame = pd.concat(rows, ignore_index=True)
    frame["responsive"] = frame["state"].isin(RESPONSIVE_STATES[dataset]).astype(int)
    return X, frame


def animal_balanced_weights(labels: np.ndarray, animals: Sequence[str]) -> np.ndarray:
    """Each class has total weight one; each animal is equal within its class."""
    y = np.asarray(labels)
    a = np.asarray(animals)
    if y.ndim != 1 or a.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned, nonempty binary labels and animal IDs are required.")
    if pd.isna(a).any() or any(not str(value).strip() for value in a):
        raise ValueError("Animal IDs cannot be missing or blank.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes are required for balanced weighting.")
    weights = np.empty(len(y), dtype=float)
    for label in (0, 1):
        animals_in_class, counts = np.unique(a[y == label], return_counts=True)
        for animal, count in zip(animals_in_class, counts):
            weights[(y == label) & (a == animal)] = 1 / (len(counts) * count)
    return weights


def loao_folds(metadata: pd.DataFrame) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """One excluded animal per fold, allowing one-class held-out sets."""
    for animal in sorted(metadata["animal"].unique()):
        test = np.flatnonzero(metadata["animal"].to_numpy() == animal)
        train = np.flatnonzero(metadata["animal"].to_numpy() != animal)
        if set(metadata.iloc[train]["responsive"].unique()) != {0, 1}:
            raise ValueError(f"Holding out {animal} leaves one-class training data.")
        yield animal, train, test


@torch.no_grad()
def weighted_roc_auc(
    scores: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    """Batched rank AUC, (candidates, scans), with exact half-credit ties."""
    if scores.ndim != 2 or labels.shape != (scores.shape[1],) or weights.shape != labels.shape:
        raise ValueError("Expected scores (candidates, scans) and aligned labels/weights.")
    if not torch.isfinite(scores).all() or not torch.isfinite(weights).all():
        raise ValueError("AUC scores and weights must be finite.")
    if torch.any(weights <= 0) or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("AUC needs binary labels and strictly positive weights.")
    positive = labels == 1
    positive_total, negative_total = weights[positive].sum(), weights[~positive].sum()
    if positive_total <= 0 or negative_total <= 0:
        raise ValueError("AUC needs both classes.")
    values, indices = scores.sort(dim=1, stable=True)
    ordered_weights = weights[indices]
    ordered_positive = positive[indices]
    neg = ordered_weights * (~ordered_positive)
    pos = ordered_weights * ordered_positive
    starts = torch.ones_like(values, dtype=torch.bool)
    starts[:, 1:] = values[:, 1:] != values[:, :-1]
    groups = starts.to(torch.long).cumsum(dim=1) - 1
    group_neg = torch.zeros_like(values).scatter_add_(1, groups, neg)
    credit = group_neg.cumsum(dim=1) - 0.5 * group_neg
    return (pos * credit.gather(1, groups)).sum(dim=1) / (positive_total * negative_total)


@torch.no_grad()
def weighted_median_by_candidate(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return each row's smallest value reaching half the total positive weight."""
    if values.ndim != 2 or weights.shape != (values.shape[1],) or values.shape[1] == 0:
        raise ValueError("Expected values (candidates, scans) and aligned scan weights.")
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
        raise ValueError("Weighted-median values and weights must be finite.")
    if torch.any(weights <= 0):
        raise ValueError("Weighted-median weights must be strictly positive.")
    ordered, indices = values.sort(dim=1, stable=True)
    ordered_weights = weights[indices]
    reached = ordered_weights.cumsum(dim=1) >= weights.sum() / 2
    positions = reached.to(torch.int64).argmax(dim=1, keepdim=True)
    return ordered.gather(1, positions).squeeze(1)


MEASURES = ("omega", "omega_norm", "fc")


@torch.no_grad()
def measure_values(
    covariances: torch.Tensor, candidates: torch.Tensor, measure: str, batch_size: int,
) -> torch.Tensor:
    """Return (candidates, scans) values: raw O, normalized O/S, or mean Fisher-z FC."""
    if measure == "fc":
        return pairwise_fisher_z(covariances, candidates).mean(dim=-1)
    if measure not in MEASURES:
        raise ValueError(f"Measure must be one of {MEASURES}.")
    values = nplets_measures(
        covariances, nplets=candidates, covmat_precomputed=True, T=None,
        device=covariances.device, batch_size=batch_size, verbose=logging.ERROR,
    )
    if measure == "omega":
        return values[..., 2]
    # S = TC + DTC >= |O|, so O/S lies in [-1, 1] and removes overall dependence strength.
    return values[..., 2] / values[..., 3]


class TrainingObjective:
    """A candidate is scored jointly across training scans, never scan pairs."""

    def __init__(
        self, covariances: np.ndarray, labels: np.ndarray, animals: Sequence[str],
        measure: str, device: str | torch.device, batch_size: int,
        synergy_rule: bool = False,
    ):
        if measure not in MEASURES:
            raise ValueError(f"Measure must be one of {MEASURES}.")
        if synergy_rule and measure != "omega":
            raise ValueError("The synergy rule applies only to raw Omega.")
        self.covariances = torch.as_tensor(covariances, dtype=torch.float64, device=device)
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.weights = torch.as_tensor(
            animal_balanced_weights(labels, animals), dtype=torch.float64, device=device,
        )
        self.measure = measure
        self.batch_size = batch_size
        self.synergy_rule = synergy_rule

    @torch.no_grad()
    def values(self, candidates: torch.Tensor) -> torch.Tensor:
        return measure_values(self.covariances, candidates, self.measure, self.batch_size)

    @torch.no_grad()
    def score_with_admissibility(
        self, candidates: torch.Tensor, polarity: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted AUC and the training-awake synergy admissibility mask."""
        return self.score_values(self.values(candidates), polarity)

    @torch.no_grad()
    def score_values(
        self, values: torch.Tensor, polarity: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score measured candidates and apply the optional synergy admissibility rule."""
        if polarity not in (-1, 1):
            raise ValueError("Polarity must be +1 or -1.")
        if values.ndim != 2 or values.shape[1] != len(self.labels):
            raise ValueError("Candidate values must have shape (candidates, training scans).")
        auc = weighted_roc_auc(polarity * values, self.labels, self.weights)
        if not self.synergy_rule:
            return auc, torch.ones_like(auc, dtype=torch.bool)
        responsive = self.labels == 1
        medians = weighted_median_by_candidate(values[:, responsive], self.weights[responsive])
        return auc, medians < 0

    def __call__(self, candidates: torch.Tensor, polarity: int = 1) -> torch.Tensor:
        if polarity not in (-1, 1):
            raise ValueError("Polarity must be +1 or -1.")
        return weighted_roc_auc(polarity * self.values(candidates), self.labels, self.weights)

    @torch.no_grad()
    def synergy_annealing_score(self, candidates: torch.Tensor) -> torch.Tensor:
        """Score -Omega by AUC and keep inadmissible candidates below the valid range."""
        if not self.synergy_rule:
            raise ValueError("Synergy annealing requires synergy_rule=True.")
        auc, admissible = self.score_with_admissibility(candidates, polarity=-1)
        return auc - (~admissible).to(auc.dtype) * 2

    @torch.no_grad()
    def free_direction_fc_score(self, candidates: torch.Tensor) -> torch.Tensor:
        """Return the better of the two weighted AUC directions for mean FC."""
        if self.measure != "fc":
            raise ValueError("Free-direction FC scoring requires measure='fc'.")
        auc = self(candidates, polarity=1)
        return torch.maximum(auc, 1 - auc)

    @torch.no_grad()
    def class_medians(self, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidate-wise weighted medians in responsive and non-responsive scans."""
        values = self.values(candidates)
        responsive = self.labels == 1
        nonresponsive = self.labels == 0
        return (
            weighted_median_by_candidate(values[:, responsive], self.weights[responsive]),
            weighted_median_by_candidate(values[:, nonresponsive], self.weights[nonresponsive]),
        )


def candidate_batches(n_regions: int, order: int, batch_size: int) -> Iterator[list[tuple[int, ...]]]:
    if not 2 <= order <= n_regions or batch_size < 1:
        raise ValueError("Require 2 <= order <= regions and positive batch size.")
    iterator = combinations(range(n_regions), order)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _update_best(
    best: dict[int, Selection],
    candidates: torch.Tensor,
    auc: torch.Tensor,
    method: str,
    seed: int,
) -> None:
    for polarity, scores in ((1, auc), (-1, 1 - auc)):
        index = int(scores.argmax().item())
        nplet = tuple(candidates[index].tolist())
        score = float(scores[index].item())
        previous = best.get(polarity)
        if previous is None or score > previous.training_auc:
            best[polarity] = Selection(nplet, score, method, seed)


def exhaustive_search(
    objective: TrainingObjective, order: int, batch_size: int = 2048,
    seed: int = 2026,
) -> dict[int, Selection]:
    """Evaluate every n-plet at orders 3-5, in GPU-sized candidate batches."""
    device = objective.covariances.device
    n_regions = objective.covariances.shape[-1]
    best: dict[int, Selection] = {}
    with tqdm(
        total=math.comb(n_regions, order),
        desc=f"Exhaustive n={order} | {objective.measure}",
        unit="candidate",
        unit_scale=True,
    ) as progress:
        for batch in candidate_batches(n_regions, order, batch_size):
            candidates = torch.tensor(batch, dtype=torch.long, device=device)
            _update_best(best, candidates, objective(candidates), "exhaustive", seed)
            progress.update(len(batch))
    return best


def simulated_annealing_search(
    objective: TrainingObjective, order: int, settings: SearchSettings,
    seed: int,
) -> dict[int, Selection]:
    """Search orders 6-9 with the shared THOI-derived annealer."""
    device = objective.covariances.device
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    best = {}
    for polarity in (1, -1):
        direction_seed = seed + int(polarity == -1)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(direction_seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(direction_seed)
            solutions, scores = simulated_annealing_parallel(
                objective.covariances, order=order, covmat_precomputed=True,
                repeat=settings.restarts, batch_size=settings.batch_size, device=device,
                max_iterations=settings.iterations, early_stop=settings.early_stop,
                initial_temp=settings.temperature, cooling_rate=settings.cooling,
                largest=True, verbose=logging.WARNING,
                candidate_score=lambda candidates: objective(candidates, polarity),
                progress_label=(
                    f"Annealing n={order} | {objective.measure} | polarity={polarity:+d}"
                ),
            )
        candidates = solutions[:, 0].sort(dim=-1).values
        score = scores[:, 0]
        maximum = score.max()
        tied = candidates[score == maximum].cpu().tolist()
        best[polarity] = Selection(
            min(tuple(row) for row in tied), float(maximum.item()),
            "annealing", direction_seed,
        )
    return best


def _selection_with_training_medians(
    objective: TrainingObjective,
    nplet: tuple[int, ...],
    auc: float,
    method: str,
    seed: int,
    *,
    n_admissible: int | None,
    n_candidates: int | None,
    n_tied: int,
    admissible_restarts: int | None = None,
) -> Selection:
    candidates = torch.tensor([nplet], dtype=torch.long, device=objective.covariances.device)
    median_r, median_nr = objective.class_medians(candidates)
    return Selection(
        nplet=nplet,
        training_auc=auc,
        method=method,
        seed=seed,
        n_admissible=n_admissible,
        n_candidates=n_candidates,
        n_tied=n_tied,
        train_median_R=float(median_r[0].item()),
        train_median_NR=float(median_nr[0].item()),
        admissible_restarts=admissible_restarts,
    )


@torch.no_grad()
def exhaustive_synergy_search(
    objective: TrainingObjective, order: int, batch_size: int = 2048,
    seed: int = 2026,
) -> Selection:
    """Exhaustively maximize animal-balanced AUC of -Omega among awake-synergistic sets."""
    if not objective.synergy_rule or objective.measure != "omega":
        raise ValueError("Exhaustive synergy search requires a raw-Omega synergy objective.")
    device = objective.covariances.device
    n_regions = objective.covariances.shape[-1]
    n_candidates = math.comb(n_regions, order)
    n_admissible = 0
    best_auc = -math.inf
    best_key: tuple[float, tuple[int, ...]] | None = None
    n_tied = 0
    with tqdm(
        total=n_candidates,
        desc=f"Exhaustive synergy n={order}",
        unit="candidate",
        unit_scale=True,
    ) as progress:
        for batch in candidate_batches(n_regions, order, batch_size):
            candidates = torch.tensor(batch, dtype=torch.long, device=device)
            values = objective.values(candidates)
            auc, admissible = objective.score_values(values, polarity=-1)
            responsive = objective.labels == 1
            median_r = weighted_median_by_candidate(
                values[:, responsive], objective.weights[responsive],
            )
            n_admissible += int(admissible.sum().item())
            for candidate_index in torch.nonzero(admissible, as_tuple=False).flatten().tolist():
                score = float(auc[candidate_index].item())
                candidate_key = (
                    float(median_r[candidate_index].item()),
                    tuple(candidates[candidate_index].tolist()),
                )
                if score > best_auc:
                    best_auc = score
                    best_key = candidate_key
                    n_tied = 1
                elif score == best_auc:
                    n_tied += 1
                    if best_key is None or candidate_key < best_key:
                        best_key = candidate_key
            progress.update(len(batch))
    if best_key is None:
        return Selection(
            (), math.nan, "exhaustive", seed, n_admissible, n_candidates, 0,
        )
    selected_nplet = best_key[1]
    return _selection_with_training_medians(
        objective, selected_nplet, best_auc, "exhaustive", seed,
        n_admissible=n_admissible, n_candidates=n_candidates, n_tied=n_tied,
    )


def _run_annealing(
    objective: TrainingObjective,
    order: int,
    settings: SearchSettings,
    seed: int,
    candidate_score: Callable[[torch.Tensor], torch.Tensor],
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = objective.covariances.device
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        solutions, scores = simulated_annealing_parallel(
            objective.covariances, order=order, covmat_precomputed=True,
            repeat=settings.restarts, batch_size=settings.batch_size, device=device,
            max_iterations=settings.iterations, early_stop=settings.early_stop,
            initial_temp=settings.temperature, cooling_rate=settings.cooling,
            largest=True, verbose=logging.WARNING, candidate_score=candidate_score,
            progress_label=label,
        )
    return solutions[:, 0].sort(dim=-1).values, scores[:, 0], seed


@torch.no_grad()
def simulated_annealing_synergy_search(
    objective: TrainingObjective, order: int, settings: SearchSettings,
    seed: int,
) -> Selection:
    """Use penalized -Omega scores and select among admissible restart endpoints."""
    if not objective.synergy_rule or objective.measure != "omega":
        raise ValueError("Synergy annealing requires a raw-Omega synergy objective.")
    direction_seed = seed + 1
    candidates, _, _ = _run_annealing(
        objective, order, settings, direction_seed, objective.synergy_annealing_score,
        f"Annealing synergy n={order}",
    )
    auc, admissible = objective.score_with_admissibility(candidates, polarity=-1)
    admissible_count = int(admissible.sum().item())
    if not admissible_count:
        return Selection(
            (), math.nan, "annealing", direction_seed, None, None, 0, math.nan, math.nan, 0,
        )
    median_r, _ = objective.class_medians(candidates)
    maximum = auc[admissible].max()
    tied_indices = torch.nonzero(admissible & (auc == maximum), as_tuple=False).flatten()
    tied = [
        (float(median_r[index].item()), tuple(candidates[index].tolist()))
        for index in tied_indices.tolist()
    ]
    selected_nplet = min(tied)[1]
    return _selection_with_training_medians(
        objective, selected_nplet, float(maximum.item()), "annealing", direction_seed,
        n_admissible=None, n_candidates=None, n_tied=len(tied),
        admissible_restarts=admissible_count,
    )


@torch.no_grad()
def exhaustive_pairwise_reference_search(
    objective: TrainingObjective, order: int, batch_size: int = 2048,
    seed: int = 2026,
) -> Selection:
    """Exhaustively find the best mean-FC n-plet in either discrimination direction."""
    if objective.measure != "fc":
        raise ValueError("The pairwise reference requires mean-FC candidates.")
    n_regions = objective.covariances.shape[-1]
    n_candidates = math.comb(n_regions, order)
    best_score = -math.inf
    best_nplet: tuple[int, ...] | None = None
    n_tied = 0
    with tqdm(
        total=n_candidates,
        desc=f"Exhaustive pairwise reference n={order}",
        unit="candidate",
        unit_scale=True,
    ) as progress:
        for batch in candidate_batches(n_regions, order, batch_size):
            candidates = torch.tensor(
                batch, dtype=torch.long, device=objective.covariances.device,
            )
            scores = objective.free_direction_fc_score(candidates)
            maximum = float(scores.max().item())
            tied = candidates[scores == scores.max()].cpu().tolist()
            if maximum > best_score:
                best_score = maximum
                best_nplet = min(tuple(row) for row in tied)
                n_tied = len(tied)
            elif maximum == best_score:
                best_nplet = min(best_nplet, *(tuple(row) for row in tied))
                n_tied += len(tied)
            progress.update(len(batch))
    if best_nplet is None:
        raise RuntimeError("Pairwise reference search evaluated no candidates.")
    return Selection(
        best_nplet, best_score, "exhaustive", seed,
        n_candidates=n_candidates, n_tied=n_tied,
    )


@torch.no_grad()
def simulated_annealing_pairwise_reference_search(
    objective: TrainingObjective, order: int, settings: SearchSettings,
    seed: int,
) -> Selection:
    """Search mean-FC candidates by the best of both AUC directions."""
    if objective.measure != "fc":
        raise ValueError("The pairwise reference requires mean-FC candidates.")
    candidates, scores, _ = _run_annealing(
        objective, order, settings, seed, objective.free_direction_fc_score,
        f"Annealing pairwise reference n={order}",
    )
    maximum = scores.max()
    tied = candidates[scores == maximum].cpu().tolist()
    return Selection(
        min(tuple(row) for row in tied), float(maximum.item()), "annealing", seed,
        n_tied=len(tied),
    )


def fit_probabilities(
    training: np.ndarray, held_out: np.ndarray, labels: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, StandardScaler, LogisticRegression]:
    """Fixed C=1, weighted training-only scaling and fitting, with no inner tuning."""
    if training.ndim != 2 or held_out.ndim != 2 or training.shape[1] != held_out.shape[1]:
        raise ValueError("Training/test features must be aligned 2-D arrays.")
    if not np.isfinite(training).all() or not np.isfinite(held_out).all():
        raise ValueError("Features must be finite.")
    scaler = StandardScaler().fit(training, sample_weight=weights)
    model = LogisticRegression(C=1, penalty="l2", solver="lbfgs", max_iter=10000)
    model.fit(scaler.transform(training), labels, sample_weight=weights)
    # ConvergenceWarning is also emitted by sklearn; never accept an exhausted fit silently.
    if np.any(model.n_iter_ >= model.max_iter):
        raise RuntimeError("Fixed-setting logistic fit did not converge.")
    probability = model.predict_proba(scaler.transform(held_out))[:, 1]
    return probability, scaler, model


def pooled_metrics(predictions: pd.DataFrame, model_column: str = "model") -> pd.DataFrame:
    rows = []
    keys = ["dataset", "order", "selection_measure", "polarity", model_column]
    for key, group in predictions.groupby(keys, sort=True):
        if group.duplicated(["state", "scan_index"]).any():
            raise ValueError("Each scan must have exactly one out-of-fold prediction per model.")
        labels = group["responsive"].to_numpy()
        weights = animal_balanced_weights(labels, group["animal"])
        probability = group["probability"].to_numpy()
        rows.append(dict(zip(keys, key), roc_auc=roc_auc_score(labels, probability, sample_weight=weights),
                         pr_auc=average_precision_score(labels, probability, sample_weight=weights),
                         weighted_prevalence=float(np.average(labels, weights=weights)),
                         positive_class="responsive", n_scans=len(group)))
    return pd.DataFrame(rows)


def animal_level_metrics(
    predictions: pd.DataFrame, model_column: str = "feature_set",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each held-out animal only against its own fold model's 0.5 boundary.

    A unit is one animal's scans of one class. Balanced accuracy averages units within
    each class, then averages the two classes, which equals scan accuracy weighted by
    ``animal_balanced_weights``. No probabilities from different fold models are compared.
    """
    keys = ["dataset", "order", "selection_measure", "polarity", model_column]
    frame = predictions.copy()
    if frame.duplicated(keys + ["state", "scan_index"]).any():
        raise ValueError("Each scan must have exactly one out-of-fold prediction per model.")
    responsive = frame["responsive"].to_numpy() == 1
    probability = frame["probability"].to_numpy()
    true_class = np.where(responsive, probability, 1 - probability)
    frame["true_class_probability"] = true_class
    frame["correct"] = np.where(true_class > 0.5, 1.0, np.where(true_class < 0.5, 0.0, 0.5))
    animal = frame.groupby(keys + ["animal", "responsive"], as_index=False, sort=True).agg(
        n_scans=("probability", "size"), accuracy=("correct", "mean"),
        true_class_probability=("true_class_probability", "mean"),
    )
    animal["majority_correct"] = animal["accuracy"] > 0.5
    by_class = animal.groupby(keys + ["responsive"], sort=True).agg(
        accuracy=("accuracy", "mean"),
        true_class_probability=("true_class_probability", "mean"),
        units_correct=("majority_correct", "sum"),
        units=("majority_correct", "size"),
    ).unstack("responsive")
    if by_class.isna().any().any() or set(by_class.columns.get_level_values(1)) != {0, 1}:
        raise ValueError("Animal-level metrics need both classes in every group.")
    summary = pd.DataFrame({
        "balanced_accuracy": (by_class[("accuracy", 0)] + by_class[("accuracy", 1)]) / 2,
        "accuracy_responsive": by_class[("accuracy", 1)],
        "accuracy_nonresponsive": by_class[("accuracy", 0)],
        "balanced_true_class_probability": (
            by_class[("true_class_probability", 0)] + by_class[("true_class_probability", 1)]
        ) / 2,
        "units_correct": (by_class[("units_correct", 0)] + by_class[("units_correct", 1)]).astype(int),
        "units": (by_class[("units", 0)] + by_class[("units", 1)]).astype(int),
    }).reset_index()
    return animal, summary


def selection_stability(selections: pd.DataFrame, n_regions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlaps, frequencies = [], []
    keys = ["dataset", "order", "selection_measure", "polarity"]
    for key, group in selections.groupby(keys, sort=True):
        selected = [(row.held_out_animal, set(row.nplet)) for row in group.itertuples()]
        for (animal_a, a), (animal_b, b) in combinations(selected, 2):
            overlaps.append(dict(zip(keys, key), animal_a=animal_a, animal_b=animal_b,
                                 jaccard=len(a & b) / len(a | b)))
        for region in range(n_regions):
            count = sum(region in nplet for _, nplet in selected)
            frequencies.append(dict(zip(keys, key), region=region, count=count,
                                    fraction=count / len(selected)))
    return pd.DataFrame(overlaps), pd.DataFrame(frequencies)


def _evaluate_selection(
    objective: TrainingObjective,
    held_covariances: torch.Tensor,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    base: dict,
    selection: Selection,
) -> tuple[list[dict], list[dict]]:
    nplet = torch.tensor(
        [selection.nplet], dtype=torch.long, device=objective.covariances.device,
    )
    train_value = objective.values(nplet)[0].cpu().numpy()
    test_value = measure_values(
        held_covariances, nplet, objective.measure, objective.batch_size,
    )[0].cpu().numpy()

    models = [
        ("scalar", base["polarity"] * train_value[:, None],
         base["polarity"] * test_value[:, None]),
    ]
    if objective.measure == "fc":
        train_vector = pairwise_fisher_z(objective.covariances, nplet)[0].cpu().numpy()
        test_vector = pairwise_fisher_z(held_covariances, nplet)[0].cpu().numpy()
        models.append(("pairwise_vector", train_vector, test_vector))
    predictions, fits = [], []
    for model_name, train_features, test_features in models:
        probability, scaler, model = fit_probabilities(
            train_features, test_features, labels, weights,
        )
        fit_base = dict(base, model=model_name)
        fits.append(dict(
            fit_base, nplet=selection.nplet, mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(), coefficient=model.coef_[0].tolist(),
            intercept=float(model.intercept_[0]),
        ))
        for position, row_index in enumerate(test_indices):
            predictions.append(dict(
                metadata.iloc[row_index].to_dict(), **fit_base, nplet=selection.nplet,
                raw_value=float(test_value[position]),
                raw_omega=float(test_value[position]) if objective.measure == "omega" else np.nan,
                probability=float(probability[position]),
            ))
    return predictions, fits


def _assemble_loao_result(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    predictions: list[dict],
    selections: list[dict],
    fits: list[dict],
) -> LOAOResult:
    prediction_frame, selection_frame = pd.DataFrame(predictions), pd.DataFrame(selections)
    if not prediction_frame.empty:
        legacy_omega_vector = (
            prediction_frame["selection_measure"].eq("omega")
            & prediction_frame["model"].eq("pairwise_vector")
        )
        prediction_frame = prediction_frame.loc[~legacy_omega_vector].reset_index(drop=True)
    fit_frame = pd.DataFrame(fits)
    if not fit_frame.empty:
        legacy_omega_vector = (
            fit_frame["selection_measure"].eq("omega")
            & fit_frame["model"].eq("pairwise_vector")
        )
        fit_frame = fit_frame.loc[~legacy_omega_vector].reset_index(drop=True)
    stability, frequency = selection_stability(selection_frame, covariances.shape[-1])
    sessions = metadata.groupby(["dataset", "animal", "state"], as_index=False).agg(
        n_sessions=("session", "nunique"), n_scans=("scan_index", "size"),
    )
    return LOAOResult(
        prediction_frame, selection_frame, pooled_metrics(prediction_frame),
        stability, frequency, sessions, fit_frame,
    )


def _clear_cuda_cache(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _validate_measures(measures: Sequence[str]) -> tuple[str, ...]:
    measures = tuple(measures)
    if not measures or len(set(measures)) != len(measures) or not set(measures) <= set(MEASURES):
        raise ValueError(f"Measures must be a nonempty, duplicate-free subset of {MEASURES}.")
    return measures


def run_loao_exhaustive(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    *,
    settings: SearchSettings = SearchSettings(),
    device: str = "cpu",
    checkpoint: LOAOCheckpointStore | None = None,
    logger: logging.Logger | None = None,
    measures: Sequence[str] = ("omega", "fc"),
) -> LOAOResult:
    """Run LOAO with exhaustive n-plet search for orders 3, 4 and 5."""
    measures = _validate_measures(measures)
    metadata = metadata.reset_index(drop=True)
    dataset = str(metadata["dataset"].iloc[0])
    predictions, selections, fits = [], [], []
    for fold_index, (animal, train, test) in enumerate(loao_folds(metadata)):
        train_rows = metadata.iloc[train]
        labels = train_rows["responsive"].to_numpy()
        weights = animal_balanced_weights(labels, train_rows["animal"])
        held_covariances = None
        active_context = f"dataset={dataset} held_out_animal={animal} fold setup"
        try:
            held_covariances = torch.as_tensor(
                covariances[test], dtype=torch.float64, device=device,
            )
            for order in (3, 4, 5):
                seed = settings.seed + fold_index * 100 + order * 2
                for measure in measures:
                    active_context = (
                        f"dataset={dataset} held_out_animal={animal} "
                        f"n={order} measure={measure}"
                    )
                    if logger:
                        logger.info("Starting exhaustive %s", active_context)
                    if checkpoint:
                        cached = checkpoint.load(
                            "exhaustive", dataset, animal, order, measure,
                        )
                        if cached is not None:
                            predictions.extend(cached[0])
                            selections.extend(cached[1])
                            fits.extend(cached[2])
                            if logger:
                                logger.info("Reused checkpoint for exhaustive %s", active_context)
                            continue

                    task_predictions, task_selections, task_fits = [], [], []
                    objective = None
                    try:
                        objective = TrainingObjective(
                            covariances[train], labels, train_rows["animal"],
                            measure, device, settings.batch_size,
                        )
                        chosen = exhaustive_search(objective, order, settings.batch_size, seed)
                        for polarity, selection in chosen.items():
                            base = dict(
                                dataset=dataset, order=order, held_out_animal=animal,
                                selection_measure=measure, polarity=polarity,
                            )
                            task_selections.append(dict(
                                base, nplet=selection.nplet, training_auc=selection.training_auc,
                                method=selection.method, seed=selection.seed, n_training_scans=len(train),
                                n_training_animals=train_rows["animal"].nunique(),
                            ))
                            fold_predictions, fold_fits = _evaluate_selection(
                                objective, held_covariances, metadata, test, labels, weights, base, selection,
                            )
                            task_predictions.extend(fold_predictions)
                            task_fits.extend(fold_fits)
                    finally:
                        del objective
                    if checkpoint:
                        checkpoint_path = checkpoint.save(
                            "exhaustive", dataset, animal, order, measure,
                            task_predictions, task_selections, task_fits,
                        )
                        if logger:
                            logger.info("Saved checkpoint %s", checkpoint_path)
                    predictions.extend(task_predictions)
                    selections.extend(task_selections)
                    fits.extend(task_fits)
                message = f"{dataset} | held-out animal {animal} | exhaustive n={order} complete"
                if logger:
                    logger.info(message)
                else:
                    print(message, flush=True)
        except Exception:
            if logger:
                logger.exception("LOAO search failed during %s", active_context)
            raise
        finally:
            del held_covariances
            _clear_cuda_cache(device)
    return _assemble_loao_result(covariances, metadata, predictions, selections, fits)


def run_loao_annealing(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    *,
    settings: SearchSettings = SearchSettings(),
    device: str = "cpu",
    checkpoint: LOAOCheckpointStore | None = None,
    logger: logging.Logger | None = None,
    measures: Sequence[str] = ("omega", "fc"),
) -> LOAOResult:
    """Run LOAO with simulated annealing for n-plet orders 6, 7, 8 and 9."""
    measures = _validate_measures(measures)
    metadata = metadata.reset_index(drop=True)
    dataset = str(metadata["dataset"].iloc[0])
    predictions, selections, fits = [], [], []
    for fold_index, (animal, train, test) in enumerate(loao_folds(metadata)):
        train_rows = metadata.iloc[train]
        labels = train_rows["responsive"].to_numpy()
        weights = animal_balanced_weights(labels, train_rows["animal"])
        held_covariances = None
        active_context = f"dataset={dataset} held_out_animal={animal} fold setup"
        try:
            held_covariances = torch.as_tensor(
                covariances[test], dtype=torch.float64, device=device,
            )
            for order in (6, 7, 8, 9):
                seed = settings.seed + fold_index * 100 + order * 2
                for measure in measures:
                    active_context = (
                        f"dataset={dataset} held_out_animal={animal} "
                        f"n={order} measure={measure}"
                    )
                    if logger:
                        logger.info("Starting annealing %s", active_context)
                    if checkpoint:
                        cached = checkpoint.load(
                            "annealing", dataset, animal, order, measure,
                        )
                        if cached is not None:
                            predictions.extend(cached[0])
                            selections.extend(cached[1])
                            fits.extend(cached[2])
                            if logger:
                                logger.info("Reused checkpoint for annealing %s", active_context)
                            continue

                    task_predictions, task_selections, task_fits = [], [], []
                    objective = None
                    try:
                        objective = TrainingObjective(
                            covariances[train], labels, train_rows["animal"],
                            measure, device, settings.batch_size,
                        )
                        chosen = simulated_annealing_search(objective, order, settings, seed)
                        for polarity, selection in chosen.items():
                            base = dict(
                                dataset=dataset, order=order, held_out_animal=animal,
                                selection_measure=measure, polarity=polarity,
                            )
                            task_selections.append(dict(
                                base, nplet=selection.nplet, training_auc=selection.training_auc,
                                method=selection.method, seed=selection.seed, n_training_scans=len(train),
                                n_training_animals=train_rows["animal"].nunique(),
                            ))
                            fold_predictions, fold_fits = _evaluate_selection(
                                objective, held_covariances, metadata, test, labels, weights, base, selection,
                            )
                            task_predictions.extend(fold_predictions)
                            task_fits.extend(fold_fits)
                    finally:
                        del objective
                    if checkpoint:
                        checkpoint_path = checkpoint.save(
                            "annealing", dataset, animal, order, measure,
                            task_predictions, task_selections, task_fits,
                        )
                        if logger:
                            logger.info("Saved checkpoint %s", checkpoint_path)
                    predictions.extend(task_predictions)
                    selections.extend(task_selections)
                    fits.extend(task_fits)
                message = f"{dataset} | held-out animal {animal} | annealing n={order} complete"
                if logger:
                    logger.info(message)
                else:
                    print(message, flush=True)
        except Exception:
            if logger:
                logger.exception("LOAO search failed during %s", active_context)
            raise
        finally:
            del held_covariances
            _clear_cuda_cache(device)
    return _assemble_loao_result(covariances, metadata, predictions, selections, fits)


def _exhaustive_search_across_folds(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    folds: Sequence[tuple[str, np.ndarray, int]],
    order: int,
    measure: str,
    synergy_rule: bool,
    settings: SearchSettings,
    device: str | torch.device,
) -> dict[str, Selection]:
    """Traverse each candidate once and update independent LOAO fold objectives."""
    scan_indices = np.unique(np.concatenate([train for _, train, _ in folds]))
    scan_positions = {int(index): position for position, index in enumerate(scan_indices)}
    scan_covariances = torch.as_tensor(
        covariances[scan_indices], dtype=torch.float64, device=device,
    )
    fold_state: dict[str, dict[str, object]] = {}
    for animal, train, fold_index in folds:
        train_rows = metadata.iloc[train]
        labels = train_rows["responsive"].to_numpy()
        fold_state[animal] = {
            "positions": np.asarray([scan_positions[int(index)] for index in train]),
            "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
            "weights": torch.as_tensor(
                animal_balanced_weights(labels, train_rows["animal"]),
                dtype=torch.float64, device=device,
            ),
            "seed": settings.seed + fold_index * 100 + order * 2,
            "best_auc": -math.inf,
            "best_nplet": None,
            "best_median_R": math.nan,
            "best_median_NR": math.nan,
            "n_tied": 0,
            "n_admissible": 0,
        }

    n_candidates = math.comb(covariances.shape[-1], order)
    with tqdm(
        total=n_candidates,
        desc=f"Exhaustive n={order} | {measure} | all pending folds",
        unit="candidate",
        unit_scale=True,
    ) as progress:
        for batch in candidate_batches(
            covariances.shape[-1], order, settings.batch_size,
        ):
            candidates = torch.tensor(batch, dtype=torch.long, device=device)
            values = measure_values(
                scan_covariances, candidates, measure, settings.batch_size,
            )
            for animal, _, _ in folds:
                state = fold_state[animal]
                positions = state["positions"]
                fold_values = values[:, positions]
                labels = state["labels"]
                weights = state["weights"]
                if synergy_rule:
                    scores = weighted_roc_auc(-fold_values, labels, weights)
                    responsive = labels == 1
                    nonresponsive = labels == 0
                    median_r = weighted_median_by_candidate(
                        fold_values[:, responsive], weights[responsive],
                    )
                    median_nr = weighted_median_by_candidate(
                        fold_values[:, nonresponsive], weights[nonresponsive],
                    )
                    admissible = median_r < 0
                    state["n_admissible"] += int(admissible.sum().item())
                else:
                    auc = weighted_roc_auc(fold_values, labels, weights)
                    scores = torch.maximum(auc, 1 - auc)
                    median_r = median_nr = None
                    admissible = torch.ones_like(scores, dtype=torch.bool)
                if not admissible.any():
                    continue
                maximum = scores[admissible].max()
                tied_indices = torch.nonzero(
                    admissible & (scores == maximum), as_tuple=False,
                ).flatten()
                if synergy_rule:
                    tied_medians = median_r[tied_indices]
                    minimum_median = tied_medians.min()
                    winner_options = tied_indices[tied_medians == minimum_median]
                    winner_index = int(winner_options[0].item())
                    candidate_key = (
                        float(minimum_median.item()),
                        tuple(candidates[winner_index].tolist()),
                    )
                    previous_key = (
                        (state["best_median_R"], state["best_nplet"])
                        if state["best_nplet"] is not None else None
                    )
                else:
                    winner_index = int(tied_indices[0].item())
                    candidate_key = tuple(candidates[winner_index].tolist())
                    previous_key = state["best_nplet"]
                score_value = float(maximum.item())
                if score_value > state["best_auc"]:
                    state["best_auc"] = score_value
                    state["best_nplet"] = candidate_key[1] if synergy_rule else candidate_key
                    state["n_tied"] = len(tied_indices)
                    if synergy_rule:
                        state["best_median_R"] = candidate_key[0]
                        state["best_median_NR"] = float(median_nr[winner_index].item())
                elif score_value == state["best_auc"]:
                    state["n_tied"] += len(tied_indices)
                    if previous_key is None or candidate_key < previous_key:
                        state["best_nplet"] = candidate_key[1] if synergy_rule else candidate_key
                        if synergy_rule:
                            state["best_median_R"] = candidate_key[0]
                            state["best_median_NR"] = float(median_nr[winner_index].item())
            progress.update(len(batch))

    selections = {}
    for animal, state in fold_state.items():
        nplet = state["best_nplet"]
        if nplet is None:
            selections[animal] = Selection(
                (), math.nan, "exhaustive", state["seed"],
                state["n_admissible"] if synergy_rule else None,
                n_candidates,
                0,
            )
        elif synergy_rule:
            selections[animal] = Selection(
                nplet, state["best_auc"], "exhaustive", state["seed"],
                state["n_admissible"], n_candidates, state["n_tied"],
                state["best_median_R"], state["best_median_NR"],
            )
        else:
            selections[animal] = Selection(
                nplet, state["best_auc"], "exhaustive", state["seed"],
                n_candidates=n_candidates, n_tied=state["n_tied"],
            )
    return selections


def _synergy_selection_record(
    dataset: str,
    order: int,
    animal: str,
    train: np.ndarray,
    train_rows: pd.DataFrame,
    selection_measure: str,
    polarity: int,
    selection: Selection,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "order": order,
        "held_out_animal": animal,
        "selection_measure": selection_measure,
        "polarity": polarity,
        "nplet": selection.nplet or None,
        "training_auc": selection.training_auc,
        "method": selection.method,
        "seed": selection.seed,
        "n_training_scans": len(train),
        "n_training_animals": train_rows["animal"].nunique(),
        "n_admissible": selection.n_admissible,
        "n_candidates": selection.n_candidates,
        "n_admissible_fraction": (
            selection.n_admissible / selection.n_candidates
            if selection.n_admissible is not None
            and selection.n_candidates is not None
            else None
        ),
        "n_tied": selection.n_tied,
        "train_median_R": selection.train_median_R,
        "train_median_NR": selection.train_median_NR,
        "admissible_restarts": selection.admissible_restarts,
        "empty": not bool(selection.nplet),
    }


def run_synergy_selection_stage(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    orders: Sequence[int],
    *,
    strategy: str,
    settings: SearchSettings,
    device: str = "cpu",
    checkpoint: LOAOCheckpointStore | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Run one checkpointed R1_D selection stage over LOAO training folds."""
    valid_strategies = {
        "synergy_exhaustive", "synergy_annealing",
        "pairwise_exhaustive", "pairwise_annealing",
    }
    if strategy not in valid_strategies:
        raise ValueError(f"Strategy must be one of {sorted(valid_strategies)}.")
    orders = tuple(orders)
    if not orders or len(set(orders)) != len(orders):
        raise ValueError("Orders must be a nonempty sequence without duplicates.")
    if any(not isinstance(order, int) or order < 3 or order > covariances.shape[-1] for order in orders):
        raise ValueError("Every search order must be between 3 and the region count.")
    if covariances.ndim != 3 or len(covariances) != len(metadata):
        raise ValueError("Covariances and metadata must contain the same scan rows.")
    metadata = metadata.reset_index(drop=True)
    if metadata["dataset"].nunique() != 1:
        raise ValueError("A selection stage must contain exactly one dataset.")
    dataset = str(metadata["dataset"].iloc[0])
    is_synergy = strategy.startswith("synergy_")
    measure = "omega" if is_synergy else "fc"
    selection_measure = "omega" if is_synergy else "fc_reference"
    method = "exhaustive" if strategy.endswith("exhaustive") else "annealing"
    rows: list[dict[str, object]] = []
    if method == "exhaustive":
        folds = list(loao_folds(metadata))
        for order in orders:
            pending = []
            train_by_animal = {}
            for fold_index, (animal, train, _) in enumerate(folds):
                context = (
                    f"strategy={strategy} dataset={dataset} "
                    f"held_out_animal={animal} n={order}"
                )
                if logger:
                    logger.info("Starting %s", context)
                if checkpoint:
                    cached = checkpoint.load(strategy, dataset, animal, order, measure)
                    if cached is not None:
                        rows.extend(cached[1])
                        if logger:
                            logger.info("Reused checkpoint for %s", context)
                        continue
                pending.append((animal, train, fold_index))
                train_by_animal[animal] = metadata.iloc[train]
            if not pending:
                continue
            try:
                selections_for_order = _exhaustive_search_across_folds(
                    covariances, metadata, pending, order, measure,
                    is_synergy, settings, device,
                )
                for animal, train, _ in pending:
                    train_rows = train_by_animal[animal]
                    record = _synergy_selection_record(
                        dataset, order, animal, train, train_rows,
                        selection_measure, -1 if is_synergy else 1,
                        selections_for_order[animal],
                    )
                    if checkpoint:
                        checkpoint_path = checkpoint.save(
                            strategy, dataset, animal, order, measure, [], [record], [],
                        )
                        if logger:
                            logger.info("Saved checkpoint %s", checkpoint_path)
                    rows.append(record)
            except Exception:
                if logger:
                    logger.exception(
                        "R1_D shared exhaustive search failed for %s order=%s",
                        strategy, order,
                    )
                raise
        return pd.DataFrame(rows).sort_values(
            ["dataset", "order", "held_out_animal"],
        ).reset_index(drop=True)

    for fold_index, (animal, train, _) in enumerate(loao_folds(metadata)):
        train_rows = metadata.iloc[train]
        labels = train_rows["responsive"].to_numpy()
        for order in orders:
            seed = settings.seed + fold_index * 100 + order * 2
            context = (
                f"strategy={strategy} dataset={dataset} "
                f"held_out_animal={animal} n={order}"
            )
            if logger:
                logger.info("Starting %s", context)
            if checkpoint:
                cached = checkpoint.load(strategy, dataset, animal, order, measure)
                if cached is not None:
                    rows.extend(cached[1])
                    if logger:
                        logger.info("Reused checkpoint for %s", context)
                    continue
            try:
                objective = TrainingObjective(
                    covariances[train], labels, train_rows["animal"], measure,
                    device, settings.batch_size, synergy_rule=is_synergy,
                )
                if strategy == "synergy_exhaustive":
                    selection = exhaustive_synergy_search(
                        objective, order, settings.batch_size, seed,
                    )
                elif strategy == "synergy_annealing":
                    selection = simulated_annealing_synergy_search(
                        objective, order, settings, seed,
                    )
                elif strategy == "pairwise_exhaustive":
                    selection = exhaustive_pairwise_reference_search(
                        objective, order, settings.batch_size, seed,
                    )
                else:
                    selection = simulated_annealing_pairwise_reference_search(
                        objective, order, settings, seed,
                    )
                record = _synergy_selection_record(
                    dataset, order, animal, train, train_rows,
                    selection_measure, -1 if is_synergy else 1, selection,
                )
                if checkpoint:
                    checkpoint_path = checkpoint.save(
                        strategy, dataset, animal, order, measure, [], [record], [],
                    )
                    if logger:
                        logger.info("Saved checkpoint %s", checkpoint_path)
                rows.append(record)
            except Exception:
                if logger:
                    logger.exception("R1_D selection failed during %s", context)
                raise
    return pd.DataFrame(rows).sort_values(
        ["dataset", "order", "held_out_animal"],
    ).reset_index(drop=True)


def _save_tables(result: object, directory: str | Path, prefix: str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in type(result).__dataclass_fields__:
        frame = getattr(result, name).copy()
        for column in ("nplet", "mean", "scale", "coefficient"):
            if column in frame:
                frame[column] = frame[column].map(json.dumps)
        frame.to_csv(directory / f"{prefix}_{name}.csv", index=False)


def save_loao_result(
    result: LOAOResult, directory: str | Path, dataset: str,
) -> None:
    """Persist flat result tables; n-plets and model arrays are JSON cells."""
    _save_tables(result, directory, dataset)


FEATURE_SETS = {
    "omega": ("omega", "hoi_vector", "omega_gradient"),
    "omega_norm": ("omega", "hoi_vector", "omega_gradient"),
    "fc": ("fc_mean", "fc_vector"),
}


@dataclass
class FeatureEvaluation:
    predictions: pd.DataFrame
    fits: pd.DataFrame
    pooled: pd.DataFrame
    animal: pd.DataFrame
    animal_summary: pd.DataFrame


@torch.no_grad()
def nplet_feature_sets(
    covariances: torch.Tensor, nplet: Sequence[int], selection_measure: str, batch_size: int,
) -> dict[str, np.ndarray]:
    """Per-scan feature sets of the selection measure's own family, each (scans, features).

    Omega selections (O or O/S) get only high-order features and FC selections only
    pairwise features, so families are never crossed. Features use only each scan's own
    covariance, so slicing training/held-out rows afterwards cannot leak across folds.
    """
    if selection_measure not in FEATURE_SETS:
        raise ValueError(f"Selection measure must be one of {tuple(FEATURE_SETS)}.")
    candidate = torch.as_tensor([tuple(nplet)], dtype=torch.long, device=covariances.device)
    if selection_measure == "fc":
        pairs = pairwise_fisher_z(covariances, candidate)[0].cpu().numpy()
        return {"fc_mean": pairs.mean(axis=1, keepdims=True), "fc_vector": pairs}
    hoi = nplets_measures(
        covariances, nplets=candidate, covmat_precomputed=True, T=None,
        device=covariances.device, batch_size=batch_size, verbose=logging.ERROR,
    )[0].cpu().numpy()
    omega = _omega_variant(hoi, selection_measure)
    order = candidate.shape[1]
    if order - 1 < 3:
        # Omega of a pair is exactly 0 (TC = DTC = mutual information).
        reduced = np.zeros((hoi.shape[0], order))
    else:
        keep = torch.tensor(
            [[j for j in range(order) if j != i] for i in range(order)],
            dtype=torch.long, device=covariances.device,
        )
        reduced_hoi = nplets_measures(
            covariances, nplets=candidate[0][keep], covmat_precomputed=True, T=None,
            device=covariances.device, batch_size=batch_size, verbose=logging.ERROR,
        ).cpu().numpy()
        reduced = _omega_variant(reduced_hoi, selection_measure).T
    return {
        "omega": omega[:, None],
        "hoi_vector": hoi[:, :2],
        # First-order O-information gradient: column i is Omega(X) - Omega(X without region i).
        "omega_gradient": omega[:, None] - reduced,
    }


def _omega_variant(hoi: np.ndarray, selection_measure: str) -> np.ndarray:
    """O from (..., [TC, DTC, O, S]) arrays, or O/S for normalized selection."""
    if selection_measure == "omega":
        return hoi[..., 2]
    return hoi[..., 2] / hoi[..., 3]


def _as_nplet(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(int(region) for region in value)


def evaluate_feature_sets(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 2048,
    selection_measures: Sequence[str] = tuple(FEATURE_SETS),
) -> FeatureEvaluation:
    """Refit each selection's own feature sets on its fold's training-selected n-plet.

    Held-out probabilities come from L2 logistic models (C=1, training-only weighted
    scaling). O and O/S selections: Omega scalar (the selection measure), (TC, DTC)
    vector, and first-order Omega gradient vector (one value per region). FC selections:
    mean Fisher-z scalar and pairwise Fisher-z vector.
    """
    metadata = metadata.reset_index(drop=True)
    measures = tuple(selection_measures)
    if not measures or not set(measures) <= set(FEATURE_SETS):
        raise ValueError(f"Selection measures must be a nonempty subset of {tuple(FEATURE_SETS)}.")
    keys = ["dataset", "order", "selection_measure", "polarity"]
    rows = selections.loc[selections["selection_measure"].isin(measures)].copy()
    if rows.empty:
        raise ValueError("No selections for the requested measures.")
    rows["held_out_animal"] = rows["held_out_animal"].astype(str)
    if set(rows["dataset"]) != set(metadata["dataset"]) or rows["dataset"].nunique() != 1:
        raise ValueError("Selections and metadata must describe the same single dataset.")
    if rows.duplicated(keys + ["held_out_animal"]).any():
        raise ValueError("Each fold/order/measure/polarity needs exactly one selection.")
    folds = {animal: (train, test) for animal, train, test in loao_folds(metadata)}
    for key, group in rows.groupby(keys):
        if set(group["held_out_animal"]) != set(folds):
            raise ValueError(f"Selections for {dict(zip(keys, key))} do not cover every fold.")

    all_covariances = torch.as_tensor(covariances, dtype=torch.float64, device=device)
    all_labels = metadata["responsive"].to_numpy()
    all_animals = metadata["animal"].to_numpy()
    cache: dict[tuple[tuple[int, ...], str], dict[str, np.ndarray]] = {}
    predictions, fits = [], []
    for row in rows.sort_values(keys + ["held_out_animal"]).itertuples(index=False):
        nplet = _as_nplet(row.nplet)
        cache_key = (nplet, row.selection_measure)
        if cache_key not in cache:
            cache[cache_key] = nplet_feature_sets(
                all_covariances, nplet, row.selection_measure, batch_size,
            )
        features = cache[cache_key]
        train, test = folds[row.held_out_animal]
        labels = all_labels[train]
        weights = animal_balanced_weights(labels, all_animals[train])
        base = dict(
            dataset=row.dataset, order=int(row.order), held_out_animal=row.held_out_animal,
            selection_measure=row.selection_measure, polarity=int(row.polarity), nplet=nplet,
        )
        for feature_set in FEATURE_SETS[row.selection_measure]:
            values = features[feature_set]
            probability, scaler, model = fit_probabilities(
                values[train], values[test], labels, weights,
            )
            fits.append(dict(
                base, feature_set=feature_set, n_features=values.shape[1],
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coefficient=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
            ))
            scalar = features["fc_mean" if row.selection_measure == "fc" else "omega"]
            for position, index in enumerate(test):
                predictions.append(dict(
                    metadata.iloc[index].to_dict(), **base, feature_set=feature_set,
                    selection_value=float(scalar[index, 0]),
                    probability=float(probability[position]),
                ))
    del all_covariances
    _clear_cuda_cache(device)
    prediction_frame = pd.DataFrame(predictions)
    animal, summary = animal_level_metrics(prediction_frame, model_column="feature_set")
    return FeatureEvaluation(
        prediction_frame, pd.DataFrame(fits),
        pooled_metrics(prediction_frame, model_column="feature_set"), animal, summary,
    )


def save_feature_evaluation(
    result: FeatureEvaluation, directory: str | Path, prefix: str,
) -> None:
    """Persist feature-set evaluation tables as ``<prefix>_<table>.csv``."""
    _save_tables(result, directory, prefix)


def _direction_logistic_loss_and_gradient(
    parameters: np.ndarray,
    training: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    coefficient, intercept = parameters
    logits = coefficient * training + intercept
    residual = weights * (expit(logits) - labels)
    loss = 0.5 * coefficient**2 + np.sum(
        weights * (np.logaddexp(0, logits) - labels * logits),
    )
    gradient = np.array([
        coefficient + np.dot(residual, training),
        residual.sum(),
    ])
    return float(loss), gradient


def fit_direction_enforced_probabilities(
    training: np.ndarray,
    held_out: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, StandardScaler, float, float, bool]:
    """Fit a weighted 1-D L2 logistic model whose standardized slope cannot be negative."""
    training = np.asarray(training, dtype=float)
    held_out = np.asarray(held_out, dtype=float)
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if training.ndim != 2 or training.shape[1] != 1:
        raise ValueError("Direction-enforced logistic regression requires one feature.")
    if held_out.ndim != 2 or held_out.shape[1] != 1:
        raise ValueError("Held-out features must have one column.")
    if len(training) != len(labels) or weights.shape != labels.shape or len(training) == 0:
        raise ValueError("Training features, labels and weights must align.")
    if not np.isfinite(training).all() or not np.isfinite(held_out).all():
        raise ValueError("Features must be finite.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Weights must be finite and strictly positive.")
    if not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary training classes are required.")
    scaler = StandardScaler().fit(training, sample_weight=weights)
    train_value = scaler.transform(training)[:, 0]
    test_value = scaler.transform(held_out)[:, 0]
    prior = float(np.average(labels, weights=weights))
    initial_intercept = math.log(prior / (1 - prior))

    fit = minimize(
        _direction_logistic_loss_and_gradient,
        x0=np.array([0.0, initial_intercept]),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, None), (None, None)),
        args=(train_value, labels, weights),
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Direction-enforced logistic fit failed: {fit.message}")
    coefficient = max(float(fit.x[0]), 0.0)
    bound_active = coefficient <= 1e-10
    if bound_active:
        coefficient = 0.0
        intercept = initial_intercept
        probability = np.full(len(held_out), prior, dtype=float)
    else:
        intercept = float(fit.x[1])
        probability = expit(coefficient * test_value + intercept)
    return probability, scaler, coefficient, intercept, bound_active


@torch.no_grad()
def matched_region_features(
    covariances: np.ndarray | torch.Tensor,
    nplet: Sequence[int],
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Return raw Omega, mean Fisher-z FC, FC vector and FC-plus-Omega for one n-plet."""
    covariances = torch.as_tensor(covariances, dtype=torch.float64)
    regions = tuple(int(region) for region in nplet)
    if covariances.ndim != 3 or covariances.shape[-1] != covariances.shape[-2]:
        raise ValueError("Covariances must have shape (scans, regions, regions).")
    if len(regions) < 3 or len(set(regions)) != len(regions):
        raise ValueError("A matched-region n-plet needs at least three unique regions.")
    if min(regions) < 0 or max(regions) >= covariances.shape[-1]:
        raise ValueError("N-plet region indices are out of range.")
    candidates = torch.tensor([regions], dtype=torch.long, device=covariances.device)
    omega = nplets_measures(
        covariances, nplets=candidates, covmat_precomputed=True, T=None,
        device=covariances.device, batch_size=batch_size, verbose=logging.ERROR,
    )[0, :, 2].cpu().numpy()
    fc_vector = pairwise_fisher_z(covariances, candidates)[0].cpu().numpy()
    mean_fc = fc_vector.mean(axis=1)
    return {
        "omega": omega,
        "mean_fc": mean_fc,
        "fc_vector": fc_vector,
        "fc_plus_omega": np.column_stack((fc_vector, omega)),
    }


def _fit_synergy_model(
    model_name: str,
    training: np.ndarray,
    held_out: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if model_name == "omega_direction":
        probability, scaler, coefficient, intercept, bound_active = (
            fit_direction_enforced_probabilities(training, held_out, labels, weights)
        )
        return probability, {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficient": [coefficient],
            "intercept": intercept,
            "direction_bound_active": bound_active,
        }
    probability, scaler, model = fit_probabilities(training, held_out, labels, weights)
    return probability, {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "direction_bound_active": False,
    }


def evaluate_synergy_models(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 2048,
) -> SynergyLOAOResult:
    """Evaluate the five planned models on held-out animals and build report tables."""
    metadata = metadata.reset_index(drop=True)
    if covariances.ndim != 3 or len(covariances) != len(metadata):
        raise ValueError("Covariances and metadata must contain the same scan rows.")
    if metadata["dataset"].nunique() != 1:
        raise ValueError("Synergy evaluation requires one dataset at a time.")
    dataset = str(metadata["dataset"].iloc[0])
    rows = selections.loc[selections["dataset"] == dataset].copy()
    synergy_rows = rows.loc[rows["selection_measure"] == "omega"]
    reference_rows = rows.loc[rows["selection_measure"] == "fc_reference"]
    if synergy_rows.empty or reference_rows.empty:
        raise ValueError("Both Omega and best-pairwise selection results are required.")
    if synergy_rows.duplicated(["order", "held_out_animal"]).any():
        raise ValueError("Omega selections must be unique per order and held-out animal.")
    if reference_rows.duplicated(["order", "held_out_animal"]).any():
        raise ValueError("Pairwise selections must be unique per order and held-out animal.")
    folds = {animal: (train, test) for animal, train, test in loao_folds(metadata)}
    for frame, label in ((synergy_rows, "Omega"), (reference_rows, "pairwise")):
        for order, group in frame.groupby("order"):
            if set(group["held_out_animal"]) != set(folds):
                raise ValueError(f"{label} selections for order {order} do not cover every fold.")

    labels_all = metadata["responsive"].to_numpy()
    animals_all = metadata["animal"].to_numpy()
    predictions: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    reference_lookup = reference_rows.set_index(["order", "held_out_animal"])
    for row in synergy_rows.sort_values(["order", "held_out_animal"]).itertuples(index=False):
        train, test = folds[row.held_out_animal]
        reference = reference_lookup.loc[(row.order, row.held_out_animal)]
        if isinstance(reference, pd.DataFrame):
            raise ValueError("Pairwise selection lookup is not unique.")
        reference_nplet = reference["nplet"]
        if not isinstance(reference_nplet, (tuple, list, np.ndarray)):
            reference_nplet = json.loads(reference_nplet) if isinstance(reference_nplet, str) else None
        if reference_nplet is None:
            raise ValueError("A pairwise reference selection cannot be empty.")
        reference_nplet = tuple(int(region) for region in reference_nplet)
        train_labels = labels_all[train]
        train_weights = animal_balanced_weights(train_labels, animals_all[train])
        train_nonresponsive = train_labels == 0
        train_covariances = torch.as_tensor(covariances[train], dtype=torch.float64, device=device)
        test_covariances = torch.as_tensor(covariances[test], dtype=torch.float64, device=device)

        reference_train = matched_region_features(train_covariances, reference_nplet, batch_size)
        reference_test = matched_region_features(test_covariances, reference_nplet, batch_size)
        reference_train_median_fc_nr = float(weighted_median_by_candidate(
            torch.as_tensor(
                reference_train["mean_fc"][None, train_nonresponsive],
                dtype=torch.float64,
            ),
            torch.as_tensor(train_weights[train_nonresponsive], dtype=torch.float64),
        )[0].item())
        reference_base = {
            "dataset": dataset, "order": int(row.order),
            "held_out_animal": row.held_out_animal,
            "selection_measure": "omega", "polarity": -1,
            "model": "best_pairwise_anywhere", "nplet": reference_nplet,
            "regions_source": "pairwise_reference",
        }
        reference_probability, reference_fit = _fit_synergy_model(
            "mean_fc", reference_train["mean_fc"][:, None],
            reference_test["mean_fc"][:, None], train_labels, train_weights,
        )
        fits.append(dict(
            reference_base, n_features=1,
            training_median_fc_NR=reference_train_median_fc_nr,
            **reference_fit,
        ))
        for position, index in enumerate(test):
            predictions.append(dict(
                metadata.iloc[index].to_dict(), **reference_base,
                raw_omega=np.nan, raw_fc_mean=float(reference_test["mean_fc"][position]),
                training_median_NR=None,
                training_median_fc_NR=reference_train_median_fc_nr,
                probability=float(reference_probability[position]),
            ))

        if row.nplet is None or (
            isinstance(row.nplet, (float, np.floating)) and math.isnan(row.nplet)
        ):
            continue
        nplet = _as_nplet(row.nplet)
        omega_train = matched_region_features(train_covariances, nplet, batch_size)
        omega_test = matched_region_features(test_covariances, nplet, batch_size)
        training_median_fc_nr = float(weighted_median_by_candidate(
            torch.as_tensor(
                omega_train["mean_fc"][None, train_nonresponsive], dtype=torch.float64,
            ),
            torch.as_tensor(train_weights[train_nonresponsive], dtype=torch.float64),
        )[0].item())
        model_inputs = (
            ("omega_direction", -omega_train["omega"][:, None], -omega_test["omega"][:, None], 1),
            ("mean_fc_same_regions", omega_train["mean_fc"][:, None],
             omega_test["mean_fc"][:, None], 1),
            ("fc_vector_same_regions", omega_train["fc_vector"],
             omega_test["fc_vector"], omega_train["fc_vector"].shape[1]),
            ("fc_vector_plus_omega", omega_train["fc_plus_omega"],
             omega_test["fc_plus_omega"], omega_train["fc_plus_omega"].shape[1]),
        )
        for model_name, train_features, test_features, n_features in model_inputs:
            base = {
                "dataset": dataset, "order": int(row.order),
                "held_out_animal": row.held_out_animal,
                "selection_measure": "omega", "polarity": -1,
                "model": model_name, "nplet": nplet,
                "regions_source": "omega_selection",
            }
            probability, fit_record = _fit_synergy_model(
                model_name, train_features, test_features, train_labels, train_weights,
            )
            fits.append(dict(
                base, n_features=n_features,
                training_median_NR=float(row.train_median_NR),
                training_median_fc_NR=training_median_fc_nr,
                **fit_record,
            ))
            for position, index in enumerate(test):
                predictions.append(dict(
                    metadata.iloc[index].to_dict(), **base,
                    raw_omega=float(omega_test["omega"][position]),
                    raw_fc_mean=float(omega_test["mean_fc"][position]),
                    training_median_NR=float(row.train_median_NR),
                    training_median_fc_NR=training_median_fc_nr,
                    probability=float(probability[position]),
                ))
        del train_covariances, test_covariances

    prediction_frame = pd.DataFrame(predictions)
    fit_frame = pd.DataFrame(fits)
    pooled, animal, animal_summary = synergy_metric_tables(prediction_frame)
    return SynergyLOAOResult(
        rows.reset_index(drop=True), prediction_frame, fit_frame, pooled, animal, animal_summary,
    )


def synergy_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize pooled probabilities and held-out animal/class units, including empty folds."""
    pooled_rows: list[dict[str, object]] = []
    animal_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    keys = ["dataset", "order", "selection_measure", "polarity", "model"]
    for key, group in predictions.groupby(keys, sort=True):
        labels = group["responsive"].to_numpy()
        probability = group["probability"].to_numpy()
        if group.duplicated(["state", "scan_index"]).any():
            raise ValueError("Each scan must have exactly one held-out prediction per model.")
        has_both_classes = set(np.unique(labels)) == {0, 1}
        if has_both_classes:
            weights = animal_balanced_weights(labels, group["animal"])
            roc_auc = roc_auc_score(labels, probability, sample_weight=weights)
            pr_auc = average_precision_score(labels, probability, sample_weight=weights)
            prevalence = float(np.average(labels, weights=weights))
            unit_frame, unit_summary = animal_level_metrics(
                group, model_column="model",
            )
            animal_rows.append(unit_frame)
            summary_record = unit_summary.iloc[0].to_dict()
        else:
            roc_auc = math.nan
            pr_auc = math.nan
            prevalence = float(np.mean(labels)) if len(labels) else math.nan
            frame = group.copy()
            true_class = np.where(labels == 1, probability, 1 - probability)
            frame["true_class_probability"] = true_class
            frame["correct"] = np.where(
                true_class > 0.5, 1.0, np.where(true_class < 0.5, 0.0, 0.5),
            )
            unit_frame = frame.groupby(
                keys + ["animal", "responsive"], as_index=False, sort=True,
            ).agg(
                n_scans=("probability", "size"), accuracy=("correct", "mean"),
                true_class_probability=("true_class_probability", "mean"),
            )
            unit_frame["majority_correct"] = unit_frame["accuracy"] > 0.5
            animal_rows.append(unit_frame)
            class_rows = unit_frame.groupby("responsive")["accuracy"].mean()
            summary_record = {
                **dict(zip(keys, key)),
                "balanced_accuracy": math.nan,
                "accuracy_responsive": class_rows.get(1, math.nan),
                "accuracy_nonresponsive": class_rows.get(0, math.nan),
                "balanced_true_class_probability": math.nan,
                "units_correct": int(unit_frame["majority_correct"].sum()),
                "units": len(unit_frame),
            }
        pooled_rows.append({
            **dict(zip(keys, key)), "roc_auc": roc_auc, "pr_auc": pr_auc,
            "weighted_prevalence": prevalence, "positive_class": "responsive",
            "n_scans": len(group),
        })
        summary_rows.append(summary_record)
    return (
        pd.DataFrame(pooled_rows),
        pd.concat(animal_rows, ignore_index=True) if animal_rows else pd.DataFrame(),
        pd.DataFrame(summary_rows),
    )


def print_synergy_report(
    selections: pd.DataFrame,
    predictions: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    near_chance_roc_auc_threshold: float = 0.55,
) -> None:
    """Print admissibility, held-out synergy, ROC-/PR-AUC and criterion summaries."""
    if not 0.5 <= near_chance_roc_auc_threshold <= 1:
        raise ValueError("Near-chance ROC-AUC threshold must be in [0.5, 1].")
    for dataset in sorted(selections["dataset"].dropna().unique()):
        print(f"\n{dataset}")
        print(
            "Pooled OOF ROC-AUC and PR-AUC compare scores from fold-specific models; "
            "interpret them as descriptive when fold score scales differ."
        )
        local_selections = selections.loc[selections["dataset"] == dataset]
        exhaustive = local_selections.loc[
            (local_selections["method"] == "exhaustive")
            & local_selections["selection_measure"].eq("omega")
        ]
        if not exhaustive.empty:
            counts = exhaustive.groupby("order").agg(
                n_admissible_median=("n_admissible", "median"),
                n_admissible_min=("n_admissible", "min"),
                n_admissible_max=("n_admissible", "max"),
                fraction_median=("n_admissible_fraction", "median"),
                fraction_min=("n_admissible_fraction", "min"),
                fraction_max=("n_admissible_fraction", "max"),
            )
            print("Admissible exhaustive candidates and fraction per order (median, range):")
            print(counts.to_string())
        empty = local_selections.loc[local_selections["empty"] == True]
        if empty.empty:
            print("Empty selections: none")
        else:
            print("Empty selections (order, held-out animal, method):")
            print(empty[["order", "held_out_animal", "method"]].to_string(index=False))
        print("Selections:")
        print(local_selections[
            ["order", "held_out_animal", "selection_measure", "nplet", "training_auc",
             "n_tied", "train_median_R", "train_median_NR", "admissible_restarts"]
        ].to_string(index=False))

        local_predictions = predictions.loc[predictions["dataset"] == dataset]
        omega_predictions = local_predictions.loc[
            local_predictions["model"] == "omega_direction",
        ]
        synergy_rows = []
        for order, group in omega_predictions.groupby("order", sort=True):
            awake = group.loc[group["responsive"] == 1, "raw_omega"]
            nonresponsive = group.loc[group["responsive"] == 0, "raw_omega"]
            median_r = float(awake.median()) if len(awake) else math.nan
            median_nr = float(nonresponsive.median()) if len(nonresponsive) else math.nan
            synergy_rows.append({
                "order": order, "median_R": median_r, "median_NR": median_nr,
                "fraction_R_below_zero": float((awake < 0).mean()) if len(awake) else math.nan,
                "direction_holds": median_r < median_nr,
            })
        synergy_table = pd.DataFrame(
            synergy_rows,
            columns=[
                "order", "median_R", "median_NR", "fraction_R_below_zero",
                "direction_holds",
            ],
        )
        print("Held-out synergy check:")
        print(synergy_table.to_string(index=False))

        pooled_local = pooled.loc[pooled["dataset"] == dataset]
        model_names = (
            "omega_direction", "mean_fc_same_regions", "fc_vector_same_regions",
            "fc_vector_plus_omega", "best_pairwise_anywhere",
        )
        main_rows = []
        for order in sorted(local_selections["order"].unique()):
            row: dict[str, object] = {"order": order}
            for model_name in model_names:
                model_pooled = pooled_local.loc[
                    (pooled_local["order"] == order)
                    & (pooled_local["model"] == model_name),
                ]
                if model_pooled.empty:
                    row[model_name] = "NA"
                else:
                    metric = model_pooled.iloc[0]
                    row[model_name] = (
                        f"ROC-AUC={metric['roc_auc']:.3f}; PR-AUC={metric['pr_auc']:.3f}"
                    )
            main_rows.append(row)
        print("Pooled out-of-fold ROC-AUC and PR-AUC by order:")
        print(pd.DataFrame(main_rows).to_string(index=False))

        vector = pooled_local.loc[
            pooled_local["model"] == "fc_vector_same_regions",
            ["order", "roc_auc", "pr_auc"],
        ].rename(
            columns={"roc_auc": "fc_vector_roc_auc", "pr_auc": "fc_vector_pr_auc"},
        )
        combined = pooled_local.loc[
            pooled_local["model"] == "fc_vector_plus_omega",
            ["order", "roc_auc", "pr_auc"],
        ].rename(
            columns={"roc_auc": "combined_roc_auc", "pr_auc": "combined_pr_auc"},
        )
        gain = vector.merge(combined, on="order", how="outer")
        gain["delta_roc_auc"] = gain["combined_roc_auc"] - gain["fc_vector_roc_auc"]
        gain["delta_pr_auc"] = gain["combined_pr_auc"] - gain["fc_vector_pr_auc"]
        print("Incremental gain (FC vector + Omega minus FC vector; both AUC metrics):")
        print(gain.to_string(index=False))

        finite_pooled = pooled_local.loc[pooled_local["order"].between(3, 9)]
        for metric in ("roc_auc", "pr_auc"):
            finite_metric = finite_pooled.loc[finite_pooled[metric].notna()]
            print(f"Post-hoc maximum {metric.upper()} across orders 3-9:")
            if finite_metric.empty:
                print("No post-hoc maximum is available.")
            else:
                maxima = finite_metric.groupby("model")[metric].idxmax()
                print(finite_metric.loc[
                    maxima, ["order", "model", metric],
                ].to_string(index=False))

        for order in sorted(local_selections["order"].unique()):
            values = pooled_local.loc[pooled_local["order"] == order].set_index("model")
            synergy = synergy_table.loc[synergy_table["order"] == order]
            omega_auc = (
                values.loc["omega_direction", "roc_auc"]
                if "omega_direction" in values.index else math.nan
            )
            criterion_1 = bool(np.isfinite(omega_auc) and omega_auc > 0.5)
            criterion_2 = bool(
                not synergy.empty
                and synergy.iloc[0]["median_R"] < 0
                and synergy.iloc[0]["direction_holds"]
            )
            mean_fc_auc = (
                values.loc["mean_fc_same_regions", "roc_auc"]
                if "mean_fc_same_regions" in values.index else math.nan
            )
            vector_auc = (
                values.loc["fc_vector_same_regions", "roc_auc"]
                if "fc_vector_same_regions" in values.index else math.nan
            )
            combined_auc = (
                values.loc["fc_vector_plus_omega", "roc_auc"]
                if "fc_vector_plus_omega" in values.index else math.nan
            )
            pairwise_near_chance = (
                np.isfinite(mean_fc_auc) and np.isfinite(vector_auc)
                and mean_fc_auc <= near_chance_roc_auc_threshold
                and vector_auc <= near_chance_roc_auc_threshold
            )
            criterion_3 = bool(
                pairwise_near_chance
                or (np.isfinite(combined_auc) and np.isfinite(vector_auc) and combined_auc > vector_auc)
            )
            median_r = synergy.iloc[0]["median_R"] if not synergy.empty else math.nan
            median_nr = synergy.iloc[0]["median_NR"] if not synergy.empty else math.nan
            fraction_negative = (
                synergy.iloc[0]["fraction_R_below_zero"] if not synergy.empty else math.nan
            )
            print(
                f"Order {order} criteria: "
                f"1={'PASS' if criterion_1 else 'FAIL'} "
                f"(Omega pooled ROC-AUC={omega_auc:.3f}, threshold > 0.5); "
                f"2={'PASS' if criterion_2 else 'FAIL'} "
                f"(median R/NR={median_r:.3f}/{median_nr:.3f}, "
                f"R<0 fraction={fraction_negative:.3f}); "
                f"3={'PASS' if criterion_3 else 'FAIL'} "
                f"(mean-FC/vector ROC-AUC={mean_fc_auc:.3f}/{vector_auc:.3f}, "
                f"combined delta ROC-AUC={combined_auc - vector_auc:.3f}, "
                f"near-chance ceiling={near_chance_roc_auc_threshold:.2f})"
            )


def save_synergy_result(
    result: SynergyLOAOResult, directory: str | Path, dataset: str,
) -> None:
    """Persist the six R1_D tables as ``<dataset>_<table>.csv``."""
    _save_tables(result, directory, dataset)
