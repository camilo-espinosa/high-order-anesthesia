"""Order-5 search benchmark using training data only; run explicitly from the CLI."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.hoi_anesthesia.io import load_covariance_dict
from src.hoi_anesthesia.validation import (
    SearchSettings, TrainingObjective, exhaustive_search, load_classification_data,
    loao_folds, simulated_annealing_search,
)


def synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["MA", "DBS"], default=["MA", "DBS"])
    parser.add_argument("--animals", nargs="+", help="Optional held-out animal IDs to benchmark.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--covariances", type=Path, default=ROOT / "data" / "covariance_matrices_gc.h5")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "scan_metadata.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "diagnostics" / "02_search_benchmark")
    args = parser.parse_args()
    settings = SearchSettings()
    covariances = load_covariance_dict(args.covariances)
    metadata = pd.read_csv(args.metadata, dtype={"animal": str, "session": str, "run": str})
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    matched_animals = set()
    for dataset in args.datasets:
        X, scans = load_classification_data(covariances, metadata, dataset)
        for fold_index, (animal, train, _) in enumerate(loao_folds(scans)):
            if args.animals and animal not in args.animals:
                continue
            matched_animals.add(animal)
            for measure in ("omega", "fc"):
                objective = TrainingObjective(
                    X[train], scans.iloc[train]["responsive"].to_numpy(),
                    scans.iloc[train]["animal"], measure, args.device, settings.batch_size,
                )
                seed = settings.seed + fold_index * 100 + 10
                synchronize(args.device)
                start = perf_counter()
                exact = exhaustive_search(objective, 5, settings.batch_size, seed)
                synchronize(args.device)
                exhaustive_seconds = perf_counter() - start

                synchronize(args.device)
                start = perf_counter()
                annealed = simulated_annealing_search(objective, 5, settings, seed)
                synchronize(args.device)
                annealing_seconds = perf_counter() - start

                for polarity in (1, -1):
                    exact_selection = exact[polarity]
                    annealed_selection = annealed[polarity]
                    a, b = set(exact_selection.nplet), set(annealed_selection.nplet)
                    rows.append(dict(
                        dataset=dataset, held_out_animal=animal, measure=measure, order=5,
                        polarity=polarity, exhaustive_auc=exact_selection.training_auc,
                        annealing_auc=annealed_selection.training_auc,
                        auc_gap=exact_selection.training_auc - annealed_selection.training_auc,
                        exhaustive_nplet=json.dumps(exact_selection.nplet),
                        annealing_nplet=json.dumps(annealed_selection.nplet),
                        exact_set_match=exact_selection.nplet == annealed_selection.nplet,
                        jaccard=len(a & b) / len(a | b),
                        exhaustive_seconds_both_polarities=exhaustive_seconds,
                        annealing_seconds_both_polarities=annealing_seconds,
                        n_training_scans=len(train), annealing_seed=annealed_selection.seed,
                    ))
                pd.DataFrame(rows).to_csv(args.output / "comparison.csv", index=False)
                print(f"{dataset} held-out {animal}, {measure}: {rows[-2:]}", flush=True)
                del objective
    if args.animals and set(args.animals) != matched_animals:
        raise ValueError(f"Unknown requested animals: {sorted(set(args.animals) - matched_animals)}")
    if not rows:
        raise ValueError("No matching training folds.")
if __name__ == "__main__":
    main()
