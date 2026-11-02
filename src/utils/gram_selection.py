"""Shared GRAM k selection from completed detection runs, never downstream scores.

Scopes: scenario=(dataset, noise_type, rho, beta), noise_mean=(dataset, rho,
beta), global=(beta). Average available seeds first, then give noise types
and dataset/rho scenarios equal weight. Beta is always a grouping key.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean

from config.datasets import DATASETS
from config.methods import GRAM_GRID_BETAS, GRAM_GRID_NEIGHBORS, gram_grid_variants, method_config
from config.path import PATHS
from .run_logging import completed_run_exists

NOISE_TYPES = ("symmetric", "pairflip", "instance")
RHOS = (0.2, 0.4)
SCOPE_KEYS = {"scenario": ("dataset", "noise_type", "rho", "beta"),
              "noise_mean": ("dataset", "rho", "beta"), "global": ("beta",)}
REQUIRED_FILES = ("run_args.json", "config.json", "metrics.csv", "queries.csv",
                  "predictions.csv", "summary.json")


def variant_for(beta: float, k: int) -> str:
    return gram_grid_variants(betas=(beta,), neighbors=(k,), kernels=("adaptive",),
                              acquisitions=("repair_iv",))[0]


def budget_averages(rows: list[dict], sample_count: int) -> tuple[float, float]:
    """Normalized trapezoidal means over actual query counts (first point to rho)."""
    rows = sorted(rows, key=lambda r: float(r["configured_budget_fraction"]))
    x = [int(r["verification_budget"]) / sample_count for r in rows]
    values = [[float(r["unverified_auprc"]) for r in rows],
              [int(r["queried_mislabels_found"]) / int(r["verification_budget"]) for r in rows]]
    if len(x) < 2 or any(b <= a for a, b in zip(x, x[1:])):
        raise ValueError("verification budgets must be strictly increasing")
    if not all(math.isfinite(v) and 0 <= v <= 1 for series in values for v in series):
        raise ValueError("invalid detection metrics")
    return tuple(sum((a+b)*(d-c)/2 for a,b,c,d in zip(v,v[1:],x,x[1:])) / (x[-1]-x[0])
                 for v in values)


def load_runs(results_root=PATHS.results_root, *, datasets=tuple(DATASETS),
              noise_types=NOISE_TYPES, rhos=RHOS, betas=GRAM_GRID_BETAS) -> list[dict]:
    """Load only the complete full-model beta x k pool; require seed 0 per candidate."""
    root = Path(results_root)
    records, contexts = [], defaultdict(set)
    for dataset, noise, rho, beta, k in product(datasets, noise_types, rhos, betas, GRAM_GRID_NEIGHBORS):
        variant = variant_for(beta, k)
        base = root / "detection" / dataset / noise / f"rho{rho:g}" / "ours" / variant
        available = []
        expected = json.loads(json.dumps(method_config("ours", variant)))
        for directory in sorted(base.glob("seed*")):
            suffix = directory.name.removeprefix("seed")
            if not suffix.isdigit() or not completed_run_exists(
                directory, protocol="full_pool_detection", required_files=REQUIRED_FILES,
            ):
                continue
            seed = int(suffix)
            config = json.loads((directory / "config.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            identity = dict(dataset=dataset, noise_type=noise, rho=rho, method="ours", seed=seed)
            if (any(summary.get(key) != value for key,value in identity.items())
                or any(config.get(key) != identity[key] for key in ("dataset", "method"))
                or any(config["scenario"].get(key) != identity[key] for key in ("noise_type", "rho", "seed"))
                or config.get("protocol") != "full_pool_detection" or summary.get("protocol") != "full_pool_detection"
                or config.get("gram_variant") != variant or summary.get("gram_variant") != variant
                or config.get("method_config") != expected):
                raise ValueError(f"detection configuration mismatch: {directory}")
            with (directory / "metrics.csv").open(newline="") as stream:
                metrics = sorted(csv.DictReader(stream), key=lambda r: float(r["configured_budget_fraction"]))
            budgets = [float(r["configured_budget_fraction"]) for r in metrics]
            expected_budgets = [.001, .005, .01, .025] + [round(.05*i,12) for i in range(1, round(rho/.05)+1)]
            counts = [int(r["verification_budget"]) for r in metrics]
            if (budgets != expected_budgets or budgets != summary["budgets"]
                or budgets != config["scenario"]["budgets"] or counts != config["budget_checkpoint_target_counts"]
                or counts[-1] != summary["actual_budget"]):
                raise ValueError(f"detection budget mismatch: {directory}")
            fingerprint = summary.get("shared_training_artifact_fingerprint")
            if not fingerprint or fingerprint != config["training_protocol"].get("shared_training_artifact_fingerprint"):
                raise ValueError(f"training fingerprint mismatch: {directory}")
            contexts[dataset,noise,rho,seed].add(json.dumps({
                "fingerprint":fingerprint,"dataset_config":config["dataset_config"],
                "model_config":config["model_config"],"budgets":budgets,"counts":counts,
                "sample_count":summary["sample_count"],"errors":summary["injected_mislabel_count"],
            },sort_keys=True))
            for r in metrics:
                if (not math.isclose(float(r["verification_fraction"]),int(r["verification_budget"])/summary["sample_count"],abs_tol=1e-5)
                    or not math.isclose(float(r["query_precision"]),int(r["queried_mislabels_found"])/int(r["verification_budget"]),abs_tol=1e-5)):
                    raise ValueError(f"metric/count mismatch: {directory}")
            ap, precision = budget_averages(metrics, summary["sample_count"])
            records.append({**identity,"beta":beta,"iv_beta":beta,"graph_neighbors":k,
                            "gram_variant":variant,"auprc":ap,"precision":precision,
                            "source":str((directory/"metrics.csv").relative_to(root))})
            available.append(seed)
        if 0 not in available:
            raise ValueError(f"missing completed seed0 for k selection: {base}")
    if any(len(values) != 1 for values in contexts.values()):
        raise ValueError("k candidates do not share the same training/noise/budget context")
    return records


def k_candidates(runs: list[dict], scope="scenario") -> list[dict]:
    """Build candidate means from per-run records; seed counts never weight scenarios."""
    if scope not in SCOPE_KEYS:
        raise ValueError(f"unknown selection scope: {scope}")
    grouped = defaultdict(list)
    seen = set()
    for r in runs:
        beta = float(r.get("beta", r.get("iv_beta")))
        k = int(r["graph_neighbors"])
        if beta not in GRAM_GRID_BETAS or k not in GRAM_GRID_NEIGHBORS:
            raise ValueError("unsupported beta/k candidate")
        variant = variant_for(beta,k)
        if r.get("gram_variant") != variant:
            raise ValueError(f"k selection requires the full adaptive model: {r.get('gram_variant')}")
        key = (r["dataset"],r["noise_type"],float(r["rho"]),beta,k)
        run_key = (*key,int(r["seed"]))
        if run_key in seen:
            raise ValueError(f"duplicate k-selection run: {run_key}")
        seen.add(run_key)
        for metric in ("auprc", "precision"):
            if not math.isfinite(float(r[metric])) or not 0 <= float(r[metric]) <= 1:
                raise ValueError(f"invalid {metric}")
        grouped[key].append(r)
    scenario = []
    for (dataset,noise,rho,beta,k), rows in grouped.items():
        seeds = sorted(int(r["seed"]) for r in rows)
        if 0 not in seeds:
            raise ValueError("each k candidate requires completed seed0")
        variant = variant_for(beta,k)
        scenario.append(dict(dataset=dataset,noise_type=noise,rho=rho,beta=beta,iv_beta=beta,
            graph_neighbors=k,method="ours",gram_variant=variant,setting=f"ours/{variant}",
            auprc=mean(float(r["auprc"]) for r in rows),precision=mean(float(r["precision"]) for r in rows),
            seed_count=len(seeds),seed_count_max=len(seeds),seeds=",".join(map(str,seeds)),seed=None,
            sources=" | ".join(sorted(str(r["source"]) for r in rows)),noise_count=1,
            scenario_count=1,dataset_count=1,rho_count=1))
    # Fail rather than silently selecting from an incomplete k sweep.
    coverage = defaultdict(set)
    for r in scenario:coverage[tuple(r[key] for key in SCOPE_KEYS["scenario"])].add(r["graph_neighbors"])
    if not coverage or any(ks != set(GRAM_GRID_NEIGHBORS) for ks in coverage.values()):
        raise ValueError("k selection requires all five k candidates for each scenario/beta")
    if scope != "scenario":
        noise_groups = defaultdict(list)
        for r in scenario:noise_groups[r["dataset"],r["rho"],r["beta"],r["graph_neighbors"]].append(r)
        averaged = []
        for rows in noise_groups.values():
            if {r["noise_type"] for r in rows} != set(NOISE_TYPES):
                raise ValueError("noise-mean selection requires all three noise types")
            averaged.append(_average(rows,noise_type="all_three_averaged"))
        scenario = averaged
    if scope == "global":
        global_groups = defaultdict(list)
        for r in scenario:global_groups[r["beta"],r["graph_neighbors"]].append(r)
        scenario = [_average(rows,dataset="all",rho=None) for rows in global_groups.values()]
    for r in scenario:r["selection_score"] = r["auprc"] + r["precision"]
    return sorted(scenario,key=lambda r:tuple(str(r[k]) for k in SCOPE_KEYS[scope])+(r["graph_neighbors"],))


def _average(rows, **identity):
    first = rows[0]
    return {**first,**identity,"auprc":mean(r["auprc"] for r in rows),
        "precision":mean(r["precision"] for r in rows),
        "seed_count":min(r["seed_count"] for r in rows),"seed_count_max":max(r["seed_count_max"] for r in rows),
        "sources":" | ".join(r["sources"] for r in rows),"seeds":"",
        "noise_count":len(NOISE_TYPES),"scenario_count":len({(r["dataset"],r["rho"]) for r in rows}),
        "dataset_count":len({r["dataset"] for r in rows}),"rho_count":len({r["rho"] for r in rows})}


def select_k(candidates: list[dict], scope="scenario") -> list[dict]:
    """Choose one k per fixed-beta group; both coordinates use the same candidate."""
    keys = SCOPE_KEYS[scope]
    grouped = defaultdict(list)
    for r in candidates:grouped[tuple(r[k] for k in keys)].append(r)
    return [min(rows,key=lambda r:(-(r["auprc"]+r["precision"]),-r["auprc"],-r["precision"],r["graph_neighbors"]))
            for rows in grouped.values()]


def get_best_k(dataset, noise_type, rho, beta=1.0, *, results_root=PATHS.results_root) -> dict:
    """Public single-scenario query; returns k, canonical variant, metrics and seed coverage."""
    runs = load_runs(results_root,datasets=(dataset,),noise_types=(noise_type,),rhos=(rho,),betas=(beta,))
    return select_k(k_candidates(runs))[0]
