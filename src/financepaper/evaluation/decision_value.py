"""Read-only F0 audit of frozen releases. Utility units are valid-reason benefits."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from financepaper.reliability.reason_sets import strongest, verified


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def check_hashes(entries):
    for path, expected in entries.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"MISSING historical artifact: {path}; recover at historical commit")
        if digest(path) != expected:
            raise ValueError(f"Historical artifact hash mismatch: {path}")


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def folder(root, dataset, fold):
    return Path(root) / dataset / f"fold_{fold}" if dataset == "taiwan" else Path(root) / dataset


def input_inventory(config):
    """Validate the historical receipt chain before opening evaluation arrays."""
    root = Path(config["source_root"])
    files = set()
    for name, fields in (("assessment_complete.json", ("artifacts", "calibration_freeze")),
                         ("calibration_freeze.json", ("artifacts", "protocol"))):
        p = root / name
        receipt = json.loads(p.read_text())
        files.add(p)
        for field in fields:
            check_hashes(receipt[field])
            files.update(map(Path, receipt[field]))
    freeze = json.loads((root / "protocol_freeze.json").read_text())
    for dataset, spec in config["datasets"].items():
        for fold in spec["folds"]:
            for condition in spec["conditions"]:
                p = folder(config["historical_root"], dataset, fold) / "outer" / condition / "current.npz"
                check_hashes({str(p): freeze["sources"][str(p)]})
                files.add(p)
    # Analysis tables are comparison evidence, never substitutes for row data.
    for name in ("metrics.csv", "runtime_summary.csv", "runtime_receipt.json"):
        p = root / "analysis" / name
        if p.exists():
            files.add(p)
    timing_receipt = root / "analysis/runtime_receipt.json"
    if timing_receipt.exists():
        timing = json.loads(timing_receipt.read_text())
        check_hashes(timing["artifacts"])
        files.update(map(Path, timing["artifacts"]))
    report_receipt = root / "analysis/report_receipt.json"
    report = json.loads(report_receipt.read_text())
    metrics = str(root / "analysis/metrics.csv")
    check_hashes({metrics: report["outputs"][metrics]})
    files.add(report_receipt)
    return {str(p): digest(p) for p in sorted(files)}


def load_dataset(config, dataset):
    meta, evs, truths, releases = [], [], [], []
    spec = config["datasets"][dataset]
    for fold in spec["folds"]:
        base = folder(config["source_root"], dataset, fold)
        # Trusted local artifact, hash-checked in input_inventory; no training.
        policies = joblib.load(base / "policies.joblib")["policies"]
        for condition in spec["conditions"]:
            p = base / "outer" / condition
            ev = load_npz(p / "reason_evidence.npz")
            frozen = load_npz(p / "decisions.npz")
            current = load_npz(folder(config["historical_root"], dataset, fold) / "outer" / condition / "current.npz")
            if not np.array_equal(current["record_ids"], ev["record_ids"]):
                raise ValueError("Current/evidence row identity mismatch")
            candidate = ev["candidate"]
            if not np.array_equal(candidate, (ev["phi"] > .01) & ~ev["hidden"]):
                raise ValueError("Candidate definition drift")
            selected = {}
            for method in config["methods"]:
                if method == "top1":
                    value = strongest(ev, 1)
                elif method == "mask_warning":
                    value = candidate.copy()  # Disclosure only; no invented behavioral response.
                else:
                    key = (config["target"], method, config["alpha"], config["conservative"])
                    value = policies[key].release(ev)
                    label = f"{key[0]}|{key[1]}|{key[2]}|{int(key[3])}"
                    if not np.array_equal(value, frozen[label]):
                        raise ValueError(f"Frozen policy replay mismatch: {dataset}/{fold}/{condition}/{method}")
                if value.dtype != bool or value.shape != candidate.shape or np.any(value & ~candidate):
                    raise ValueError("Invalid release mask")
                selected[method] = value
            # Only after all current-only releases have been reconstructed.
            truth = load_npz(p / "verification_only.npz")
            if not np.array_equal(truth["meaningful"], verified(truth["restored_phi"], ev["hidden"], "meaningful")):
                raise ValueError("Meaningful target drift")
            meta.append(pd.DataFrame({"record_id": ev["record_ids"], "cluster": ev["cluster_ids"],
                                      "fold": fold, "condition": condition}))
            evs.append(candidate)
            truths.append(truth["meaningful"])
            releases.append(selected)
    meta = pd.concat(meta, ignore_index=True)
    if meta.duplicated(["record_id", "condition"]).any():
        raise ValueError("Duplicate customer-condition rows")
    if (meta.groupby("cluster").fold.nunique() != 1).any():
        raise ValueError("Cluster crosses folds")
    if (meta.groupby("record_id").cluster.nunique() != 1).any():
        raise ValueError("Record cluster changes across masks")
    return (meta, np.concatenate(evs), np.concatenate(truths),
            {m: np.concatenate([r[m] for r in releases]) for m in config["methods"]})


def cluster_weights(meta, draws, seed):
    """Keep all masks/rows of a cluster together; resample within historical fold."""
    ids, inverse = np.unique(meta.cluster.to_numpy(), return_inverse=True)
    folds = meta.groupby("cluster").fold.first().reindex(ids).to_numpy()
    weights = np.zeros((draws, len(ids)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for fold in np.unique(folds):
        positions = np.flatnonzero(folds == fold)
        for i in range(draws):
            weights[i, positions] = np.bincount(rng.integers(len(positions), size=len(positions)), minlength=len(positions))
    return weights, inverse


def cluster_totals(values, inverse, size):
    return np.bincount(inverse, weights=values, minlength=size)


def interval(samples):
    finite = np.asarray(samples)[np.isfinite(samples)]
    return np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)


def ratio_samples(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)


def pareto_flags(valid, failed):
    """Maximize valid reasons, minimize failed reasons; equal points are co-frontier."""
    v, f = np.asarray(valid), np.asarray(failed)
    return np.array([not np.any((v >= a) & (f <= b) & ((v > a) | (f < b))) for a, b in zip(v, f)])


def audit_dataset(config, dataset, output):
    meta, candidate, truth, releases = load_dataset(config, dataset)
    weights, inverse = cluster_weights(meta, config["bootstrap_draws"], config["seed"])
    rows, utility, whole, checks = [], [], [], []
    reported = pd.read_csv(Path(config["source_root"]) / "analysis/metrics.csv")
    timing = pd.read_csv(Path(config["source_root"]) / "analysis/runtime_summary.csv")
    masks = {"overall": np.ones(len(meta), bool)}
    masks.update({c: meta.condition.to_numpy() == c for c in config["datasets"][dataset]["conditions"]})
    for condition, mask in tqdm(masks.items(), desc=f"{dataset} conditions", unit="condition"):
        n = int(mask.sum())
        sample_n = weights @ cluster_totals(mask, inverse, weights.shape[1])
        release_all_valid = (candidate & truth).sum(1) * mask
        release_all_failed = (candidate & ~truth).sum(1) * mask
        for method, release in releases.items():
            count = release.sum(1) * mask
            failed = (release & ~truth).sum(1) * mask
            valid = count - failed
            sample_v = weights @ cluster_totals(valid, inverse, weights.shape[1])
            sample_f = weights @ cluster_totals(failed, inverse, weights.shape[1])
            sample_dv = weights @ cluster_totals(valid - release_all_valid, inverse, weights.shape[1])
            sample_df = weights @ cluster_totals(failed - release_all_failed, inverse, weights.shape[1])
            runtime = timing[(timing.dataset == dataset) & (timing.method == method)]
            row = dict(dataset=dataset, condition=condition, target="meaningful_positive_per_reason",
                       method=method, historical_alpha=config["alpha"], n_cases=n,
                       n_records=int(meta.loc[mask, "record_id"].nunique()), n_clusters=int(meta.loc[mask, "cluster"].nunique()),
                       candidate_reasons=int(candidate[mask].sum()), released_reasons=int(count.sum()),
                       valid_reasons=int(valid.sum()), failed_reasons=int(failed.sum()),
                       cases_ge1=int((count > 0).sum()), cases_ge2=int((count >= 2).sum()),
                       coverage_ge1=float((count > 0).sum() / n), coverage_ge2=float((count >= 2).sum() / n),
                       reason_risk=float(failed.sum() / count.sum()) if count.sum() else np.nan,
                       compute_seconds_per_16_case_batch_reported=float(runtime.iloc[0].total_seconds_mean) if len(runtime) else np.nan,
                       compute_status="REPORTED_historical_5_repetitions" if len(runtime) else "NOT_RUN",
                       compute_measurement_scope="fold0_mcar30_first16_only_NOT_this_condition" if len(runtime) else "NOT_RUN",
                       compute_cost="UNKNOWN", delay_cost="UNKNOWN", evidence="REPRODUCED_cached_artifact_audit")
            rows.append(row)
            whole.append(dict(dataset=dataset, condition=condition, method=method,
                              target="any_meaningful_failure_among_released_not_top2_revision",
                              cases_with_release=int((count > 0).sum()), cases_with_any_failure=int((failed > 0).sum()),
                              risk=float((failed > 0).sum() / (count > 0).sum()) if (count > 0).any() else np.nan))
            old = reported[(reported.dataset == dataset) & (reported.condition == condition) &
                           (reported.target == "meaningful") & (reported.method == method) &
                           (reported.alpha == config["alpha"]) & (reported.conservative == 0)]
            if method not in ("top1", "mask_warning"):
                if len(old) != 1:
                    raise ValueError("Missing/ambiguous historical comparison row")
                for field in ("released_reasons", "failed_reasons", "candidate_reasons"):
                    match = row[field] == int(old.iloc[0][field])
                    checks.append(dict(dataset=dataset, condition=condition, method=method, field=field,
                                       recomputed=row[field], reported=int(old.iloc[0][field]), match=match))
                    if not match:
                        raise ValueError(f"Historical count mismatch: {checks[-1]}")
            for cost in config["cost_ratios"]:
                value = valid.sum() - cost * failed.sum()
                samples = ratio_samples(sample_v - cost * sample_f, sample_n)
                delta = ratio_samples(sample_dv - cost * sample_df, sample_n)
                lo, hi = interval(samples)
                dlo, dhi = interval(delta)
                utility.append(dict(dataset=dataset, condition=condition, method=method, c_over_b=cost,
                                    utility_total=float(value), utility_per_case=float(value / n),
                                    lo=float(lo), hi=float(hi), delta_vs_release_all_per_case=float((value - (release_all_valid.sum() - cost * release_all_failed.sum())) / n),
                                    delta_lo=float(dlo), delta_hi=float(dhi), finite_bootstrap_draws=int(np.isfinite(samples).sum()),
                                    units="valid_reason_benefit", compute_delay="NOT_SUBTRACTED_UNKNOWN"))
    counts = pd.DataFrame(rows)
    counts["pareto_frontier"] = False
    for _, group in counts.groupby("condition"):
        counts.loc[group.index, "pareto_frontier"] = pareto_flags(group.valid_reasons, group.failed_reasons)
    util = pd.DataFrame(utility)
    maximum = util.groupby(["condition", "c_over_b"]).utility_total.transform("max")
    util["grid_winner"] = np.isclose(util.utility_total, maximum)
    for name, table in (("counts", counts), ("grid", util), ("whole_explanation", pd.DataFrame(whole)),
                        ("reconciliation", pd.DataFrame(checks))):
        table.to_csv(output / f"utility_{dataset}_{name}.csv", index=False)
    return counts, util


def break_even(counts):
    rows = []
    for (dataset, condition), block in counts.groupby(["dataset", "condition"]):
        baseline = block[block.method == "release_all"].iloc[0]
        for row in block.itertuples():
            lost = int(baseline.valid_reasons - row.valid_reasons)
            avoided = int(baseline.failed_reasons - row.failed_reasons)
            interpretation = ("policy wins strictly above ratio before unknown compute/delay" if avoided > 0 else
                              "identical gross utility at every ratio" if lost == 0 else
                              "never beats release_all at nonnegative cost ratios")
            rows.append(dict(dataset=dataset, condition=condition, method=row.method, valid_lost=lost,
                             failures_avoided=avoided, c_over_b=lost / avoided if avoided > 0 else np.nan,
                             interpretation=interpretation))
    return pd.DataFrame(rows)


def plot_frontiers(counts, utility, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, dataset in enumerate(("taiwan", "polish")):
        c = counts[(counts.dataset == dataset) & (counts.condition == "overall")]
        u = utility[(utility.dataset == dataset) & (utility.condition == "overall")]
        for i, row in enumerate(c.itertuples()):
            color = plt.get_cmap("tab10")(i)
            axes[0, col].scatter(row.failed_reasons, row.valid_reasons, color=color, marker="s" if row.pareto_frontier else "x")
            q = u[u.method == row.method]
            axes[1, col].plot(q.c_over_b, q.utility_per_case, marker=".", color=color, label=row.method)
        frontier = c[c.pareto_frontier].drop_duplicates(["failed_reasons", "valid_reasons"]).sort_values("failed_reasons")
        axes[0, col].plot(frontier.failed_reasons, frontier.valid_reasons, "k--", alpha=.4)
        axes[0, col].set(title=dataset, xlabel="Failed reasons (lower is better)", ylabel="Valid reasons (higher is better)")
        axes[1, col].set(xlabel="Assumed cost / benefit", ylabel="Utility / customer-condition case")
        axes[1, col].legend(fontsize=7, ncol=2)
    fig.suptitle("Exploratory frozen-policy audit; compute and delay costs unknown\nSquares = Pareto frontier; equal masks may overlap")
    fig.tight_layout()
    fig.savefig(output / "utility_frontier.png", dpi=170)
    fig.savefig(output / "utility_frontier.pdf")
    plt.close(fig)
