"""Registered structural audit of a frozen model; never opens verification files."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from financepaper.data.schema import FEATURE_NAMES
from financepaper.data.temporal import TEMPORAL_FIELDS, elapsed_delta
from financepaper.explanations.conditional_moments import CompiledAttributions
from financepaper.explanations.contrast_structure import (
    Expression, completion_contract, contrast_projection, channel_events, structure,
)
from financepaper.reliability.distribution_audit import CURRENT_KEYS, sha256, validate_current
from financepaper.reliability.validation import TAIWAN_GROUPS, aggregate_groups, revision_arrays


def check_budget(cfg, start):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        rss *= 1024
    if time.perf_counter()-start > cfg["budget_seconds"] or rss > cfg["max_rss_gib"]*1024**3:
        raise RuntimeError("Registered time/RSS budget exceeded; no reduced-model fallback")
    return rss


def verify_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Frozen artifact hash drift: {path}")


def run(cfg, output):
    started = time.perf_counter()
    verify_file(cfg["predictor"], cfg["predictor_sha256"])
    verify_file(cfg["partitions"], cfg["partition_sha256"])
    parts = json.loads(Path(cfg["partitions"]).read_text())
    expected = np.array(parts["revision_calibration"])
    for role, ids in parts.items():
        if role != "revision_calibration" and np.intersect1d(expected, ids).size:
            raise ValueError(f"Development-role overlap: {role}")
    chosen = np.random.default_rng(cfg["seed"]).permutation(np.sort(expected))[:cfg["customers"]]
    if len(chosen) != cfg["customers"]:
        raise ValueError("Insufficient fixed development cohort")
    # This trusted local artifact is hash-checked before deserialization. No new
    # raw dataset, outcome array, restored input or outer cache is accessed.
    bundle = joblib.load(cfg["predictor"])
    model = bundle["fits"]["xgb25"]["model"]
    prep = bundle["context"]["preprocessor"]
    background, origins = bundle["background"]["flat"], tuple(bundle["background"]["origins"])
    if set(bundle["context"]["X_train"].index) != set(parts["predictor_train"]):
        raise ValueError("Training artifact IDs differ from frozen partition")
    if not set(bundle["background"]["record_ids"]) <= set(parts["predictor_train"]):
        raise ValueError("Explainer background is not training-only")
    del bundle
    begin = time.perf_counter()
    compiler = CompiledAttributions.from_xgboost(model, background)
    compile_seconds = time.perf_counter()-begin
    if len(compiler.rectangles) > cfg["max_terms"]:
        raise RuntimeError("Compiled term budget exceeded")
    base = Expression(compiler.rectangles, compiler.coefficients)
    metadata = {f"__observed_{f}": 1. for f in FEATURE_NAMES}
    complete_delta = elapsed_delta(np.ones((len(TEMPORAL_FIELDS), 3), dtype=bool))/5.
    metadata.update({f"__delta_{f}": float(complete_delta[t, j])
                     for t, fields in enumerate(TEMPORAL_FIELDS) for j, f in enumerate(fields)})
    check_budget(cfg, started)
    print(f"compiled {len(base.rectangles)} terms in {compile_seconds:.3f}s", flush=True)
    rows, timing, preflight = [], [], None
    hashes = {cfg["predictor"]: cfg["predictor_sha256"], cfg["partitions"]: cfg["partition_sha256"]}
    for condition in cfg["conditions"]:
        folder = Path(cfg["cache"])/condition
        path = folder/"current.npz"
        verify_file(path, cfg["current_sha256"][condition])
        receipt = json.loads((folder/"complete.json").read_text())
        if receipt["current_only"].get(str(path)) != cfg["current_sha256"][condition]:
            raise ValueError("Current-only receipt disagrees with locked hash")
        hashes[str(path)] = cfg["current_sha256"][condition]
        hashes[str(folder/"complete.json")] = sha256(folder/"complete.json")
        with np.load(path, allow_pickle=False) as z:
            current = {key: z[key] for key in CURRENT_KEYS}
        validate_current(current, expected, parts["predictor_train"])
        indices = [int(np.flatnonzero(current["record_ids"] == i)[0]) for i in chosen]
        current = {key: value[indices] for key, value in current.items()}
        before, group_hidden = aggregate_groups(current["phi"], current["hidden"], FEATURE_NAMES, TAIWAN_GROUPS)
        cached, _ = aggregate_groups(current["completion_phi"][:, :cfg["completion_draws"]],
                                     current["hidden"], FEATURE_NAMES, TAIWAN_GROUPS)
        frozen = revision_arrays(before, cached, group_hidden, k=2)
        for i, record_id in enumerate(chosen):
            step = time.perf_counter()
            points = prep.transform(pd.DataFrame(current["completions"][i, :cfg["completion_draws"]],
                                                  columns=FEATURE_NAMES)).flatten(True)[0]
            points = np.asarray(points, dtype=np.float32)
            partial, fixed, grouping, available, original = completion_contract(
                origins, FEATURE_NAMES, TAIWAN_GROUPS, current["hidden"][i], points,
                expected_metadata=metadata)
            begin = time.perf_counter()
            group_base = base.project(grouping)
            group_projection_seconds = time.perf_counter()-begin
            begin = time.perf_counter()
            groups, counts = group_base.specialize(partial, fixed)
            specialize_group_seconds = time.perf_counter()-begin
            full_values = compiler.values(points) @ grouping
            np.testing.assert_allclose(full_values, cached[i], atol=5e-6, rtol=1e-6)
            np.testing.assert_allclose(groups.values(points), full_values, atol=1e-10, rtol=1e-10)
            row = dict(condition=condition, record_id=int(record_id), hidden_fields=int(current["hidden"][i].sum()),
                       eligible=bool(frozen["eligible"][i]), group_projection_seconds=group_projection_seconds,
                       specialize_group_seconds=specialize_group_seconds, group_global_terms=len(group_base.rectangles),
                       group_counts=counts, groups=structure(groups, original),
                       max_cache_error=float(np.max(np.abs(full_values-cached[i]))))
            rule = contrast_projection(before[i], available)
            if row["eligible"] != (rule is not None):
                raise AssertionError("Candidate eligibility drift")
            if not current["valid"][i] or not current["completion_valid"][i].all():
                raise ValueError("Frozen attribution validity guard failed")
            if rule is not None:
                projection, candidates, slices = rule
                begin = time.perf_counter()
                post = groups.project(projection)
                post_project_seconds = time.perf_counter()-begin
                begin = time.perf_counter()
                pre, pre_counts = group_base.project(projection).specialize(partial, fixed)
                pre_prepare_seconds = time.perf_counter()-begin
                projected_values = post.values(points)
                np.testing.assert_allclose(pre.values(points), projected_values, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(full_values @ projection, projected_values, atol=1e-10, rtol=1e-10)
                for val in (projected_values, pre.values(points), full_values @ projection):
                    np.testing.assert_array_equal(channel_events(val, slices), frozen["event"][i])
                # Bit signatures include every nonzero query term. Reuse here is
                # finite-support deduplication available to a generic evaluator.
                signatures = post.indicators(points)
                row.update(candidates=candidates.tolist(), contrast_post=structure(post, original),
                           contrast_pre=structure(pre, original), pre_counts=pre_counts,
                           post_project_seconds=post_project_seconds, pre_prepare_seconds=pre_prepare_seconds,
                           union_ratio=len(post.rectangles)/max(len(groups.rectangles), 1),
                           unique_support_signatures=len(np.unique(signatures, axis=0)),
                           max_expression_error=float(np.max(np.abs(pre.values(points)-projected_values))))
                if i < cfg["timed_customers"]:
                    # Rotate orders by repetition to reduce systematic first-call
                    # bias. No hidden truth, new completion draw or calibration.
                    methods = ["full_encoded", "specialized_groups", "post_project", "pre_project"]
                    for rep in range(cfg["timing_repetitions"]):
                        for name in methods[rep % 4:]+methods[:rep % 4]:
                            begin = time.perf_counter()
                            if name == "full_encoded":
                                value = base.values(points) @ grouping @ projection
                            elif name == "specialized_groups":
                                value = groups.values(points) @ projection
                            else:
                                value = (post if name == "post_project" else pre).values(points)
                            elapsed = time.perf_counter()-begin
                            np.testing.assert_allclose(value, projected_values, atol=1e-10, rtol=1e-10)
                            timing.append(dict(condition=condition, record_id=int(record_id), repetition=rep,
                                               method=name, evaluate_seconds=elapsed))
            row["row_seconds"] = time.perf_counter()-step
            rows.append(row)
            if preflight is None:
                preflight = dict(row_seconds=row["row_seconds"], compile_seconds=compile_seconds,
                                 estimated_128_row_seconds=row["row_seconds"]*128,
                                 peak_rss_bytes=check_budget(cfg, started))
                print("preflight "+json.dumps(preflight), flush=True)
                (output/"preflight.json").write_text(json.dumps(preflight, indent=2)+"\n")
            check_budget(cfg, started)
            # Durable progress, including all ineligible rows. Never silently skip.
            with (output/"rows.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
        print(f"{condition}: completed {len(chosen)} current records", flush=True)
    pd.DataFrame(timing).to_csv(output/"timing.csv", index=False)
    summary = {}
    for condition in cfg["conditions"]:
        block = [r for r in rows if r["condition"] == condition]
        good = [r for r in block if r["eligible"]]
        summary[condition] = dict(customers=len(block), eligible=len(good),
            median_union_ratio=float(np.median([r["union_ratio"] for r in good])) if good else None,
            median_full_group_terms=float(np.median([r["groups"]["terms"] for r in good])) if good else None,
            median_contrast_terms=float(np.median([r["contrast_post"]["terms"] for r in good])) if good else None,
            median_unique_signatures=float(np.median([r["unique_support_signatures"] for r in good])) if good else None,
            max_cache_error=max(r["max_cache_error"] for r in block),
            max_expression_error=max((r["max_expression_error"] for r in good), default=0.))
    primary = summary[cfg["primary"]]
    headroom = (primary["eligible"] >= cfg["minimum_eligible"] and
                primary["median_union_ratio"] <= cfg["union_ratio_gate"])
    return dict(status="VERIFIED_DEVELOPMENT_STRUCTURAL_AUDIT", conditions=summary,
                engineering_headroom_screen=bool(headroom), method_novelty="NOT_ESTABLISHED",
                compile_seconds=compile_seconds, global_terms=len(base.rectangles),
                encoded_dimension=compiler.dimension, background_shape=list(background.shape),
                coefficient_bytes=base.coefficients.nbytes, preflight=preflight,
                elapsed_seconds=time.perf_counter()-started, peak_rss_bytes=check_budget(cfg, started),
                financial_outcomes_used=False, verification_files_accessed=False,
                customer_ids=chosen.tolist(), artifact_hashes=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/contrast_structure_audit.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = Path(cfg["output"])
    if output.exists():
        parser.error(f"Output exists; refuse to overwrite audit: {output}")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sources = [str(args.config), cfg["protocol"], cfg["amendment"], "scripts/run_contrast_structure_audit.py",
               "src/financepaper/explanations/contrast_structure.py", "src/financepaper/explanations/conditional_moments.py",
               "src/financepaper/data/temporal.py", "src/financepaper/reliability/distribution_audit.py",
               "src/financepaper/reliability/validation.py", "tests/test_contrast_structure.py", "uv.lock"]
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), config=cfg,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
        source_sha256={s: sha256(root/s) for s in sources}, threads=1)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    try:
        with threadpool_limits(limits=1):
            result = run(cfg, output)
    except Exception as exc:
        (output/"failure.json").write_text(json.dumps(dict(status="STOPPED", error=repr(exc)), indent=2)+"\n")
        raise
    (output/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["conditions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
