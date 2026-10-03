"""Frozen development audit, distinct from historical assessment and serving."""
from pathlib import Path
import json
import time
import platform

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import spearmanr

from financepaper.data.polish import load_polish, verification_masks
from financepaper.experiments.decisive_validation import (
    hashes, validate_hashes, write_json, polish_adapter,
)
from financepaper.reliability.validation import POLISH_NAMES, POLISH_GROUPS, aggregate_groups
from financepaper.reliability.reason_sets import evidence, verified, strongest, matched_random
from financepaper.evaluation.verification_headroom import (
    actual_reveal_states, optimize_oracle, selected_outputs, paired_intervals,
)


def freeze(config):
    cfg = yaml.safe_load(Path(config).read_text())
    root, hist = Path(cfg["output"]), Path(cfg["historical_root"])
    if root.exists():
        raise FileExistsError("Never overwrite an audit; use its frozen evaluate/report commands")
    X, _, clusters = load_polish(cfg["polish_path"])
    partpath = hist/"polish/partition_ids.json"
    parts = json.loads(partpath.read_text())
    groups = pd.Series(clusters, index=X.index)
    excluded = set().union(*(set(groups.loc[parts[k]]) for k in
                             ("outer", "train", "default_calibration", "release_calibration", "revision_calibration")))
    candidates = groups.loc[parts["selector_train"]].sort_index()
    candidates = candidates[~candidates.isin(excluded)].drop_duplicates()
    ids = np.sort(np.random.default_rng(cfg["seed"]).choice(candidates.index,
                  cfg["customers"], replace=False))
    twpaths = [Path(f"outputs/revision_study/fold_{f}/partition_ids.json") for f in range(3)]
    tw = [json.loads(p.read_text()) for p in twpaths]
    outer = set().union(*(set(p["outer"]) for p in tw))
    pool = set().union(*(set(v) for p in tw for k, v in p.items() if k != "outer"))
    if pool-outer:
        raise RuntimeError("Taiwan availability differs from predeclared no-eligible-cohort audit")
    predictor = hist/"polish/predictors.joblib"
    policy = Path(cfg["policy_root"])/"polish/policies.joblib"
    # Verify artifacts against their preexisting frozen receipts, not only fresh hashes.
    for receipt, key in [(hist/"polish/assessment_freeze.json", "artifacts"),
                         (Path(cfg["policy_root"])/"polish/calibration_complete.json", "artifacts")]:
        validate_hashes(json.loads(receipt.read_text())[key])
    sources = list(Path("src/financepaper").rglob("*.py")) + twpaths
    sources += [Path(config), Path("scripts/run_verification_headroom.py"), Path("uv.lock"),
                Path("docs/VERIFICATION_HEADROOM_PROTOCOL.md"), Path(cfg["polish_path"]),
                predictor, policy, partpath]
    root.mkdir(parents=True)
    cohort = root/"cohort.npz"
    masks = verification_masks(X.loc[ids], cfg["seed"])
    np.savez_compressed(cohort, ids=ids, clusters=groups.loc[ids].to_numpy(),
                        **{k: masks[k] for k in cfg["conditions"]})
    write_json(root/"protocol_freeze.json", dict(config=cfg, sources=hashes(sources),
        cohort=hashes([cohort]), predictor=str(predictor), policy=str(policy),
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(), hardware=platform.platform(),
        taiwan=dict(pool=len(pool), prior_assessment=len(outer), eligible=0, status="NOT RUN"),
        polish=dict(eligible_clusters=len(candidates), sampled_clusters=len(ids),
                    assessment_cluster_overlap=0, status="DEVELOPMENT REUSE"), Freddie="NOT RUN"))
    print(f"Frozen {len(ids)} Polish development clusters; Taiwan eligible=0; Freddie NOT RUN", flush=True)


def load(root):
    frozen = json.loads((root/"protocol_freeze.json").read_text())
    validate_hashes(frozen["sources"]); validate_hashes(frozen["cohort"])
    with np.load(root/"cohort.npz") as f:
        cohort = {k: f[k] for k in f.files}
    return frozen, frozen["config"], cohort


def evaluate(root):
    frozen, cfg, cohort = load(root)
    if (root/"complete.json").exists():
        raise FileExistsError("Audit complete; do not rerun outcomes")
    X, _, _ = load_polish(cfg["polish_path"])
    X = X.loc[cohort["ids"]]
    bundle = joblib.load(frozen["predictor"])
    adapter, donors = polish_adapter(bundle), bundle["donors"]
    if set(X.index) & set(donors.reference.index):
        raise RuntimeError("Development customers found in donor pool")
    write_json(root/"opened.json", dict(timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                                       freeze=hashes([root/"protocol_freeze.json"])))
    artifacts = []
    for cno, condition in enumerate(cfg["conditions"]):
        dest = root/condition
        if (dest/"complete.json").exists():
            r = json.loads((dest/"complete.json").read_text()); validate_hashes(r["artifacts"])
            artifacts += list(r["artifacts"])
            continue
        dest.mkdir(exist_ok=True)
        started = time.perf_counter()
        owner, field, values, remaining = actual_reveal_states(X.to_numpy(), cohort[condition])
        hidden = X.isna().to_numpy() | cohort[condition]
        chunks, probs, phi_fields, variances = [], [], [], []
        timings = dict(completion_seconds=0., attribution_seconds=0., scoring_seconds=0.,
                       attribution_calls=0, attributed_rows=0, natural_changes=0, observed_changes=0)
        for start in range(0, len(owner), cfg["batch_size"]):
            sl = slice(start, start+cfg["batch_size"])
            part = pd.DataFrame(values[sl], columns=POLISH_NAMES)
            stamp = time.perf_counter()
            draws, _ = donors.sample(part, remaining[sl], k=cfg["k"],
                         seed=cfg["seed"]+100000*cno+start+1)
            timings["completion_seconds"] += time.perf_counter()-stamp
            if not np.array_equal(np.where(remaining[sl,None,:], np.nan, draws),
                                  np.repeat(values[sl,None,:], cfg["k"], axis=1), equal_nan=True):
                raise RuntimeError("STOP: completions changed observed or naturally unknown values")
            stamp = time.perf_counter()
            before = adapter.attribute(part)
            flat = adapter.attribute(pd.DataFrame(draws.reshape(-1, len(POLISH_NAMES)), columns=POLISH_NAMES))
            if not before["valid"].all() or not flat["valid"].all():
                raise RuntimeError("STOP: invalid SHAP additivity; document before continuing")
            timings["attribution_seconds"] += time.perf_counter()-stamp
            timings["attribution_calls"] += 2
            timings["attributed_rows"] += len(part)*(cfg["k"]+1)
            stamp = time.perf_counter()
            cp = flat["phi"].reshape(len(part), cfg["k"], -1)
            a, h = aggregate_groups(before["phi"], hidden[owner[sl]], POLISH_NAMES, POLISH_GROUPS)
            ca, _ = aggregate_groups(cp, hidden[owner[sl]], POLISH_NAMES, POLISH_GROUPS)
            # Only donor methods are tested. No conditional-family claim is made.
            chunks.append(evidence(a, ca, ca, h, before["probability"]))
            probs.append(before["logits"]); phi_fields.append(before["phi"])
            variances.append(cp.var(1))
            timings["scoring_seconds"] += time.perf_counter()-stamp
            print(f"{condition}: states {min(start+len(part),len(owner))}/{len(owner)}", flush=True)
        ev = {k: np.concatenate([b[k] for b in chunks]) for k in chunks[0] if not k.startswith("conditional_")}
        ev["candidate"] &= ev["candidate"][:len(X)][owner]
        # Verification truth is serialized separately after all state evidence.
        np.savez_compressed(dest/"states.npz", **ev, owner=owner, field=field,
                            logits=np.concatenate(probs), feature_phi=np.concatenate(phi_fields),
                            feature_variance=np.concatenate(variances))
        full = adapter.attribute(X)
        if not full["valid"].all():
            raise RuntimeError("STOP: invalid verification attribution")
        restored, h = aggregate_groups(full["phi"], hidden, POLISH_NAMES, POLISH_GROUPS)
        np.savez_compressed(dest/"verification_only.npz", restored_phi=restored,
            meaningful=verified(restored,h,"meaningful"), ranked=verified(restored,h,"ranked"),
            logits=full["logits"], probability_shift=abs(expit(np.concatenate(probs)[:len(X)])-expit(full["logits"])))
        timings.update(total_seconds=time.perf_counter()-started, states=len(owner),
                       eligible_queries=int((field>=0).sum()), verification_attributed_rows=len(X),
                       predictor_refit=False, terminal_refit=False, device="cpu")
        write_json(dest/"runtime.json", timings)
        paths = [dest/x for x in ("states.npz", "verification_only.npz", "runtime.json")]
        write_json(dest/"complete.json",dict(artifacts=hashes(paths)))
        artifacts += paths
    write_json(root/"complete.json",dict(artifacts=hashes(artifacts), freeze=hashes([root/"protocol_freeze.json"])))


def npz(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def report(root):
    _, cfg, cohort = load(root)
    receipt = json.loads((root/"complete.json").read_text())
    validate_hashes(receipt["artifacts"]); validate_hashes(receipt["freeze"])
    policies = joblib.load(Path(cfg["policy_root"])/"polish/policies.joblib")["policies"]
    n = len(cohort["ids"])
    rng = np.random.default_rng(cfg["seed"]+900000)
    weights = rng.multinomial(n, np.full(n, 1/n), size=cfg["bootstrap_draws"])
    rows, paired, bounds, transitions, orders, solver_log = [], [], [], [], [], []
    out = root/"analysis"; out.mkdir(exist_ok=True)
    decisions = {}
    for condition in cfg["conditions"]:
        ev = npz(root/condition/"states.npz"); truth = npz(root/condition/"verification_only.npz")
        owner, field = ev["owner"], ev["field"]
        initial = ev["candidate"][:n]
        def record(label, selected, query, target, alpha, **meta):
            row, samples = paired_intervals(selected, truth[target], initial, query, weights)
            rows.append(dict(condition=condition,target=target,alpha=alpha,method=label,**meta,**row))
            decisions[(condition,target,alpha,label)] = (selected, query, row, samples)
            return row, samples
        for target in cfg["targets"]:
            survived = truth[target]
            for alpha in cfg["alphas"]:
                zero_ev = {k: v[:n] for k,v in ev.items()}
                zero = {"release_all":initial, "top1":strongest(zero_ev,1)}
                for method in ("rank","whole_mc1","whole_mc2","whole_mc_all","stable_donor"):
                    zero[method] = policies[(target,method,alpha,False)].release(zero_ev)
                for method, chosen in zero.items():
                    record("zero_"+method,chosen,np.zeros(n),target,alpha)
                for method in cfg["terminal_methods"]:
                    sets = policies[(target,method,alpha,False)].release(ev)
                    for objective in ("reasons","customers"):
                        for budget in (0,1):
                            action, log = optimize_oracle(sets,survived,owner,field,alpha=alpha,
                                                          budget=budget,objective=objective)
                            label=f"oracle_{method}_{objective}_b{budget}"
                            selected=selected_outputs(sets,action)
                            query=(action>=0)&(field[np.maximum(action,0)]>=0)
                            record(label,selected,query,target,alpha)
                            solver_log.append(dict(condition=condition,target=target,alpha=alpha,
                                                   method=label,**log))
                            np.savez_compressed(out/f"{condition}_{target}_{alpha}_{label}.npz",
                                                actions=action,selected=selected)
                            if objective=="reasons" and budget==1:
                                idx=np.where(action>=0,action,np.arange(n))
                                state_ev={k:v[idx] for k,v in ev.items()}
                                counts=selected.sum(1)
                                for control,matched in (("strength",strongest(state_ev,counts)),
                                    ("random",matched_random(state_ev,counts,cohort["ids"]))):
                                    if not np.array_equal(matched.sum(1),counts):
                                        raise RuntimeError("Matched-size control mismatch")
                                    record(label+"_matched_"+control,matched,query,target,alpha)
                        a=decisions[(condition,target,alpha,f"oracle_{method}_{objective}_b1")]
                        b=decisions[(condition,target,alpha,f"oracle_{method}_{objective}_b0")]
                        for metric in ("customer_ge1","reason_coverage","risk"):
                            delta=a[3][metric]-b[3][metric]
                            finite=delta[np.isfinite(delta)]
                            lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                            paired.append(dict(condition=condition,target=target,alpha=alpha,method=method,
                                objective=objective,metric=metric,difference=a[2][metric]-b[2][metric],lo=lo,hi=hi))
                feasible=[decisions[(condition,target,alpha,"zero_"+m)][2] for m in zero
                          if decisions[(condition,target,alpha,"zero_"+m)][2]["risk"]<=alpha]
                best=max((r["customer_ge1"] for r in feasible),default=0.)
                bounds.append(dict(condition=condition,target=target,alpha=alpha,
                    candidate_customer_ceiling=float(initial.any(1).mean()), best_feasible_zero_coverage=best,
                    maximum_possible_gain=float(initial.any(1).mean()-best),
                    feasible_zero_baselines=len(feasible)))
        # Pure evaluator diagnostics: actual reveals may retract or create claims.
        current_positive=(ev["phi"]>.01)&~ev["hidden"]
        new=(current_positive & ~initial[owner]).sum(1)
        retracted=(initial[owner]&~current_positive).sum(1)
        legitimate=(initial[owner]&~current_positive&~truth["meaningful"][owner]).sum(1)
        gain=(ev["candidate"]&truth["meaningful"][owner]).sum(1)
        importance=abs(ev["feature_phi"][:n]); variance=ev["feature_variance"][:n]
        for i in range(n):
            ix=np.flatnonzero((owner==i)&(field>=0)); base=i
            if not len(ix): continue
            signals={"realized_prediction_change":abs(expit(ev["logits"][ix])-expit(ev["logits"][base])),
                "own_shap_importance":importance[i,field[ix]],"own_shap_variance":variance[i,field[ix]],
                "rank_support_gain":((ev["donor_rank"][ix]-ev["donor_rank"][base])*initial[i]).sum(1)}
            utility=gain[ix]-gain[base]
            for name,score in signals.items():
                corr=spearmanr(score,utility).statistic if np.ptp(score)>0 and np.ptp(utility)>0 else np.nan
                chosen=int(np.argmax(score))
                orders.append(dict(condition=condition,record_id=int(cohort["ids"][i]),signal=name,
                                   correlation=corr,chosen_actual_valid_reason_gain=int(utility[chosen]),
                                   best_actual_valid_reason_gain=int(utility.max())))
            for cutoff in (.01,.02,.05):
                transitions.append(dict(condition=condition,record_id=int(cohort["ids"][i]),cutoff=cutoff,
                    stable_prediction=bool(truth["probability_shift"][i]<=cutoff),
                    any_initial_failure=bool((initial[i]&~truth["meaningful"][i]).any()),
                    any_query_retracts=bool((retracted[ix]>0).any()),
                    any_legitimate_retraction=bool((legitimate[ix]>0).any()),
                    any_new_group=bool((new[ix]>0).any()),
                    max_new_groups=int(new[ix].max())))
        print(f"Oracle report: {condition}",flush=True)
    for name, data in (("metrics",rows),("paired_query_gain",paired),("elementary_bounds",bounds),
                        ("transitions",transitions),("ordering",orders),("solver_receipts",solver_log)):
        pd.DataFrame(data).to_csv(out/f"{name}.csv",index=False)
    # Predeclared screen uses the primary endpoint only, never ranked substitution.
    potential=[]
    for b in bounds:
        if b["target"]!="meaningful" or b["maximum_possible_gain"]<cfg["minimum_customer_gain"]: continue
        for p in paired:
            if (p["condition"]==b["condition"] and p["target"]=="meaningful" and p["alpha"]==b["alpha"]
                and p["objective"]=="customers" and p["metric"]=="customer_ge1"
                and p["difference"]>=cfg["minimum_customer_gain"] and p["lo"]>0):
                actual=decisions[(p["condition"],p["target"],p["alpha"],f"oracle_{p['method']}_customers_b1")][2]
                if actual["customer_ge1"]-b["best_feasible_zero_coverage"]>=cfg["minimum_customer_gain"]:
                    potential.append(p)
    write_json(out/"decision.json",dict(decision="GO" if potential else "NO-GO",passing=potential,
        criterion="primary +5pp over feasible zero rule AND +5pp query oracle gain with conditional CI >0",
        Taiwan="NOT RUN: zero eligible development records after historical assessment exclusion",
        Freddie="NOT RUN: official files unavailable",oracle="evaluation-only optimistic finite-family bound"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,condition in zip(axes,cfg["conditions"]):
        f=frame[(frame.condition==condition)&(frame.target=="meaningful")&(frame.alpha==.1)]
        for method in ("zero_release_all","zero_top1","zero_rank","zero_stable_donor",
                       "oracle_rank_customers_b1","oracle_stable_donor_customers_b1"):
            r=f[f.method==method].iloc[0]
            ax.scatter(100*r.risk,100*r.customer_ge1,label=method)
        ax.set_title("Polish development / "+condition)
        ax.set_xlabel("Verified reason failure (%)")
        ax.axvline(10,color="grey",ls="--",lw=1)
    axes[0].set_ylabel("Customers with >=1 retained reason (%)")
    axes[1].legend(fontsize=7,loc="lower right")
    fig.suptitle("One-query hindsight bounds; not deployable policy results")
    fig.tight_layout();fig.savefig(out/"headroom.png",dpi=180);fig.savefig(out/"headroom.pdf");plt.close(fig)
    write_json(out/"report_receipt.json",dict(artifacts=hashes([out/f"{name}.csv" for name in
        ("metrics","paired_query_gain","elementary_bounds","transitions","ordering","solver_receipts")]
        +[out/"decision.json",out/"headroom.png",out/"headroom.pdf"]),
        bootstrap_draws=cfg["bootstrap_draws"],unit="original exact-feature cluster",
        oracle_intervals="conditional on selected oracle actions; not optimization-adjusted"))
    print("Decision:","GO" if potential else "NO-GO",flush=True)
