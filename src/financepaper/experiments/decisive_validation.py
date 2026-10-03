"""Frozen-predictor semantic, completion-budget and external validation stages."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import platform
import time
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from financepaper.data.schema import FEATURE_NAMES, CATEGORICAL_FEATURES
from financepaper.data.taiwan import load_taiwan
from financepaper.data.polish import (load_polish, download_polish, polish_partitions,
                                    verification_masks, restore_artificial, MEMBER_SHA)
from financepaper.experiments import revision_study as old
from financepaper.experiments import revision_assessment as assessment
from financepaper.experiments import revision_selection as selection
from financepaper.experiments import temporal_pilot as pilot
from financepaper.reliability.current import CurrentEvidence, current_features
from financepaper.reliability.validation import (TAIWAN_GROUPS, POLISH_GROUPS, POLISH_NAMES,
    ValidationDonors, aggregate_groups, revision_arrays, variants, observable_strata,
    assigned_environment, fit_validation_policy)
from financepaper.models.logistic import make_logistic
from financepaper.models.xgboost import make_xgboost
from financepaper.training.calibration import PositiveSlopePlattCalibrator

TW_ENV = ("mcar10", "mcar30", "mar30", "group_missing")
PL_ENV = ("mcar10", "mcar30")
POLICY_VARIANTS = ("feature3", "group1", "group2", "group3")
K_VALUES = (1, 2, 4, 8, 16)
write_json = pilot._write_json


def hashes(paths):
    return {str(p): sha256(Path(p).read_bytes()).hexdigest() for p in sorted(map(Path, paths))}


def validate_hashes(mapping):
    for file, expected in mapping.items():
        if sha256(Path(file).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen file changed: {file}")


def freeze(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol_freeze.json").exists():
        raise FileExistsError("validation protocol already frozen")
    paths = list(Path("src/financepaper").rglob("*.py")) + [Path(config_path)]
    paths += [Path("docs") / p for p in ("DOMAIN_REASON_GROUPS.md", "EXTERNAL_DATASET_PROTOCOL.md", "DECISIVE_VALIDATION_PROTOCOL.md")]
    paths += [Path("scripts/run_decisive_validation.py"), Path("uv.lock")]
    historical = Path(cfg["historical_root"])
    paths += [historical / f"fold_{f}" / p for f in range(3) for p in ("predictors.joblib", "release_model.joblib", "partition_ids.json")]
    write_json(root / "protocol_freeze.json", dict(config=cfg, sources=hashes(paths),
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(), external_opened=False,
        versions=pilot._package_versions(), hardware=platform.platform(), machine=platform.machine(),
        device="cpu", historical_commit="04bf054", external_sha256=MEMBER_SHA))
    print("Protocol/source/predictor freeze written; external assessment unopened.", flush=True)


class FlatAdapter:
    """Same interventional raw-logit SHAP game, with persistent explainer cache."""
    def __init__(self, model, encode, background, original_indices, calibrator):
        import shap
        self.model, self.encode = model, encode
        self.background, self.indices, self.calibrator = background, original_indices, calibrator
        self.linear = hasattr(model, "coef_")
        if self.linear:
            self.mean = np.asarray(background).mean(0, dtype=np.float64)
            self.expected = float(model.decision_function(self.mean[None])[0])
        else:
            self.explainer = shap.TreeExplainer(model,
                data=shap.maskers.Independent(background, max_samples=len(background)),
                model_output="raw", feature_perturbation="interventional")
            self.expected = float(self.explainer.expected_value)

    def predict_encoded(self, matrix):
        return (self.model.decision_function(matrix) if self.linear else
                self.model.predict(matrix, output_margin=True)).astype(float)

    def explain_encoded(self, matrix):
        encoded = ((matrix-self.mean)*self.model.coef_[0] if self.linear else
                   np.asarray(self.explainer.shap_values(matrix, check_additivity=True)))
        return np.stack([encoded[:, idx].sum(1) for idx in self.indices], 1), encoded.sum(1)

    def attribute(self, X):
        matrix = self.encode(X)
        logits = self.predict_encoded(matrix)
        phi, total = self.explain_encoded(matrix)
        residual = total-(logits-self.expected)
        valid = np.isfinite(residual) & (abs(residual) <= .002+.001*abs(logits-self.expected))
        return dict(phi=phi, logits=logits, probability=self.calibrator.transform(logits),
                    valid=valid, residual=residual)


def taiwan_adapter(bundle, model="xgb25"):
    prep = bundle["context"]["preprocessor"]
    origins = bundle["background"]["origins"]
    return FlatAdapter(bundle["fits"][model]["model"], lambda X: prep.transform(X).flatten(True)[0],
        bundle["background"]["flat"], [[i for i, f in enumerate(origins) if f == name] for name in FEATURE_NAMES],
        PositiveSlopePlattCalibrator.from_dict(bundle["selection"]["models"][model]["calibrator"]))


def collect(adapter, donors, X, artificial, *, seed, path):
    """Offline generator: serving evidence and verification-only files are separate."""
    path = Path(path)
    if (path / "complete.json").exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    artificial = np.asarray(artificial, bool)
    natural = X.isna().to_numpy()
    if np.any(natural & artificial):
        raise AssertionError("cannot verify a naturally unknown cell")
    partial = X.mask(artificial)
    before = adapter.attribute(partial)
    started = time.perf_counter()
    draws, donor_ids = donors.sample(partial, artificial, k=16, seed=seed)
    generation_seconds = time.perf_counter()-started
    completion_phi, completion_probability, completion_valid = [], [], []
    for d in range(16):
        frame = pd.DataFrame(draws[:, d], index=X.index, columns=X.columns)
        if not np.array_equal(frame.to_numpy()[~artificial], partial.to_numpy()[~artificial], equal_nan=True):
            raise AssertionError("completion changed observed or natural unknown values")
        a = before if not artificial.any() else adapter.attribute(frame)
        completion_phi.append(a["phi"])
        completion_probability.append(a["probability"])
        completion_valid.append(a["valid"])
    np.savez_compressed(path / "current.npz", record_ids=X.index.to_numpy(), hidden=natural|artificial,
        natural=natural, artificial=artificial, phi=before["phi"], logits=before["logits"],
        probability=before["probability"], valid=before["valid"], residual=before["residual"],
        completion_phi=np.stack(completion_phi,1), completion_probability=np.stack(completion_probability,1),
        completion_valid=np.stack(completion_valid,1), completions=draws, donor_ids=donor_ids,
        partial_values=partial.to_numpy())
    # Restoration occurs only after current evidence has been computed and saved.
    restored = restore_artificial(partial, X, artificial)
    full = adapter.attribute(restored)
    np.savez_compressed(path / "verification_only.npz", phi=full["phi"], logits=full["logits"],
        probability=full["probability"], valid=full["valid"], residual=full["residual"],
        truth=restored.to_numpy())
    write_json(path / "complete.json", dict(customers=len(X), seed=seed, generation_seconds=generation_seconds,
        current_only=hashes([path / "current.npz"]), verification_only=hashes([path / "verification_only.npz"])))


def read_pair(path):
    with np.load(Path(path) / "current.npz") as f:
        current = {k: f[k] for k in f.files}
    with np.load(Path(path) / "verification_only.npz") as f:
        verification = {k: f[k] for k in f.files}
    return current, verification


def external_current_features(c):
    p = np.clip(c["probability"], 1e-9, 1-1e-9)
    a, h = c["phi"], c["hidden"]
    observed = np.where(h, 0, a)
    ordered = np.sort(np.where(h, -np.inf, a), axis=1)[:, ::-1]
    frame = pd.DataFrame(dict(probability=p, entropy=-(p*np.log(p)+(1-p)*np.log1p(-p)),
        prediction_variance=c["completion_probability"][:, :8].var(1), missing_fraction=h.mean(1),
        positive_count=((a>.01)&~h).sum(1), attribution_abs_sum=abs(observed).sum(1),
        positive_sum=np.maximum(observed,0).sum(1), negative_sum=np.minimum(observed,0).sum(1),
        third_reason=np.nan_to_num(ordered[:,2], neginf=0),
        rank_gap=np.nan_to_num(ordered[:,2]-ordered[:,3], nan=0, posinf=0, neginf=0)))
    return pd.concat([frame, pd.DataFrame(h.astype(float), columns=[f"missing_{j}" for j in range(h.shape[1])])],axis=1)


def detector_scores(c, dataset, selectors):
    if dataset == "taiwan":
        features = current_features(CurrentEvidence(c["probability"], c["hidden"], c["phi"],
            c["completion_probability"][:, :8], c["completion_phi"][:, :8]))
    else:
        features = external_current_features(c)
    result = dict(entropy=features.entropy.to_numpy()/np.log(2),
        max_probability=1-np.maximum(features.probability.to_numpy(),1-features.probability.to_numpy()),
        prediction_variance=features.prediction_variance.to_numpy(), missing_fraction=features.missing_fraction.to_numpy())
    for name, item in selectors.items():
        result[name] = item["model"].predict_proba(features[item["columns"]])[:, 1]
    return result


def derive(path, dataset, selectors, *, sensitivity=True):
    c, v = read_pair(path)
    names, groups = (FEATURE_NAMES, TAIWAN_GROUPS) if dataset == "taiwan" else (POLISH_NAMES, POLISH_GROUPS)
    basic = detector_scores(c, dataset, selectors)
    tables = []
    for label, a, h, ca, k, mode in variants(c, names, groups, sensitivity=sensitivity):
        full = v["phi"] if mode is None else aggregate_groups(v["phi"], c["hidden"], names, groups,
                                                            observed_only=mode == "group")[0]
        rev = revision_arrays(a, full, h, k=k)
        mc = revision_arrays(a, ca, h, k=k)
        valid_current = c["valid"] & c["completion_valid"].all(1)
        eligible = rev["eligible"] & valid_current
        event = np.where(eligible & v["valid"], rev["event"].astype(float), np.nan)
        frame = pd.DataFrame(dict(record_id=c["record_ids"], variant=label, eligible=eligible, event=event,
            stratum=observable_strata(c["hidden"]), missing_fraction=c["hidden"].mean(1),
            natural_fraction=c["natural"].mean(1), artificial_fraction=c["artificial"].mean(1),
            probability=c["probability"], prob_shift=abs(expit(c["logits"])-expit(v["logits"])),
            calibrated_shift=abs(c["probability"]-v["probability"]),
            shift=(abs(a-full)*~h).sum(1)/np.maximum(((abs(a)+abs(full))*~h).sum(1), 1e-12),
            num_revised=rev["num_revised"], sign_event=rev["sign"]>0, rank_event=rev["rank"]>0))
        for name, value in basic.items():
            frame[f"score_{name}"] = value
        for budget in K_VALUES:
            frame[f"score_mc{budget}"] = mc["event"][:, :budget].mean(1)
        frame["score_rank_instability"] = mc["rank"][:, :8].mean(1)
        frame["score_sign_instability"] = mc["sign"][:, :8].mean(1)
        frame["score_attribution_variance"] = np.where(h, 0, ca[:, :8].var(1)).mean(1)
        if sensitivity:
            for eps in (0, .005, .01, .02):
                e = revision_arrays(a, full, h, k=k, rank_epsilon=eps)["event"]
                frame[f"event_gap_{eps:g}"] = np.where(eligible & v["valid"], e.astype(float), np.nan)
        tables.append(frame)
    return pd.concat(tables, ignore_index=True)


def fit_policies(tables, dataset, seed):
    """Only independent revision/release calibration tables are accepted here."""
    environments = TW_ENV if dataset == "taiwan" else PL_ENV
    calibration, release = tables["revision_calibration"], tables["release_calibration"]
    calibrators, policies = {}, {}
    for variant in POLICY_VARIANTS:
        fc, fr = calibration[calibration.variant == variant].copy(), release[release.variant == variant].copy()
        for frame, offset in ((fc, 50), (fr, 60)):
            assignment = assigned_environment(frame.record_id, environments, seed+offset)
            frame["assigned"] = frame.condition.eq(frame.record_id.map(assignment))
        good = fc.assigned & fc.eligible & fc.event.notna()
        for column in [c for c in fc if c.startswith("score_")]:
            method = column[6:]
            cal = assessment.calibrate_scores(fc.loc[good,column].to_numpy(), fc.loc[good,"event"].to_numpy(int))
            calibrators[(variant,method)] = cal
            fr["probability"] = cal.transform(logit(np.clip(fr[column].to_numpy(), 1e-7, 1-1e-7)))
            for alpha in (.05,.10,.15):
                # All K and detector controls pooled; main K8 additionally tests B/C.
                for family in (("pooled","stratified","robust") if method == "mc8" else ("pooled",)):
                    for conservative in (False,True):
                        policies[(variant,method,alpha,family,conservative)] = fit_validation_policy(fr,
                            family=family, risk_target=alpha, conservative=conservative, environments=environments)
    return dict(calibrators=calibrators, policies=policies)


def apply_policies(table, bundle):
    scores, releases = [], []
    for (variant,method), cal in bundle["calibrators"].items():
        f = table[table.variant == variant].copy()
        keep = ["record_id","condition","variant","eligible","event","stratum","missing_fraction",
                "natural_fraction","artificial_fraction","prob_shift","calibrated_shift","shift","probability"]
        f = f[keep].assign(method=method, raw_score=f[f"score_{method}"].to_numpy())
        f["calibrated_score"] = cal.transform(logit(np.clip(f.raw_score,1e-7,1-1e-7)))
        scores.append(f)
        for (var, met, alpha, family, conservative), policy in bundle["policies"].items():
            if (var,met) != (variant,method):
                continue
            rr = f[["record_id","condition","variant","eligible","event","stratum"]].copy()
            rr["released"] = policy.release(f.calibrated_score, f.eligible, f.stratum)
            releases.append(rr.assign(method=method, alpha=alpha, family=family, conservative=conservative))
    return pd.concat(scores,ignore_index=True), pd.concat(releases,ignore_index=True)


def run_taiwan(root):
    frozen = json.loads((root/"protocol_freeze.json").read_text()); validate_hashes(frozen["sources"])
    cfg = frozen["config"]; hist = Path(cfg["historical_root"])
    oldcfg, base = old.load_config("configs/revision_study.yaml")
    data = load_taiwan(Path(base["data_path"])); parts,_ = old.make_partitions(data,oldcfg)
    started = time.perf_counter()
    for fold in range(3):
        dest = root/"taiwan"/f"fold_{fold}"; dest.mkdir(parents=True,exist_ok=True)
        if (dest/"complete.json").exists():
            continue
        bundle = joblib.load(hist/f"fold_{fold}/predictors.joblib")
        release = joblib.load(hist/f"fold_{fold}/release_model.joblib")
        selectors = {"prediction_only":release["selectors"]["prediction_boosted"],
                     "learned_selector":release["selectors"][release["chosen"]]}
        adapter = taiwan_adapter(bundle)
        donor = ValidationDonors(bundle["context"]["X_train"], CATEGORICAL_FEATURES, seed=bundle["seed"])
        write_json(dest/"donor_reference.json",dict(ids=donor.reference.index.tolist(), train_ids=bundle["context"]["X_train"].index.tolist()))
        tables = {}
        for split,offset in (("revision_calibration",5000),("release_calibration",6000),("outer",7000)):
            # Outer is evaluated only after this fold's new policies are frozen.
            if split == "outer":
                fitted = fit_policies(tables,"taiwan",bundle["seed"])
                joblib.dump(fitted,dest/"policies.joblib")
                write_json(dest/"policy_freeze.json", dict(external_opened=False, source_freeze=sha256((root/"protocol_freeze.json").read_bytes()).hexdigest(), artifacts=hashes([dest/"policies.joblib"])))
            Xall = data.X.iloc[parts[fold][split]]
            masks = old.masks_for(Xall,bundle["context"]["mar"],bundle["seed"]+offset)
            rng = np.random.default_rng(bundle["seed"]+offset+20000)
            gh = np.zeros(Xall.shape,bool)
            for i, group in enumerate(rng.integers(3,size=len(Xall))):
                gh[i,[FEATURE_NAMES.index(f) for f in list(TAIWAN_GROUPS.values())[group]]] = True
            masks["group_missing"] = pd.DataFrame(gh,index=Xall.index,columns=Xall.columns)
            if split == "outer":
                ids = np.load(hist/f"fold_{fold}/outer/xgb25_complete.npz")["record_ids"]
                X = Xall.loc[ids]
            else:
                X = Xall
            frames = []
            for condition,h in masks.items():
                target = dest/split/condition
                collect(adapter,donor,X,h.loc[X.index].to_numpy(),seed=bundle["seed"]+offset+13,path=target)
                c,_ = read_pair(target)
                # Preservation check against historical K8/current event inputs.
                oldpath = hist/f"fold_{fold}"/("outer" if split=="outer" else split)/"current.csv"
                if condition != "group_missing":
                    historical = pd.read_csv(oldpath)
                    historical = historical[historical.condition==condition]
                    f = current_features(CurrentEvidence(c["probability"],c["hidden"],c["phi"],c["completion_probability"][:,:8],c["completion_phi"][:,:8]))
                    if not np.array_equal(historical.record_id,c["record_ids"]) or not np.allclose(historical.mc_revision,f.mc_revision,atol=1e-12):
                        raise RuntimeError("STOP: reference K8 changed; document before continuing")
                frames.append(derive(target,"taiwan",selectors).assign(condition=condition))
                print(f"Taiwan fold{fold}/{split}/{condition} cached",flush=True)
            tables[split] = pd.concat(frames,ignore_index=True)
            tables[split].to_csv(dest/f"{split}_derived.csv",index=False)
        scores, released = apply_policies(tables["outer"],fitted)
        scores.to_csv(dest/"scores.csv",index=False); released.to_csv(dest/"releases.csv",index=False)
        write_json(dest/"complete.json",dict(elapsed_seconds=time.perf_counter()-started, customers=len(X)))
    print("Taiwan validation completed with frozen predictors and original K8 streams.",flush=True)


class PolishPreprocessor:
    def fit(self,X):
        self.median = X.median().fillna(0).to_numpy()
        x = np.where(X.isna(), self.median, X.to_numpy())
        self.mean, self.scale = x.mean(0), x.std(0)
        self.scale = np.where(self.scale>0,self.scale,1.)
        self.train_ids = X.index.to_numpy()
        return self

    def transform(self,X):
        x = X.to_numpy(float); available = np.isfinite(x)
        filled = np.where(available,x,self.median)
        result = np.concatenate([(filled-self.mean)/self.scale,available.astype(float)],1)
        if not np.isfinite(result).all():
            raise ValueError("invalid external numeric transform")
        return result


def polish_adapter(bundle,name="xgb"):
    return FlatAdapter(bundle["models"][name],bundle["preprocessor"].transform,bundle["background"],
                       [[i] for i in range(64)],bundle["calibrators"][name])


def run_external_fit(root):
    frozen = json.loads((root/"protocol_freeze.json").read_text());validate_hashes(frozen["sources"])
    cfg = frozen["config"]; dest = root/"polish";dest.mkdir(exist_ok=True)
    if (dest/"assessment_freeze.json").exists():
        raise FileExistsError("external method already frozen")
    path = download_polish(cfg["external_path"]);X,y,clusters = load_polish(path)
    parts = polish_partitions(X,y,clusters,cfg["seed"]);seed=cfg["seed"]
    write_json(dest/"partition_ids.json",{s:X.index[ix].tolist() for s,ix in parts.items()})
    write_json(dest/"data_audit.json",dict(sha256=MEMBER_SHA,n=len(X),positive=int(y.sum()),features=64,
        natural_missing_cells=int(X.isna().sum().sum()),natural_missing_rows=int(X.isna().any(axis=1).sum()),
        duplicate_rows=int(X.duplicated().sum()),unique_feature_clusters=len(set(clusters)),
        missing_by_feature=X.isna().sum().to_dict(), partitions={s:dict(n=len(ix),positive=int(y.iloc[ix].sum()),
            complete_donor_candidates=int(X.iloc[ix].notna().all(axis=1).sum())) for s,ix in parts.items()}))
    train = X.iloc[parts["train"]];prep=PolishPreprocessor().fit(train)
    donors=ValidationDonors(train.dropna(),seed=seed)
    write_json(dest/"donor_reference.json",dict(ids=donors.reference.index.tolist(),train_ids=train.index.tolist(),
        complete_train_n=len(train.dropna()),selection="complete predictor-training records only"))
    matrices=[]
    for view in range(25):
        rng=np.random.default_rng(seed+700000+view)
        rates=rng.choice([0.,.1,.2,.3],len(train))
        h=(rng.random(train.shape)<rates[:,None]) & train.notna().to_numpy()
        matrices.append(prep.transform(train.mask(h)))
    matrix=np.concatenate(matrices);labels=np.tile(y.iloc[parts["train"]],25);weights=np.full(len(labels),1/25)
    models={};calibrators={};times={}
    calibration=X.iloc[parts["default_calibration"]]
    masks=verification_masks(calibration,seed+2000)
    for name,model in (("lr",make_logistic(seed)),("xgb",make_xgboost(seed))):
        started=time.perf_counter();model.fit(matrix,labels,sample_weight=weights);times[name]=time.perf_counter()-started
        models[name]=model
        margins=[]
        for h in masks.values():
            xx=prep.transform(calibration.mask(h))
            margins.append(model.decision_function(xx) if name=="lr" else model.predict(xx,output_margin=True))
        calibrators[name]=PositiveSlopePlattCalibrator().fit(np.concatenate(margins),np.tile(y.iloc[parts["default_calibration"]],3))
    bgpos=np.sort(np.random.default_rng(seed+300).choice(len(train),64,replace=False))
    bundle=dict(models=models,calibrators=calibrators,preprocessor=prep,background=prep.transform(train.iloc[bgpos]),
                background_ids=train.index[bgpos].tolist(),donors=donors,seed=seed)
    joblib.dump(bundle,dest/"predictors.joblib");write_json(dest/"fit_summary.json",dict(device="cpu",training_seconds=times,
        xgb_trees=200,xgb_depth=3,train_customers=len(train),training_views=25,total_weight=float(weights.sum())))
    adapter=polish_adapter(bundle);selectors={};tables={}
    for split,offset in (("selector_train",4000),("revision_calibration",5000),("release_calibration",6000)):
        part=X.iloc[parts[split]];frames=[];features=[];events=[]
        for condition,h in verification_masks(part,seed+offset).items():
            target=dest/split/condition
            collect(adapter,donors,part,h,seed=seed+offset+13,path=target)
            c,v=read_pair(target)
            if split=="selector_train":
                f=external_current_features(c);rr=revision_arrays(c["phi"],v["phi"],c["hidden"])
                good=rr["eligible"]&c["valid"]&v["valid"]
                if condition!="complete":
                    features.append(f.loc[good]);events.append(rr["event"][good])
            else:
                frames.append(derive(target,"polish",selectors).assign(condition=condition))
            print(f"Polish fit stage/{split}/{condition} cached",flush=True)
        if split=="selector_train":
            f=pd.concat(features,ignore_index=True);target=np.concatenate(events).astype(int)
            for name,columns in (("prediction_only",list(f.columns[:4])),("learned_selector",list(f.columns))):
                model=selection.make_selector("boosted",seed);model.fit(f[columns],target)
                selectors[name]=dict(model=model,columns=columns)
            joblib.dump(selectors,dest/"selectors.joblib")
        else:
            tables[split]=pd.concat(frames,ignore_index=True)
            tables[split].to_csv(dest/f"{split}_derived.csv",index=False)
    policy=fit_policies(tables,"polish",seed);joblib.dump(policy,dest/"policies.joblib")
    files=[dest/p for p in ("predictors.joblib","selectors.joblib","policies.joblib","partition_ids.json","data_audit.json")]
    write_json(dest/"assessment_freeze.json",dict(outer_opened=False,timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        artifacts=hashes(files),source_freeze_sha256=sha256((root/"protocol_freeze.json").read_bytes()).hexdigest()))
    print("External predictor, completion, all calibrators/policies frozen before assessment.",flush=True)


def run_external_evaluate(root):
    frozen=json.loads((root/"protocol_freeze.json").read_text());validate_hashes(frozen["sources"])
    dest=root/"polish";gate=json.loads((dest/"assessment_freeze.json").read_text());validate_hashes(gate["artifacts"])
    if (dest/"assessment_complete.json").exists():
        raise FileExistsError("external evaluation already opened/completed")
    if gate["source_freeze_sha256"]!=sha256((root/"protocol_freeze.json").read_bytes()).hexdigest():
        raise RuntimeError("external protocol freeze changed")
    X,y,clusters=load_polish(frozen["config"]["external_path"])
    ids=json.loads((dest/"partition_ids.json").read_text())["outer"];X=X.loc[ids];y=y.loc[ids]
    bundle=joblib.load(dest/"predictors.joblib");selectors=joblib.load(dest/"selectors.joblib");policy=joblib.load(dest/"policies.joblib")
    masks=verification_masks(X,bundle["seed"]+7000);adapter=polish_adapter(bundle)
    frames=[];predictions=[]
    # Marker precedes the first assessment model call, even if a run is interrupted.
    write_json(dest/"assessment_opened.json",dict(timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),freeze=hashes([dest/"assessment_freeze.json"])))
    for condition,h in masks.items():
        target=dest/"outer"/condition
        collect(adapter,bundle["donors"],X,h,seed=bundle["seed"]+7013,path=target)
        frames.append(derive(target,"polish",selectors).assign(condition=condition))
        for model in ("xgb","lr"):
            ad=adapter if model=="xgb" else polish_adapter(bundle,"lr")
            a=ad.attribute(X.mask(h));full=ad.attribute(X)
            predictions.append(pd.DataFrame(dict(record_id=X.index,condition=condition,model=model,y=y,
                prob_raw=expit(a["logits"]),prob_calibrated=a["probability"],prob_full=full["probability"])))
            if model=="lr":
                np.savez_compressed(dest/"outer"/f"lr_{condition}.npz",record_ids=X.index.to_numpy(),hidden=X.isna().to_numpy()|h,
                    original_before=a["phi"],original_full=full["phi"],valid_before=a["valid"],valid_full=full["valid"],
                    logits_before=a["logits"],logits_full=full["logits"])
        print(f"External confirmation/{condition} completed; no adaptation permitted.",flush=True)
    table=pd.concat(frames,ignore_index=True);table.to_csv(dest/"outer_derived.csv",index=False)
    scores,releases=apply_policies(table,policy)
    scores.to_csv(dest/"scores.csv",index=False);releases.to_csv(dest/"releases.csv",index=False)
    pd.concat(predictions).to_csv(dest/"predictions.csv",index=False)
    write_json(dest/"assessment_complete.json",dict(customers=len(X),no_method_adaptation=True,
        timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),artifacts=hashes([dest/"outer_derived.csv",dest/"scores.csv",dest/"releases.csv",dest/"predictions.csv"])))
