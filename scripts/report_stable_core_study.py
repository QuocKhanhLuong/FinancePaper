"""Assessment-only summaries, paired cluster intervals and six figure families."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from financepaper.experiments.stable_core_study import folder,spec,load_npz
from financepaper.experiments.decisive_validation import hashes,validate_hashes,write_json


def bootstrap_weights(meta,draws=1000):
    ids,inv=np.unique(meta.cluster.to_numpy(),return_inverse=True)
    folds=meta.groupby("cluster").fold.first().reindex(ids).to_numpy()
    w=np.zeros((draws,len(ids)),float);rng=np.random.default_rng(20261005)
    for fold in np.unique(folds):
        pos=np.flatnonzero(folds==fold)
        for b in range(draws):
            w[b,pos]=np.bincount(rng.integers(len(pos),size=len(pos)),minlength=len(pos))
    return w,inv


def ratio(n,d,w,inv):
    n,d=np.asarray(n,float),np.asarray(d,float)
    cn=np.bincount(inv,weights=n,minlength=w.shape[1]);cd=np.bincount(inv,weights=d,minlength=w.shape[1])
    den=w@cd;num=w@cn
    samples=np.divide(num,den,out=np.full(len(w),np.nan),where=den>0)
    finite=samples[np.isfinite(samples)]
    lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
    return (float(n.sum()/d.sum()) if d.sum() else np.nan,float(lo),float(hi)),samples


def load_dataset(root,dataset):
    meta=[];evs=[];truths=[];decs=[]
    for fold in (range(3) if dataset=="taiwan" else [0]):
        for condition in spec(dataset)[2]:
            p=folder(root,dataset,fold)/"outer"/condition
            ev=load_npz(p/"reason_evidence.npz");truth=load_npz(p/"verification_only.npz")
            evs.append(ev);truths.append(truth);decs.append(load_npz(p/"decisions.npz"))
            meta.append(pd.DataFrame(dict(record_id=ev["record_ids"],cluster=ev["cluster_ids"],fold=fold,condition=condition)))
    cat=lambda blocks:{k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    return pd.concat(meta,ignore_index=True),cat(evs),cat(truths),cat(decs)


def summary(root):
    receipt=json.loads((root/"assessment_complete.json").read_text());validate_hashes(receipt["artifacts"])
    out=root/"analysis";out.mkdir(exist_ok=True)
    rows=[];paired=[];groups=[];counts=[];decoupling=[];support=[];curves=[]
    for dataset in ("taiwan","polish"):
        meta,ev,truth,decisions=load_dataset(root,dataset)
        w,inv=bootstrap_weights(meta);candidate=ev["candidate"].sum(1)
        conds=["overall",*spec(dataset)[2]]
        samples={}
        for key,selected in decisions.items():
            target,method,alpha,conservative=key.split("|");survives=truth[target]
            n=selected.sum(1);f=(selected&~survives).sum(1);success=n-f
            candidate_success=(ev["candidate"]&survives).sum(1)
            for condition in conds:
                block=np.ones(len(meta),bool) if condition=="overall" else meta.condition.to_numpy()==condition
                metric_inputs={"reason_risk":(f,n),"reason_coverage":(n,candidate),
                    "customer_ge1":(n>0,np.ones(len(n))),"customer_ge2":(n>=2,np.ones(len(n))),
                    "mean_reasons":(n,np.ones(len(n))),"any_failure_risk":(f>0,n>0),
                    "candidate_survivor_recall":(success,candidate_success)}
                row=dict(dataset=dataset,target=target,method=method,alpha=float(alpha),
                         conservative=int(conservative),condition=condition,n_records=int(block.sum()),
                         candidate_reasons=int(candidate[block].sum()),released_reasons=int(n[block].sum()),
                         failed_reasons=int(f[block].sum()))
                for metric,(num,den) in metric_inputs.items():
                    result,samp=ratio(np.asarray(num)*block,np.asarray(den)*block,w,inv)
                    row[metric],row[metric+"_lo"],row[metric+"_hi"]=result
                    if condition=="overall" and metric in ("reason_risk","customer_ge1","reason_coverage"):
                        samples[(key,metric)]=samp
                row["budget_attained"]=bool(np.isfinite(row["reason_risk"]) and row["reason_risk"]<=float(alpha))
                rows.append(row)
            if float(alpha)==.1 and conservative=="0":
                for size in range(ev["candidate"].shape[1]+1):
                    counts.append(dict(dataset=dataset,target=target,method=method,size=size,count=int((n==size).sum()),fraction=float((n==size).mean())))
                for j,name in enumerate(spec(dataset)[1]):
                    denom=int(selected[:,j].sum());fail=int((selected[:,j]&~survives[:,j]).sum())
                    groups.append(dict(dataset=dataset,target=target,method=method,group=name,released=denom,failed=fail,
                                       risk=fail/denom if denom else np.nan))
        indexed=pd.DataFrame([r for r in rows if r["dataset"]==dataset and r["condition"]=="overall"])
        for target in ("meaningful","ranked"):
            for alpha in (.05,.1,.15):
                for method in ("stable_donor","stable_conditional","stable_both"):
                    k=f"{target}|{method}|{alpha}|0"
                    for baseline in ("whole_legacy2","whole_mc1","whole_mc2","whole_mc_all","rank","frequency","strength",
                                     method+"_matched_strength",method+"_matched_random"):
                        bk=f"{target}|{baseline}|{alpha}|0"
                        for metric in ("reason_risk","customer_ge1","reason_coverage"):
                            delta=samples[k,metric]-samples[bk,metric]
                            valid=delta[np.isfinite(delta)];lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
                            a=indexed[(indexed.target==target)&(indexed.method==method)&(indexed.alpha==alpha)&(indexed.conservative==0)].iloc[0]
                            b=indexed[(indexed.target==target)&(indexed.method==baseline)&(indexed.alpha==alpha)&(indexed.conservative==0)].iloc[0]
                            paired.append(dict(dataset=dataset,target=target,alpha=alpha,method=method,baseline=baseline,metric=metric,
                                difference=a[metric]-b[metric],lo=lo,hi=hi))
            # Do not confuse rate among candidate reasons with any-reason revision.
            for condition in spec(dataset)[2]:
                block=meta.condition.to_numpy()==condition
                eligible=ev["candidate"].any(1)&block
                failed=(ev["candidate"]&~truth[target]).any(1)
                for cutoff in (.01,.02,.05):
                    stable=truth["probability_shift"]<=cutoff
                    A=int((eligible&stable&~failed).sum());B=int((eligible&stable&failed).sum())
                    C=int((eligible&~stable&~failed).sum());D=int((eligible&~stable&failed).sum())
                    decoupling.append(dict(dataset=dataset,target=target,condition=condition,cutoff=cutoff,A=A,B=B,C=C,D=D,
                                          revision_among_stable=B/(A+B) if A+B else np.nan))
        # Completion-model disagreement is evaluated at the same frozen operating point.
        for alpha in (.05,.1,.15):
            a=decisions[f"meaningful|stable_donor|{alpha}|0"]
            b=decisions[f"meaningful|stable_conditional|{alpha}|0"]
            for name,mask in (("both",a&b),("donor_only",a&~b),("conditional_only",b&~a)):
                n=int(mask.sum());fail=int((mask&~truth["meaningful"]).sum())
                support.append(dict(dataset=dataset,alpha=alpha,agreement=name,reasons=n,failures=fail,risk=fail/n if n else np.nan))
    frames={"metrics":rows,"paired_differences":paired,"group_failure":groups,"reason_counts":counts,
            "decoupling":decoupling,"family_disagreement":support}
    for name,records in frames.items():pd.DataFrame(records).to_csv(out/(name+".csv"),index=False)
    figures(out)
    write_json(out/"report_receipt.json",dict(bootstrap_draws=1000,unit="customer; Polish duplicate-feature cluster",
        scope="fixed policies; exploratory previously inspected datasets",sources=hashes([Path(__file__)]),
        outputs=hashes(sorted(out.glob("*.csv"))+sorted(out.glob("*.png"))+sorted(out.glob("*.pdf")))))
    print("Paired cluster summaries and plots A–F written.",flush=True)


def figures(out):
    table=pd.read_csv(out/"metrics.csv")
    selected=table[(table.target=="meaningful")&(table.conservative==0)&(table.condition=="overall")]
    methods=["whole_legacy2","whole_mc1","rank","variance","frequency","strength","stable_donor","stable_conditional","stable_both"]
    plt.rcParams.update({"font.size":9,"figure.dpi":150})
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+".png"),bbox_inches="tight");fig.savefig(out/(name+".pdf"),bbox_inches="tight");plt.close(fig)
    for name,metric,title in (("A_reason_risk_coverage","reason_coverage","Candidate-reason coverage"),
                              ("B_customer_risk_coverage","customer_ge1","Customers receiving >=1 reason")):
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,dataset in zip(axes,("taiwan","polish")):
            for method in methods:
                f=selected[(selected.dataset==dataset)&(selected.method==method)].sort_values("alpha")
                ax.plot(f.reason_risk*100,f[metric]*100,"o-",label=method,markersize=3)
            ax.set(title=dataset,xlabel="Verified reason failure (%)",ylabel=title+" (%)",ylim=(-2,102))
            ax.axvline(10,color="grey",ls=":",lw=.8);ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6,loc="best")
        fig.suptitle("Frozen 5/10/15% calibration operating points; empirical policies")
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    chosen=["whole_legacy2","whole_mc1","whole_mc_all","rank","strength","stable_donor","stable_both"]
    for ax,dataset in zip(axes,("taiwan","polish")):
        f=selected[(selected.dataset==dataset)&(selected.alpha==.1)].set_index("method").loc[chosen]
        ax.bar(np.arange(len(f))-.18,f.customer_ge1*100,width=.36,label=">=1 reason")
        ax.bar(np.arange(len(f))+.18,f.customer_ge2*100,width=.36,label=">=2 reasons")
        ax.set_xticks(range(len(f)),f.index,rotation=40,ha="right");ax.set(title=dataset,ylabel="Customer coverage (%)",ylim=(0,105));ax.legend(loc="lower left")
        for i,r in enumerate(f.reason_risk):ax.text(i,101,f"risk {r:.1%}" if np.isfinite(r) else "empty",ha="center",fontsize=6)
    save(fig,"C_whole_vs_partial")
    distribution=pd.read_csv(out/"reason_counts.csv")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,dataset in zip(axes,("taiwan","polish")):
        for method in ("whole_legacy2","rank","stable_donor","stable_conditional","stable_both"):
            f=distribution[(distribution.dataset==dataset)&(distribution.target=="meaningful")&(distribution.method==method)]
            ax.plot(f["size"],f.fraction*100,"o-",label=method)
        ax.set(title=dataset,xlabel="Number of released reasons",ylabel="Customer-condition records (%)");ax.legend(fontsize=7)
    save(fig,"D_reason_set_sizes")
    group=pd.read_csv(out/"group_failure.csv")
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,dataset in zip(axes,("taiwan","polish")):
        for method in ("release_all","rank","stable_donor","stable_both"):
            f=group[(group.dataset==dataset)&(group.target=="meaningful")&(group.method==method)]
            ax.plot(np.arange(len(f)),f.risk*100,"o-",label=method)
        ax.set_xticks(range(len(f)),f.group,rotation=40,ha="right");ax.set(title=dataset,ylabel="Verified reason failure (%)");ax.legend(fontsize=7)
    save(fig,"E_failure_by_group")
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for col,dataset in enumerate(("taiwan","polish")):
        for method in ("stable_donor","stable_conditional","stable_both"):
            f=selected[(selected.dataset==dataset)&(selected.method==method)].sort_values("alpha")
            axes[0,col].plot(f.alpha*100,f.customer_ge1*100,"o-",label=method)
            axes[1,col].plot(f.alpha*100,f.reason_risk*100,"o-",label=method)
        axes[0,col].set(title=dataset,ylabel="Customer coverage (%)")
        axes[1,col].set(xlabel="Declared calibration budget (%)",ylabel="Observed reason failure (%)")
        axes[1,col].plot([5,10,15],[5,10,15],"k:",label="budget")
        axes[0,col].legend(fontsize=7)
    save(fig,"F_completion_families")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="outputs/stable_core_study")
    summary(Path(parser.parse_args().output))
