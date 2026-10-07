"""Audit an existing synthetic bundle and build an offline, single-slot preview."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

import pandas as pd
from tqdm.auto import tqdm

from run_decision_value_pilot import log, stage, utc
from financepaper.evaluation.decision_value import check_hashes, digest, write_json
from financepaper.evaluation.decision_review import audit_fixture, build_preview


def table(frame):
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def write_report(output):
    audit = json.loads((output / "audit.json").read_text())
    baseline = pd.read_csv(output / "rule_baselines.csv")
    lengths = pd.read_csv(output / "visible_lengths.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    report = f"""# F1 review kit — actual software audit, no expert or human evidence

Run `{output.name}`, created `{manifest['created_utc']}`, CPU / Python
`{manifest['python']}`. Base `{manifest['base_git_head']}`; branch
`{manifest['branch']}`. This is a new preparation stage, not a rerun of F0.

## VERIFIED / REPRODUCED

- Verified the original vignette stage hashes. Existing cases, labels, assignment
  and F0 outputs were read-only; no new predictor or policy was fitted.
- Independently enumerated all 24 feature orders for each of
  {audit['shapley_comparisons']} current/full vector comparisons across
  {audit['pool_cases_checked']} synthetic pool cases. Maximum absolute difference
  from the closed-form observed-group contributions: {audit['max_abs_shapley_error']:.3g}.
- Enumerated feasible H values independently of `independent_answer`: all
  {audit['pool_cases_checked']} pool labels agree with the stated entailment rule.
- Checked {audit['selected_cases_checked']} selected cases and
  {audit['public_renderings_checked']} public renderings against the frozen
  current-only renderer and allocation. No repeated case per slot.
- Exported one self-contained HTML preview from `slot_000.jsonl` only, with
  strict field validation and no private-label input to the exporter. No answers
  or human timings are collected. Browser/test QA is recorded separately in
  `validation_receipt.json`; creating HTML alone is not browser validation.
- Stage logs, actual elapsed time, tqdm/ETA, seeds/hashes and resume receipts
  remain in the ignored run directory. The original bundle remains unchanged.

## Deterministic baseline result — not simulated human performance

{table(baseline)}

The current-record rule uses only A and H, already present in arm 1. Thus the
answer is entirely recoverable without a score, explanation or completion
distribution. The selected fixture has 7 supported / 9 review cases; this
balanced-by-computational-stratum sample is not population prevalence.

**Interpretation:** this fixture can test presentation/comprehension effects.
It cannot demonstrate new factual information supplied by uncertainty disclosure.
Human difficulty, ceiling effects and actual benefit remain unknown. Do not tune
the task to make the more complicated arm win after observing participant data.

## Visible information length

Variable visible text, whitespace-delimited words, excludes JSON schema and
shared page chrome; this differs deliberately from the earlier JSON-length proxy.

{table(lengths)}

Arm 4 remains longer than arm 3. The fixed page layout does not prove equal
reading effort. No filler, new arm or revised case labels were introduced.

## REPORTED, PROPOSED and NOT RUN

F0 utility, historical model metrics and runtimes are **REPORTED from the prior
stage**, not recomputed here. The F1 answer is mathematically checkable, but its
professional relevance and the rubric have not been approved by an expert.

The new [review guide](https://github.com/QuocKhanhLuong/FinancePaper/blob/{manifest['branch']}/docs/DECISION_VALUE_REVIEW_GUIDE.md)
contains concrete relevance/rubric/effort questions. The reviewer preview stays
local at `{output}/expert_preview/index.html`; it was not sent to anyone.

**HUMAN STUDY, expert approval, ethics/consent, power simulation, final SESOI and
sample-size justification: NOT RUN / PENDING.** No AI ratings replace people.
The fixture remains **UI/software preparation only** until its relevance gate
is resolved. No algorithm novelty, financial return or real-world utility claim.

**One next action:** supervisor/domain expert reviews the task and proposed
rubric using the local preview and guide, and decides whether to retain it for
a separately approved usability pilot.
"""
    (output / "RESULTS.md").write_text(report)
    return [output / "RESULTS.md"]


def run(bundle_run, output, *, resume=False, stop_after=None, publish=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Review runs must stay in ignored runs/decision_value_pilot")
    if output.resolve() == bundle_run.resolve() or output.resolve().is_relative_to(bundle_run.resolve()):
        raise ValueError("Do not write into the historical input run")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((bundle_run / "manifest.json").read_text())
    if original["status"] != "COMPLETE":
        raise ValueError("Original run is incomplete")
    check_hashes(original["frozen"]["sources"])
    receipt = json.loads((bundle_run / "vignettes_complete.json").read_text())
    check_hashes(receipt["outputs"])
    bundle = bundle_run / "case_bundle"
    seed = original["frozen"]["config"]["seed"]
    sources = [Path(__file__), Path("scripts/run_decision_value_pilot.py"),
               Path("src/financepaper/evaluation/decision_review.py"), Path("src/financepaper/evaluation/decision_vignettes.py"),
               Path("src/financepaper/evaluation/decision_value.py"), Path("templates/decision_review.html"),
               Path("docs/DECISION_VALUE_REVIEW_GUIDE.md"), Path("uv.lock")]
    inputs = {**receipt["outputs"], str(bundle_run / "manifest.json"): digest(bundle_run / "manifest.json"),
              str(bundle_run / "vignettes_complete.json"): digest(bundle_run / "vignettes_complete.json")}
    frozen = dict(inputs=inputs, sources={str(p): digest(p) for p in sources}, seed=seed, slot="slot_000")
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if not resume or manifest["frozen"] != frozen:
            raise ValueError("Resume rejected: require --resume and identical inputs/code")
    else:
        manifest = dict(created_utc=utc(), frozen=frozen, device="cpu", python=platform.python_version(),
                        base_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                        status="RUNNING", human_study="NOT_RUN")
        write_json(path, manifest)
    for name in tqdm(("audit", "preview", "report"), desc="Review preparation", unit="stage"):
        if name == "audit":
            work = lambda: audit_fixture(bundle, seed, output)
        elif name == "preview":
            def work():
                destination = output / "expert_preview/index.html"
                destination.parent.mkdir(exist_ok=True)
                destination.write_text(build_preview(bundle / "expert/slot_000.jsonl", Path("templates/decision_review.html")))
                shutil.copy2("docs/DECISION_VALUE_REVIEW_GUIDE.md", output / "REVIEW_GUIDE.md")
                return [destination, output / "REVIEW_GUIDE.md"]
        else:
            work = lambda: write_report(output)
        stage(output, name, work, resume=resume)
        if stop_after == name:
            log(output, "run", "intentional_pause", completed_stage=name)
            return
    check_hashes(inputs)
    manifest.update(status="COMPLETE", completed_utc=utc())
    write_json(path, manifest)
    names = ["RESULTS.md", "audit.json", "rule_baselines.csv", "stratum_counts.csv", "visible_lengths.csv"]
    public = dict(run=str(output), source_run=str(bundle_run), source_run_sha256=digest(bundle_run / "manifest.json"),
                  sources=frozen["sources"], inputs_verified=len(inputs), manifest_sha256=digest(path),
                  aggregates={name: digest(output / name) for name in names}, human_study="NOT_RUN")
    write_json(output / "publication_receipt.json", public)
    if publish:
        publish.mkdir(parents=True, exist_ok=False)
        for name in [*names, "publication_receipt.json"]:
            shutil.copy2(output / name, publish / name)
    log(output, "run", "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=["audit", "preview", "report"])
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    run(args.bundle_run, args.output, resume=args.resume, stop_after=args.stop_after, publish=args.publish)
