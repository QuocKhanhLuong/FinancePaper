"""Write one aggregate report and optionally publish an explicit safe allowlist."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pandas as pd

from financepaper.evaluation.decision_value import check_hashes, digest, write_json


def markdown(frame):
    columns = list(frame.columns)
    def value(x):
        return f"{x:.6f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"] +
                     ["| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def report(run, publish=None):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise ValueError("Incomplete run")
    if manifest["frozen"]["config"] != json.loads(Path("configs/decision_value_pilot.json").read_text()):
        raise ValueError("This report template requires the declared pilot configuration")
    check_hashes(manifest["frozen"]["sources"])
    check_hashes(manifest["frozen"]["inputs"])
    for receipt in manifest["stage_receipts"].values():
        check_hashes(receipt["outputs"])
    counts = pd.read_csv(run / "utility_counts.csv")
    utility = pd.read_csv(run / "utility_grid.csv")
    crossings = pd.read_csv(run / "utility_break_even.csv")
    overall = counts[counts.condition == "overall"]
    winners = utility[(utility.condition == "overall") & utility.grid_winner].groupby(["dataset", "c_over_b"]).agg(
        winners=("method", lambda xs: ", ".join(xs)), utility_per_case=("utility_per_case", "first")).reset_index()
    contrasts = utility[(utility.condition == "overall") & (utility.method == "stable_both") & utility.c_over_b.isin([1, 2, 10])]
    times = pd.DataFrame([dict(stage=k, actual_elapsed_seconds=v["elapsed_seconds"]) for k, v in manifest["stage_receipts"].items()])
    suite = ET.parse(run / "tests_final.xml").getroot()
    totals = {k: sum(int(s.get(k, 0)) for s in suite.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    tests = json.loads((run / "tests_final_receipt.json").read_text())
    bundle = json.loads((run / "case_bundle/bundle_receipt.json").read_text())
    lengths = pd.read_csv(run / "case_bundle/presentation_lengths.csv")
    reconciliations = pd.concat([pd.read_csv(run / f"utility_{d}_reconciliation.csv") for d in ("taiwan", "polish")])
    if not reconciliations["match"].all():
        raise ValueError("Counts disagree")
    baseline_test = Path("runs/decision_value_pilot/20261007T094500Z/tests_baseline_receipt.json")
    text = f"""# Decision-value pilot — kết quả tổng hợp

Run `{run.name}`; actual UTC start `{manifest['created_utc']}`. Run IDs are labels;
the manifest/progress timestamps are authoritative. Base main `{manifest['base_git_head']}`;
branch `{manifest['branch']}`. **F0 đã chạy; F1/F2 đã chuẩn bị; HUMAN STUDY NOT RUN.**
Đây là exploratory reuse trên Taiwan/Polish đã xem, không phải external confirmation.

## VERIFIED / REPRODUCED — thực sự chạy

- Đồng bộ lockfile; CPU `{manifest['platform']}`, Python `{manifest['python']}`.
- Kiểm tra SHA256 của {len(manifest['frozen']['inputs'])} input/receipt files, trước và sau chạy;
  replay frozen release masks trước khi đọc verification labels; kiểm tra lại meaningful target.
- {len(reconciliations)} count comparisons khớp historical metrics. Không thiếu artifact cần dùng;
  không cần recovery worktree hoặc refit/recompute SHAP. Historical results/configs giữ nguyên.
- Audit 2,400 Taiwan customers / 9,600 customer-condition cases (4 conditions), và
  1,182 Polish statements / 2,364 cases / 1,169 exact-feature clusters (2 conditions).
  Repeated masks không được tính là khách hàng độc lập; không xác nhận company identity.
- Utility grid cố định `[0,.5,1,2,5,10]`; 1,000 paired cluster bootstrap draws,
  seed 20261007, Taiwan stratified by historical fold. Tất cả 1,000 draws có denominator.
- Partial-run resume đã thực chạy: dừng sau Taiwan, xác minh hash và bỏ qua Taiwan
  khi tiếp tục Polish/frontier/vignettes. `progress.jsonl`, tqdm/ETA, stage receipts,
  source/input hashes và logs nằm local trong `{run}`.
- Suite cuối: **{totals['passed']} passed, {totals['failures']} failed, {totals['skipped']} skipped**,
  elapsed wall time {tests['elapsed_seconds']:.3f}s. Skip: MPS hardware unavailable
  trong môi trường test; actual research device CPU. Ba SHAP deprecation warnings.
  Suite ban đầu thật chạy: 119 passed / 1 failed / 1 skipped; lỗi hash provenance
  tìm hai docs đã chuyển archive. Chỉ sửa hai đường dẫn trong `source_hashes`, không
  đổi nội dung archive, config hay cơ chế fitting. Tests có synthetic model fits;
  không có training predictor nghiên cứu mới.

Actual audit stage timing (không phải online policy latency; có chạy đồng thời suite):

{markdown(times)}

Initial plot render built a font cache. A development run first exposed non-writable
global plotting caches; final runner stores them under the ignored run directory.
A runtime-scope metadata correction was applied before this final run; the utility
grid is byte-identical to the development run. No policy, threshold or grid changed.

## Counts và baseline mạnh

Target duy nhất của bảng dưới: meaningful-positive failure **theo từng reason**.
Units của valid/failed/released/candidate là reasons; coverage denominator là mọi
customer-condition case, kể cả case không có candidate. Full condition tables và
denominators ở [utility_counts.csv](utility_counts.csv).

{markdown(overall[['dataset','method','valid_reasons','failed_reasons','released_reasons','candidate_reasons','coverage_ge1','coverage_ge2']])}

**Release-all là baseline mạnh.** Mask warning, whole-MC-all và rank release rule
đồng mask với release-all tại operating point này; top-1 đồng mask với whole-MC1.
Không được diễn giải utility bằng nhau là tác động người dùng bằng nhau. Các phép
so sánh rank ở đây không phải chạy lại AP của historical rank-instability detector.
Top-1 có failure thấp nhưng bỏ nhiều valid reasons, không thắng trên grid đã đặt.
Taiwan conditional-only bị donor dominated trong (valid,failed) ở mixture này.

## Utility sensitivity, frontier và CI

`U = valid - (c/b)*failed`, b=1. Compute/delay costs **UNKNOWN, NOT SUBTRACTED**:
đây là gross hypothetical utility theo valid-reason benefit units, không phải tiền,
measured welfare, individual guarantee hay tối ưu policy trên test.

{markdown(winners)}

Winners là mô tả empirical maximizers giữa policies cố định, không inference rằng
mỗi winner khác tất cả competitors có ý nghĩa thống kê. Condition-level grid có
khác biệt: Taiwan group_missing vẫn donor cao nhất tại c/b=10; Polish MCAR10 đã
both-family cao nhất tại c/b=5. Không chọn một winner mới làm model triển khai.

Paired difference của both-family so với release-all, utility/case, CI95% percentile:

{markdown(contrasts[['dataset','c_over_b','delta_vs_release_all_per_case','delta_lo','delta_hi']])}

Taiwan c/b=2 có CI chứa 0. Không bỏ negative/inconclusive result này. CI conditional
on fixed fitted policies, không bao gồm fitting uncertainty, company-ID uncertainty
hoặc human utility uncertainty. [Tất cả grid/CI](utility_grid.csv).

Break-even tính lại từ row-level counts (chi phí compute/delay chưa tính):

{markdown(crossings[(crossings.condition=='overall') & (crossings.method=='stable_both')][['dataset','valid_lost','failures_avoided','c_over_b']])}

Khớp 893/505=1.768317 và 254/261=0.973180. Đây là crossing against release-all,
không phải crossing against strongest competing policy. [Các crossing khác](utility_break_even.csv).

![Policy frontier and utility sensitivity](utility_frontier.png)

## REPORTED — chỉ đọc từ evidence lịch sử

Policy training, original TreeSHAP, predictive quality và grouped top-2 historical
revision/AP không được chạy lại. Không dùng meaningful per-reason failure thay
revision cả explanation. Các file `utility_*_whole_explanation.csv` ghi riêng
**any meaningful failure among released reasons**, không gọi là top-2 revision.

Historical timing files được hash-check, nhưng thời gian chỉ **REPORTED**: first
16 fold-0 MCAR30 cases, 5 repetitions, CPU. Ví dụ donor/both-family là
0.084791/0.296253 seconds per 16-case batch ở Taiwan, 0.078514/0.789566 ở Polish.
Không ngoại suy thành từng condition, all-customer total hoặc milliseconds có
giá tiền. Top-1/release-all/whole-MC1/2/all/mask-warning không có timing đo riêng
trong artifact này: NOT_RUN, không gán proxy là đo thật.

## F1 — đáp án độc lập có ở mức synthetic, chưa có expert evidence

Đã export {bundle['cases']} synthetic cases từ pool {bundle['pool_cases']}, bốn
strata (revised/stable/near-tie/MC false-negative), mỗi strata 4. Inclusion probability
lưu private; balanced sample không đại diện prevalence. Có {bundle['allocation_slots']}
allocation slots, **0 participants, 0 ratings**, mỗi slot chỉ thấy một arm/case.
Randomized case order; 4 arms cân bằng; private answers/full records/restored
probability/computational labels tách khỏi expert JSONL. Empty rating form không
có câu trả lời giả. Không upload hoặc gửi bundle cho ai.

Independent task answer: current-record entailment của `A>=H`, H thuộc {{0,1}};
missing H phải xét cả hai khả năng. Answer chỉ phụ thuộc A và observed/missing H,
không phụ thuộc SHAP revision, policy score hoặc hidden future. Model output là
fixed analytic synthetic fixture, không phải bank-risk predictor đã validate.
Tên nhóm trung tính; positive contribution không biến thành một sự kiện tài chính.

Rubric này có correctness toán học và separation tests; **expert/business validity
NOT RUN**. Expert phải duyệt relevance/rubric trước khi gọi là nhiệm vụ đánh giá
người dùng phù hợp. Arm 4 dài hơn arm 3: chưa có matched-effort/time evidence.
Serialized-payload token-by-whitespace counts (bao gồm schema keys, chỉ UI proxy):

{markdown(lengths)}

## PROPOSED / NOT RUN

[Protocol/preregistration draft](https://github.com/QuocKhanhLuong/FinancePaper/blob/{manifest['branch']}/docs/DECISION_VALUE_PROTOCOL.md)
khóa primary 4-vs-3 accuracy, consent/ethics gate, missing/exclusion/stopping rules,
reviewer/case clustering và usability separation. Final SESOI, acceptable delay,
expert rubric approval, crossed-design power simulation, confirmatory sample size,
UI effort matching và registration **NOT RUN / PENDING**. Không có new financial
returns, human benefit, method novelty, expert agreement hoặc external validation.
TabM/Freddie/conformal/architecture sweeps không được chạy trong lượt này.

**Đúng một next action:** supervisor/domain expert duyệt synthetic task và rubric
độc lập trong protocol trước mọi bước thu thập người dùng.

## Reproduce và provenance

```bash
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python scripts/run_decision_value_pilot.py --output runs/decision_value_pilot/NEW_RUN
uv run --frozen --extra temporal python scripts/run_decision_value_pilot.py --output runs/decision_value_pilot/NEW_RUN --resume
# Sau khi lưu tests_final.xml/tests_final_receipt.json bằng suite CPU:
uv run --frozen --extra temporal python scripts/report_decision_value_pilot.py --run runs/decision_value_pilot/NEW_RUN
```

Runner refuses input/code/config drift on resume. New aggregate publication uses
an explicit allowlist; logs/models/row arrays/private and expert bundles remain local.
No raw or human data is committed. Main được fast-forward tới e7cac08 rồi tạo branch
mới; không merge research vào main. Khi bắt đầu, tracked/untracked Git worktree sạch;
ignored data/outputs được giữ và historical audit input hashes không đổi.
Lockfile sync removed two pre-existing extra packages; after the suite they were
restored at the same versions: treelite 4.7.0 and woodelf-explainer 0.4.8 from the
original local vendor directory. No dependency or historical lockfile was changed.

Data attribution: Yeh, I. (2009), [UCI Taiwan](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
DOI 10.24432/C55S3H; Tomczak, S. (2016), [UCI Polish](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
DOI 10.24432/C5F600. Official pages list CC BY 4.0 (checked 2026-10-07).
These are transformed aggregate results; targets remain distinct.
"""
    (run / "RESULTS.md").write_text(text)
    csv_names = [f"utility_{d}_{kind}.csv" for d in ("taiwan", "polish")
                 for kind in ("counts", "grid", "whole_explanation", "reconciliation")]
    csv_names += ["utility_counts.csv", "utility_grid.csv", "utility_break_even.csv"]
    safe = [run / name for name in csv_names + ["utility_frontier.png", "utility_frontier.pdf", "RESULTS.md"]]
    publication = {"run": str(run), "base_git_head": manifest["base_git_head"], "branch": manifest["branch"],
                   "status": "F0_REPRODUCED_F1_F2_PREPARED_HUMAN_NOT_RUN", "tests": {**totals, **tests},
                   "sources": manifest["frozen"]["sources"], "manifest_sha256": digest(run / "manifest.json"),
                   "input_files_verified": len(manifest["frozen"]["inputs"]),
                   "report_writer_sha256": digest(Path(__file__)),
                   "test_code_sha256": digest("tests/test_decision_value.py"),
                   "baseline_test_receipt_sha256": digest(baseline_test) if baseline_test.exists() else None,
                   "provenance_path_fix_sha256": digest("src/financepaper/experiments/robustness_followup.py"),
                   "artifacts": {p.name: digest(p) for p in safe}}
    write_json(run / "publication_receipt.json", publication)
    if publish:
        publish.mkdir(parents=True, exist_ok=False)
        for path in [*safe, run / "publication_receipt.json"]:
            shutil.copy2(path, publish / path.name)
    print(f"Report written: {run / 'RESULTS.md'}; safe aggregate files: {len(safe)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    report(args.run, args.publish)
