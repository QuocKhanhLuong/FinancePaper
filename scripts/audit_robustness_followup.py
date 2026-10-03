"""Recompute the completed follow-up artifacts without refitting models."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
from hashlib import sha256
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score, brier_score_loss, log_loss

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('outputs/robustness_followup_controlled'))
args = parser.parse_args()
out = args.output if args.output.is_absolute() else root / args.output
manifest = json.loads((out/'manifest.json').read_text())
assert manifest['status'] == 'completed'
resolved = json.loads((out/'resolved_config.json').read_text())
k = resolved['base']['explanations']['k']
epsilon = resolved['base']['explanations']['attribution_tolerance']
rank_tolerance = resolved['base']['explanations']['rank_tolerance']
magnitude = resolved['followup']['reason_min_attribution']
n_test, n_explain = manifest['test_n'], manifest['explanation_n']
for kind, base in [('source_sha256', root), ('artifact_sha256', out)]:
    for relative, expected in manifest[kind].items():
        assert sha256((base/relative).read_bytes()).hexdigest() == expected, relative
ids = json.loads((out/'evaluation/explanation_ids.json').read_text())
assert len(ids) == n_explain and len(set(ids)) == n_explain
metrics = json.loads((out/'evaluation/prediction_metrics.json').read_text())
metric_index = {(m['seed'],m['model'],m['condition'],m['probability']):m for m in metrics}
matches = pd.read_csv(out/'evaluation/matched_coverage.csv')
prediction_count = reason_count = archive_count = revision_count = 0
for seed in manifest['restart_seeds']:
    directory = out/f'evaluation/seed_{seed}'
    selection = json.loads((out/f'seed_{seed}/frozen_selection.json').read_text())
    assert selection['test_opened'] is False
    frame = pd.read_csv(directory/'predictions.csv')
    assert not frame.duplicated(['model','condition','record_id']).any()
    prediction_count += len(frame)
    for (name, condition), rows in frame.groupby(['model','condition']):
        assert len(rows) == n_test
        for kind in ('raw','calibrated'):
            p = rows[f'{kind}_probability'].to_numpy()
            y = rows.y.to_numpy()
            threshold = selection['models'][name][f'{kind}_threshold']
            binary = p >= threshold
            values = dict(roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),
                          recall=recall_score(y,binary),f1=f1_score(y,binary),
                          brier=brier_score_loss(y,p),log_loss=log_loss(y,p))
            actual = metric_index[(seed,name,condition,kind)]
            for key,value in values.items():
                assert abs(value-actual[key]) < 1e-8, (seed,name,condition,kind,key,value,actual[key])
    reasons = pd.read_csv(directory/'reasons.csv')
    assert not reasons.duplicated(['model','condition','record_id']).any()
    reason_count += len(reasons)
    for (name,condition), rows in reasons.groupby(['model','condition']):
        assert len(rows) == n_explain and rows.record_id.tolist() == ids
        archive = np.load(directory/f'{name}_{condition}_attributions.npz')
        archive_count += 1
        fields = archive['feature_names'].tolist()
        np.testing.assert_array_equal(archive['record_ids'], ids)
        mask = pd.read_csv(out/f'evaluation/masks/test_{condition}.csv',index_col=0).loc[ids,fields].to_numpy(dtype=bool)
        np.testing.assert_array_equal(mask, archive['hidden'])
        for i, row in enumerate(rows.itertuples(index=False)):
            phi, full = archive['original_before'][i], archive['original_full'][i]
            candidates = np.flatnonzero(~mask[i] & (phi > magnitude))
            indices = candidates[np.argsort(-phi[candidates],kind='stable')][:k]
            if not archive['valid_before'][i] or len(indices) < k:
                indices = np.array([],dtype=int)
            expected = [fields[j] for j in indices]
            assert json.loads(row.reasons) == expected
            eligible = bool(len(indices)==k and archive['valid_before'][i] and archive['valid_full'][i])
            assert bool(row.eligible) == eligible
            if eligible:
                event = any(full[j] <= epsilon or (full[~mask[i]] > full[j]+epsilon).sum() >= k + rank_tolerance for j in indices)
                assert bool(row.revision_event) == event
                revision_count += int(event)
            else:
                assert pd.isna(row.revision_event)
            if condition == 'complete':
                assert row.absolute_raw_probability_shift == 0
                assert not eligible or not bool(row.revision_event)
    names = sorted(reasons.model.unique())
    for condition, rows in reasons.groupby('condition'):
        common = set(ids)
        for name in names:
            common &= set(rows.loc[(rows.model==name)&rows.eligible&rows.attribution_valid,'record_id'])
        recorded = matches[(matches.seed==seed)&(matches.condition==condition)]
        for item in recorded.itertuples(index=False):
            count = min(int(np.floor(n_explain*item.target_coverage)),len(common))
            subset = rows[(rows.model==item.model)&rows.record_id.isin(common)].sort_values(
                ['reason_strength','record_id'],ascending=[False,True]).iloc[:count]
            assert item.common_n == len(common) and item.selected_n == count
            assert item.revised == subset.revision_event.sum()
            assert abs(item.actual_coverage-count/n_explain) < 1e-12
receipt = dict(status='passed', source_hashes=len(manifest['source_sha256']),
    artifact_hashes=len(manifest['artifact_sha256']),prediction_rows=prediction_count,
    reason_rows=reason_count,attribution_archives=archive_count,revision_events=revision_count,
    checks='all primary prediction metrics, masks, reasons, eligibility, revision events, matched coverage, complete controls',
    independent_worker_review=False, verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
(out/'evaluation/audit_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
