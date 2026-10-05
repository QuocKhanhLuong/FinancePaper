#!/usr/bin/env python3
"""Post-pilot strong baseline, frozen in ROUND2_FOURIER_CONTROL_PROTOCOL.md."""
import argparse
import json
from pathlib import Path
import resource
import platform
import time
import numpy as np
import pandas as pd
import shap
import yaml
from xgboost import XGBClassifier
from financepaper.explanations.conditional_moments import CompiledAttributions, ProductLaw
from financepaper.explanations.moment_baselines import uniform_four_atom_moments
from run_conditional_moments_pilot import git, sha, stage_budget


def run(output):
    config = Path('configs/conditional_moments_pilot.yaml')
    cfg = yaml.safe_load(config.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('Use a new output directory')
    files = [Path(__file__), config, Path('src/financepaper/explanations/moment_baselines.py'),
             Path('src/financepaper/explanations/conditional_moments.py'),
             Path('docs/method_pivot/ROUND2_FOURIER_CONTROL_PROTOCOL.md')]
    manifest = {'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                'source_hashes': {str(p): sha(p) for p in files}, 'config': cfg,
                'kind': 'KNOWN_BASIS_MOMENT_ADAPTATION_NOT_OFFICIAL_FOURIERSHAP',
                'status': 'RUNNING', 'device': 'cpu'}
    rows, setups = [], []
    started = time.perf_counter()
    observed = np.ones(cfg['features'], bool)
    observed[cfg['hidden_features']] = False
    law = ProductLaw([cfg['atoms']]*cfg['features'], [[.25]*4]*cfg['features'])
    try:
        with stage_budget(cfg['total_seconds']):
            for seed in cfg['seeds']:
                for world in cfg['worlds']:
                    rng = np.random.default_rng(seed)
                    x = rng.normal(size=(cfg['training_rows'], cfg['features'])).astype(np.float32)
                    logit = -.8+.9*x[:, 0]+.6*x[:, 1]+.4*x[:, 2]
                    if world == 'interaction':
                        logit += 1.6*x[:, 0]*x[:, 1]
                    y = rng.binomial(1, 1/(1+np.exp(-logit)))
                    query = rng.normal(size=(cfg['query_rows'], cfg['features'])).astype(np.float32)
                    model = XGBClassifier(n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'],
                                          learning_rate=cfg['learning_rate'], n_jobs=1, random_state=seed).fit(x,y)
                    bg = x[:cfg['background_rows']]
                    t = time.perf_counter()
                    compiler = CompiledAttributions.from_xgboost(model, bg)
                    setups.append({'seed': seed, 'world': world, 'shared_compile_seconds': time.perf_counter()-t})
                    stock = shap.TreeExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)),
                                              feature_perturbation='interventional', model_output='raw')
                    np.testing.assert_allclose(compiler.values(query), stock.shap_values(query), atol=2e-5)
                    for qi, point in enumerate(query):
                        partial = point.astype(float)
                        partial[~observed] = np.nan
                        for repeat in range(cfg['timing_repetitions']+1):
                            t = time.perf_counter()
                            original = compiler.moments(partial, observed, law, max_terms=cfg['max_terms'],
                                                        timeout_seconds=cfg['stage_seconds'])
                            original_seconds = time.perf_counter()-t
                            t = time.perf_counter()
                            control = uniform_four_atom_moments(compiler, partial, observed, cfg['atoms'],
                                                               timeout_seconds=cfg['stage_seconds'])
                            control_seconds = time.perf_counter()-t
                            error_mean = float(np.max(np.abs(control['mean']-original.mean)))
                            error_cov = float(np.max(np.abs(control['covariance']-original.covariance)))
                            if max(error_mean, error_cov) > 1e-8:
                                raise AssertionError('Independent operator mismatch')
                            rows.append({'seed': seed, 'world': world, 'query': qi, 'repeat': repeat-1,
                                         'rectangle_seconds': original_seconds, 'fourier_seconds': control_seconds,
                                         'mean_error': error_mean, 'covariance_error': error_cov,
                                         'frequencies': control['frequencies'], 'expanded_terms': control['expanded_terms'],
                                         'coefficient_bytes': control['coefficient_bytes']})
                    print(json.dumps({'completed': [seed, world], 'elapsed': time.perf_counter()-started}), flush=True)
        manifest['status'] = 'COMPLETE'
    except Exception as exc:
        manifest.update(status='FAILED', error=repr(exc))
        raise
    finally:
        pd.DataFrame(rows).to_csv(output/'measurements.csv', index=False)
        (output/'setups.json').write_text(json.dumps(setups, indent=2)+'\n')
        manifest['elapsed_seconds'] = time.perf_counter()-started
        manifest['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if platform.system() == 'Darwin' else 1024)
        manifest['artifact_hashes'] = {p.name: sha(p) for p in [output/'measurements.csv', output/'setups.json']}
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({'status': manifest['status'], 'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
