#!/usr/bin/env python3
"""Prospective CPU-only numerical gate, never a financial benchmark."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
from itertools import product
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier
from financepaper.explanations.conditional_moments import (
    CompiledAttributions, FiniteJointLaw, Leaf, MomentBudgetExceeded, ProductLaw,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


@contextmanager
def stage_budget(seconds):
    def handler(*_):
        raise MomentBudgetExceeded('Stage exceeded wall-clock deadline')
    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, max(.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def moments(values):
    # Population moments of the declared equal-weight finite law, not ddof=1.
    mean = values.mean(axis=0)
    centered = values-mean
    return mean, centered.T@centered/len(values)


def cached_hd_class():
    from woodelf.core.path_to_s_vectors.woodelf_p2s import HighDepthWoodelfPathToSVectors

    class CachedHD(HighDepthWoodelfPathToSVectors):
        """Adaptation: cache exact background-specific tables across query calls.

        No upstream formula modified. The public HD hook accepts this subclass.
        Consumer patterns only index the returned table in upstream code.
        All dependencies of the table are part of the key. Baseline gets the
        same opportunity for model/background amortization as the candidate.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tables = {}

        def get_background_s_matrix(self, features_in_path, consumer_patterns,
                                    background_patterns, w, w_neighbor=None):
            key = (tuple(features_in_path), background_patterns.dtype.str,
                   background_patterns.shape, background_patterns.tobytes(),
                   float(w), None if w_neighbor is None else float(w_neighbor))
            if key not in self.tables:
                self.tables[key] = super().get_background_s_matrix(
                    features_in_path, consumer_patterns, background_patterns, w, w_neighbor)
            return self.tables[key]
    return CachedHD


def run(config, output):
    from woodelf import WoodelfExplainer
    from woodelf.core.cube_metric import ShapleyValues
    import shap

    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('Use a new output directory; previous receipts are immutable')
    cfg = yaml.safe_load(config.read_text())
    for package, key in [('woodelf_explainer', 'woodelf_version'), ('treelite', 'treelite_version')]:
        if importlib.metadata.version(package) != str(cfg[key]):
            raise ValueError(f'Expected pinned {package}={cfg[key]}')
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'xgboost', 'shap', 'woodelf_explainer', 'treelite']}
    from threadpoolctl import threadpool_info
    direct_url = importlib.metadata.distribution('woodelf_explainer').read_text('direct_url.json')
    origin = json.loads(direct_url) if direct_url else {}
    if origin.get('vcs_info', {}).get('commit_id') == cfg['woodelf_commit']:
        verified_vendor_commit = cfg['woodelf_commit']
    elif origin.get('url', '').startswith('file:'):
        vendor = Path(unquote(urlparse(origin['url']).path))
        verified_vendor_commit = subprocess.check_output(
            ['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
        if subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Official comparator checkout has local modifications')
    else:
        raise ValueError('Install the pinned official WOODELF source, not an unverified wheel')
    if verified_vendor_commit != cfg['woodelf_commit']:
        raise ValueError('Comparator source commit differs from preregistered revision')
    sources = [Path(__file__), Path('src/financepaper/explanations/conditional_moments.py'),
               config, Path('tests/test_conditional_moments.py'),
               Path('docs/method_pivot/ROUND2_PREREGISTRATION.md'),
               Path('docs/method_pivot/ROUND2_METHOD_SPEC.md'),
               Path('docs/method_pivot/ROUND2_DECISION.md'),
               Path('docs/method_pivot/ROUND2_AUDIT_AMENDMENT.md'), Path('uv.lock')]
    manifest = {'status': 'RUNNING', 'git_commit': git('rev-parse', 'HEAD'),
                'git_dirty': bool(git('status', '--porcelain')), 'device': 'cpu',
                'platform': platform.platform(), 'versions': versions,
                'cpu_model': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                    if platform.system() == 'Darwin' else platform.processor(),
                'host_ram_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
                    if platform.system() == 'Darwin' else None,
                'threadpools': threadpool_info(),
                'woodelf_direct_url': json.loads(direct_url) if direct_url else None,
                'woodelf_verified_commit': verified_vendor_commit,
                'python': sys.version, 'python_executable': sys.executable,
                'dirty_paths': git('status', '--porcelain').splitlines(),
                'runtime_environment': {k: os.environ.get(k) for k in
                    ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'DYLD_LIBRARY_PATH']},
                'hidden_query_values_used_for_moments_or_q': False,
                'outcomes_used_only_for_synthetic_predictor_training': True,
                'q_is_supplied_conditional_law_not_estimated_by_this_method': True,
                'novelty_gate': 'NO_GO_CURRENT_FORMULATION',
                'source_hashes': {str(p): sha(p) for p in sources},
                'config': cfg, 'evidence_type': 'SYNTHETIC_NUMERICAL_ONLY'}
    rows, workloads, diagnostics, response_arrays = [], [], [], {}
    started = time.perf_counter()
    dimension, repetitions = cfg['features'], cfg['timing_repetitions']
    observed = np.ones(dimension, bool)
    observed[cfg['hidden_features']] = False
    atom_vector = np.array(cfg['atoms'])
    atom_weights = np.full(len(atom_vector), 1/len(atom_vector))
    law = ProductLaw([atom_vector]*dimension, [atom_weights]*dimension)
    hidden_support = np.array(list(product(atom_vector, repeat=(~observed).sum())))
    CachedHD = cached_hd_class()

    def measure(fn):
        remaining = cfg['total_seconds']-(time.perf_counter()-started)
        if remaining <= 0:
            raise MomentBudgetExceeded('Total runner budget exceeded')
        t = time.perf_counter()
        with stage_budget(min(cfg['stage_seconds'], remaining)):
            value = fn()
        return value, time.perf_counter()-t

    try:
        # Fixed tiny correlated oracle/timing, distinct from the scale workload.
        fixture = CompiledAttributions([
            Leaf(0., ((0, -np.inf, .5),)),
            Leaf(1., ((0, .5, np.inf), (1, -np.inf, .5))),
            Leaf(3., ((0, .5, np.inf), (1, .5, np.inf))),
        ], np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]))
        fixture_law = FiniteJointLaw([[0., 0.], [1., 1.]], [.5, .5])
        fixture_rows = []
        for partial, mask in [([np.nan, np.nan], [False, False]), ([0., np.nan], [True, False])]:
            mask = np.array(mask)
            fixture_q = fixture_law.fix_observed(np.array(partial), mask)
            expected_m, expected_c = moments(fixture.values(fixture_q.points))
            for specialized in [True, False]:
                for repeat in range(repetitions+1):
                    value, secs = measure(lambda: fixture.moments(partial, mask, fixture_law,
                                                                  specialize=specialized))
                    np.testing.assert_allclose(value.mean, expected_m, atol=1e-12)
                    np.testing.assert_allclose(value.covariance, expected_c, atol=1e-12)
                    fixture_rows.append({'observed_count': int(mask.sum()), 'specialized': specialized,
                                         'repeat': repeat-1, 'seconds': secs, 'residual_terms': value.residual_terms})
        pd.DataFrame(fixture_rows).to_csv(output/'small_joint_fixture.csv', index=False)
        for seed in cfg['seeds']:
            for world in cfg['worlds']:
                rng = np.random.default_rng(seed)
                x = rng.normal(size=(cfg['training_rows'], dimension)).astype(np.float32)
                logit = -.8 + .9*x[:, 0] + .6*x[:, 1] + .4*x[:, 2]
                if world == 'interaction':
                    logit += 1.6*x[:, 0]*x[:, 1]
                y = rng.binomial(1, 1/(1+np.exp(-logit)))
                query = rng.normal(size=(cfg['query_rows'], dimension)).astype(np.float32)
                bg = x[:cfg['background_rows']]
                model = XGBClassifier(n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'],
                                      learning_rate=cfg['learning_rate'], n_jobs=1, random_state=seed)
                _, fit_seconds = measure(lambda: model.fit(x, y))
                compiler, compile_seconds = measure(lambda: CompiledAttributions.from_xgboost(model, bg))
                hd, hd_init_seconds = measure(lambda: WoodelfExplainer(model, pd.DataFrame(bg),
                                            model_output='raw', feature_perturbation='interventional'))
                cache, hd_cache_init_seconds = measure(lambda: CachedHD(ShapleyValues(), cfg['max_depth'], GPU=False))

                def hd_values(points):
                    return hd.shap_values(pd.DataFrame(points), path_to_matrices_calculator=cache,
                                          verbose=False)

                _, hd_warmup_seconds = measure(lambda: hd_values(query))
                stock = shap.TreeExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)),
                                          feature_perturbation='interventional', model_output='raw')
                point_error = float(np.max(np.abs(compiler.values(query)-stock.shap_values(query))))
                hd_error = float(np.max(np.abs(hd_values(query)-stock.shap_values(query))))
                if max(point_error, hd_error) > cfg['mean_covariance_tolerance']:
                    raise AssertionError(f'Point oracle mismatch {point_error}, {hd_error}')
                workload = {'seed': seed, 'world': world, 'fit_seconds': fit_seconds,
                            'compile_seconds': compile_seconds, 'hd_init_seconds': hd_init_seconds,
                            'hd_cache_init_seconds': hd_cache_init_seconds,
                            'hd_warmup_seconds': hd_warmup_seconds,
                            'compiled_terms': len(compiler.rectangles), 'point_error': point_error,
                            'hd_point_error': hd_error}
                workloads.append(workload)
                print(json.dumps({'workload': workload}), flush=True)
                for qi, point in enumerate(query):
                    base = {'seed': seed, 'world': world, 'query': qi}
                    support = np.tile(point, (len(hidden_support), 1))
                    support[:, ~observed] = hidden_support
                    (mean, covariance), exact_seconds = measure(lambda: moments(hd_values(support)))
                    score_full = model.predict(support, output_margin=True).astype(float)
                    score_background = model.predict(bg, output_margin=True).astype(float).mean()
                    rows.append({**base, 'method': 'woodelf_hd_enumeration', 'status': 'OK',
                                 'seconds': exact_seconds, 'k': len(support),
                                 'mean_error': 0., 'covariance_error': 0.})
                    partial = point.astype(float)
                    partial[~observed] = np.nan
                    for specialize in [True, False]:
                        name = 'specialized_moments' if specialize else 'generic_moments'
                        for repeat in range(repetitions+1):  # first is warm-up, logged separately
                            try:
                                result, seconds = measure(lambda: compiler.moments(
                                    partial, observed, law, specialize=specialize,
                                    max_terms=cfg['max_terms'], max_matrix_bytes=cfg['max_matrix_bytes'],
                                    timeout_seconds=cfg['stage_seconds']))
                                error_m = float(np.max(np.abs(result.mean-mean)))
                                error_c = float(np.max(np.abs(result.covariance-covariance)))
                                rows.append({**base, 'method': name, 'repeat': repeat-1,
                                             'status': 'OK', 'seconds': seconds,
                                             'mean_error': error_m, 'covariance_error': error_c,
                                             'residual_terms': result.residual_terms,
                                             'joint_matrix_bytes': result.joint_matrix_bytes,
                                             'rectangle_queries': result.rectangle_queries})
                                if max(error_m, error_c) > cfg['mean_covariance_tolerance']:
                                    raise AssertionError(f'Moment oracle mismatch: {error_m}, {error_c}')
                                if specialize and repeat == 0:
                                    key = f'{world}_{seed}_{qi}'
                                    response_arrays[f'{key}_mean'] = result.mean
                                    response_arrays[f'{key}_covariance'] = result.covariance
                                    efficiency = float(result.mean.sum() - (score_full.mean()-score_background))
                                    variance_efficiency = float(result.covariance.sum()-score_full.var())
                                    symmetry = float(np.max(np.abs(result.covariance-result.covariance.T)))
                                    min_eigenvalue = float(np.linalg.eigvalsh(result.covariance).min())
                                    diagnostics.append({**base, 'efficiency_error': efficiency,
                                                        'variance_efficiency_error': variance_efficiency,
                                                        'symmetry_error': symmetry, 'min_eigenvalue': min_eigenvalue,
                                                        'prediction_raw_score_variance': float(score_full.var()),
                                                        'attribution_variance_trace': float(np.trace(result.covariance))})
                                    if max(abs(efficiency), abs(variance_efficiency), symmetry, -min_eigenvalue) > cfg['mean_covariance_tolerance']:
                                        raise AssertionError('Scale covariance diagnostic invariant failed')
                            except MomentBudgetExceeded as exc:
                                rows.append({**base, 'method': name, 'status': 'BUDGET_EXCEEDED',
                                             'detail': str(exc)})
                                break
                    for k in cfg['mc_samples']:
                        for repeat in range(repetitions+1):
                            draw_rng = np.random.default_rng(np.random.SeedSequence([seed, qi, k, repeat]))
                            draws = support[draw_rng.integers(len(support), size=k)]
                            for name, fn in [('woodelf_hd_mc', hd_values),
                                             ('stock_shap_mc', stock.shap_values)]:
                                (m, c), seconds = measure(lambda: moments(fn(draws)))
                                rows.append({**base, 'method': name, 'k': k, 'repeat': repeat-1,
                                             'status': 'OK', 'seconds': seconds,
                                             'mean_error': float(np.max(np.abs(m-mean))),
                                             'covariance_error': float(np.max(np.abs(c-covariance)))})
                    print(json.dumps({'done': base, 'elapsed': time.perf_counter()-started}), flush=True)
        manifest['status'] = 'COMPLETE'
        manifest['execution_status'] = 'COMPLETE'
        manifest['full_scale_generic_ablation'] = 'BUDGET_EXCEEDED' if any(
            r['method'] == 'generic_moments' and r['status'] != 'OK' for r in rows) else 'COMPLETE'
        manifest['correlated_law_scale_performance'] = 'NOT_RUN'
    except MomentBudgetExceeded as exc:
        manifest.update(status='BUDGET_EXCEEDED', error=str(exc))
    except Exception as exc:
        manifest.update(status='FAILED', error=repr(exc))
        raise
    finally:
        pd.DataFrame(rows).to_csv(output/'measurements.csv', index=False)
        pd.DataFrame(diagnostics).to_csv(output/'covariance_diagnostics.csv', index=False)
        np.savez_compressed(output/'query_moments.npz', **response_arrays)
        (output/'workloads.json').write_text(json.dumps(workloads, indent=2)+'\n')
        manifest['elapsed_seconds'] = time.perf_counter()-started
        # ru_maxrss is bytes on macOS, KiB on Linux; whole-process high-water mark.
        manifest['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if platform.system() == 'Darwin' else 1024)
        manifest['artifact_hashes'] = {p.name: sha(p) for p in
                                      [output/'measurements.csv', output/'workloads.json',
                                       output/'covariance_diagnostics.csv', output/'query_moments.npz',
                                       output/'small_joint_fixture.csv'] if p.exists()}
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({'status': manifest['status'], 'output': str(output),
                          'elapsed': manifest['elapsed_seconds']}), flush=True)
    return manifest['status']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/conditional_moments_pilot.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    status = run(args.config, args.output)
    raise SystemExit(0 if status == 'COMPLETE' else 2)
