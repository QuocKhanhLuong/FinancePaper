#!/usr/bin/env python3
"""Validate raw artifact hashes and summarize the fixed numerical gate."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def summarize(run):
    manifest = json.loads((run/'manifest.json').read_text())
    for name, expected in manifest['artifact_hashes'].items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    raw = pd.read_csv(run/'measurements.csv')
    # Repeat -1 is warm-up. Enumeration has one timing per query, not 5 repeats.
    data = raw[(raw.status == 'OK') & (raw.repeat.fillna(0) >= 0)].copy()
    data['k'] = data.k.fillna(0).astype(int)
    summary = data.groupby(['method', 'k']).agg(
        measurements=('seconds', 'size'), seconds_mean=('seconds', 'mean'),
        seconds_sd=('seconds', 'std'), mean_error_mean=('mean_error', 'mean'),
        covariance_error_mean=('covariance_error', 'mean'),
        covariance_error_max=('covariance_error', 'max'),
    ).reset_index()
    summary.to_csv(run/'summary.csv', index=False)
    data.groupby(['seed', 'world', 'method', 'k']).agg(
        seconds_mean=('seconds', 'mean'), covariance_error_mean=('covariance_error', 'mean')
    ).to_csv(run/'by_seed_world.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for name, label in [('stock_shap_mc', 'Stock TreeSHAP + MC'),
                        ('woodelf_hd_mc', 'Cached WOODELF-HD + MC')]:
        subset = summary[summary.method == name]
        axes[0].plot(subset.k, subset.seconds_mean*1000, 'o-', label=label)
        axes[1].plot(subset.seconds_mean*1000, subset.covariance_error_mean, 'o-', label=label)
    exact = summary[summary.method == 'specialized_moments'].iloc[0]
    enumeration = summary[summary.method == 'woodelf_hd_enumeration'].iloc[0]
    axes[0].axhline(exact.seconds_mean*1000, color='black', linestyle='--', label='Exact specialized moments')
    axes[0].axhline(enumeration.seconds_mean*1000, color='grey', linestyle=':', label='Exact finite enumeration')
    axes[0].set(xscale='log', yscale='log', xlabel='K completions', ylabel='Query time (ms)')
    axes[1].scatter([exact.seconds_mean*1000], [exact.covariance_error_mean], marker='*', s=130,
                    color='black', label='Exact specialized moments')
    axes[1].set(xscale='log', yscale='log', xlabel='Query time (ms)',
                ylabel='Mean max-entry covariance error')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle('Synthetic numerical workload only; compilation reported separately')
    fig.savefig(run/'accuracy_cost.png', dpi=180)
    fig.savefig(run/'accuracy_cost.pdf')
    plt.close(fig)
    receipt = {'manifest_sha256': hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
               'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'warmup_excluded': True,
               'sd_scope': '5 repeats x 24 queries for MC/specialized; 24 distinct queries for enumeration',
               'not_a_confidence_interval': True,
               'raw_counts': {str(k): int(v) for k, v in raw.groupby(['method','status']).size().items()}}
    (run/'report_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(summary.to_string(index=False))


def summarize_control(control):
    manifest = json.loads((control/'manifest.json').read_text())
    for name, expected in manifest['artifact_hashes'].items():
        if hashlib.sha256((control/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Control artifact hash mismatch: {name}')
    data = pd.read_csv(control/'measurements.csv').query('repeat >= 0')
    paired = data.groupby(['seed','world']).agg(
        rectangle_ms=('rectangle_seconds', lambda s: s.mean()*1000),
        fourier_ms=('fourier_seconds', lambda s: s.mean()*1000),
    )
    paired.to_csv(control/'paired_summary.csv')
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for (seed, world), row in paired.iterrows():
        ax.plot([0,1], [row.rectangle_ms, row.fourier_ms], 'o-', label=f'{seed} / {world}')
    ax.set_xticks([0,1], ['Proposed rectangle moments', 'Known Fourier-basis control'])
    ax.set_ylabel('Mean query time (ms)')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.set_title('Same moment target; fixed four-atom product law\nAdaptation, not official FourierSHAP results')
    fig.savefig(control/'paired_cost.png', dpi=180)
    fig.savefig(control/'paired_cost.pdf')
    plt.close(fig)
    print(paired.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    summarize(args.run)
    if args.control is not None:
        summarize_control(args.control)
