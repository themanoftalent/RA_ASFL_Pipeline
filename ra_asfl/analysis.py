"""Analyze completed experiments."""
import json
from pathlib import Path
import numpy as np
from scipy import stats


def paired_comparison(proposed, baseline):
    a, b = np.asarray(proposed, float), np.asarray(baseline, float)
    if len(a) != len(b) or len(a) < 2 or not np.isfinite(a-b).all():
        raise ValueError('Need at least two complete, matched seed-level observations')
    delta = a-b
    mean, sd = float(delta.mean()), float(delta.std(ddof=1))
    interval = stats.t.interval(0.95, len(delta)-1, loc=mean, scale=sd/np.sqrt(len(delta)))
    return {'n': len(delta), 'mean_difference': mean, 'sample_sd': sd,
            'ci95': [float(z) for z in interval],
            'paired_t_p': float(stats.ttest_rel(a, b).pvalue),
            'exact_wilcoxon_p': float(stats.wilcoxon(delta, alternative='two-sided',
                                                    method='exact').pvalue)}


def factorial_interaction(f0, f1, f2, f3):
    arrays = [np.asarray(a, float) for a in (f0, f1, f2, f3)]
    if len(set(map(len, arrays))) != 1:
        raise ValueError('Factorial configurations require matched seeds')
    delta = arrays[3]-arrays[2]-arrays[1]+arrays[0]
    return {'interaction_by_seed': delta.tolist(),
            'analysis': paired_comparison(delta, np.zeros_like(delta))}


def collect_runs(folder, metric='macro_f1'):
    output = []
    for file in sorted(Path(folder).rglob('results.json')):
        doc = json.loads(file.read_text())
        output.append({'file': str(file), 'seed': doc['seed'],
                       'method': doc['method'], 'dataset': doc['dataset'],
                       'metric': doc['test_metrics'].get(metric)})
    return output
