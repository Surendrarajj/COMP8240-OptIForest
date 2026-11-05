"""Seeded, resumable evaluation of the supplied OptIForest implementation.

Run from any directory. Each finished seed is saved before the next starts.
The detector source is preserved. Raw numeric features and duplicate rows are
retained to match demo.py. Labels enter metric calculation only.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detectors import OptIForest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True,
                        choices=['ad', 'annthyroid', 'breastw', 'pima',
                                 'pageblocks', 'yeast', 'synthetic'])
    parser.add_argument('--runs', type=int, default=15)
    parser.add_argument('--start-seed', type=int, default=42)
    parser.add_argument('--standardize', action='store_true',
                        help='Fit StandardScaler to feature columns before unsupervised evaluation')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    if args.output is None:
        args.output = ROOT / ('results_scaled' if args.standardize else 'results')
    if not args.standardize and args.dataset != 'ad' and (ROOT/'STOP_RAW').exists():
        print('Raw extension pilot stopped; use --standardize for final evaluation', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    # Prevent two resumed processes from writing results for the same dataset.
    lock_stream = (args.output / (args.dataset + '.lock')).open('w')
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    if not args.standardize and args.dataset != 'ad' and (ROOT/'STOP_RAW').exists():
        print('Raw extension pilot stopped', flush=True)
        return
    folder = 'data' if args.dataset == 'ad' else 'data_extended'
    data_path = ROOT / folder / (args.dataset + '.csv')
    data = pd.read_csv(data_path, header=None).to_numpy(dtype=float)
    X, y = data[:, :-1], data[:, -1].astype(int)
    if not np.isfinite(data).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected finite numeric features and binary 0/1 labels')
    sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if args.standardize:
        X = StandardScaler().fit_transform(X)
    metadata = {
        'dataset': args.dataset, 'sha256': sha256, 'rows': len(X),
        'features': X.shape[1], 'anomalies': int(y.sum()),
        'preprocessing': ('StandardScaler on full feature matrix; duplicates retained'
                          if args.standardize else 'raw features; duplicates retained; no scaling'),
        'evaluation': 'fit and score same full dataset; labels for metrics only',
        'trees': 100, 'sample_size': min(512, len(X)),
        'threshold': 403, 'branch': 0, 'granularity': 1,
        'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': np.__version__, 'pandas': pd.__version__,
        'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
        'cpu': platform.processor(), 'timing': 'wall time; shared host load varies',
    }
    metadata_path = args.output / (args.dataset + '_metadata.json')
    path = args.output / (args.dataset + '_runs.csv')
    records = []
    if path.exists():
        old_meta = json.loads(metadata_path.read_text())
        if old_meta != metadata:
            raise ValueError('Existing results have different metadata; choose a new output')
        with path.open(newline='') as stream:
            records = list(csv.DictReader(stream))
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    completed = {(r['model'], int(r['seed'])) for r in records}
    fields = ['dataset', 'model', 'seed', 'roc_auc', 'average_precision',
              'fit_seconds', 'score_seconds', 'runtime_warning_count']
    for seed in range(args.start_seed, args.start_seed + args.runs):
        for name in ['OptIForest', 'IsolationForest']:
            if (name, seed) in completed:
                continue
            random.seed(seed)
            np.random.seed(seed)
            if name == 'OptIForest':
                model = OptIForest('L2OPT', 100, 403, 0)
            else:
                model = IsolationForest(n_estimators=100, max_samples=512,
                                        contamination='auto', random_state=seed,
                                        n_jobs=1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', RuntimeWarning)
                start = time.perf_counter()
                model.fit(X)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                scores = -np.asarray(model.decision_function(X)).reshape(-1)
                score_seconds = time.perf_counter() - start
            if not np.isfinite(scores).all():
                raise ValueError(f'{name} returned non-finite scores')
            record = dict(dataset=args.dataset, model=name, seed=seed,
                          roc_auc=roc_auc_score(y, scores),
                          average_precision=average_precision_score(y, scores),
                          fit_seconds=fit_seconds, score_seconds=score_seconds,
                          runtime_warning_count=sum(issubclass(w.category, RuntimeWarning)
                                                    for w in caught))
            new_file = not path.exists()
            with path.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                if new_file:
                    writer.writeheader()
                writer.writerow(record)
            np.savez_compressed(args.output / f'{args.dataset}_{name}_seed{seed}.npz',
                                labels=y, scores=scores)
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
