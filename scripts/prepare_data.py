"""Convert the supplied ADBench NPZ files and validate the stored CSVs.

No imputation, scaling, deduplication or relabelling is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {'annthyroid': '2_annthyroid', 'breastw': '4_breastw',
         'pima': '29_Pima', 'pageblocks': '27_PageBlocks', 'yeast': '47_yeast'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--convert', action='store_true',
                        help='Regenerate CSVs from the supplied raw NPZ files')
    args = parser.parse_args()
    rows = []
    for name, raw in FILES.items():
        source = ROOT / 'data_extended' / 'raw' / f'{raw}.npz'
        target = ROOT / 'data_extended' / f'{name}.csv'
        with np.load(source, allow_pickle=False) as archive:
            expected = np.column_stack((archive['X'], archive['y']))
        if args.convert:
            pd.DataFrame(expected).to_csv(target, index=False, header=False)
        observed = pd.read_csv(target, header=None).to_numpy(dtype=float)
        if observed.shape != expected.shape or not np.allclose(
                observed, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f'NPZ/CSV mismatch: {name}')
        if not np.isfinite(observed).all() or set(np.unique(observed[:, -1])) != {0, 1}:
            raise ValueError(f'Invalid data in {name}')
        rows.append(dict(dataset=name, rows=len(observed),
                         features=observed.shape[1]-1,
                         anomalies=int(observed[:, -1].sum()),
                         duplicate_full_rows=int(pd.DataFrame(observed).duplicated().sum()),
                         duplicate_features=int(pd.DataFrame(observed[:, :-1]).duplicated().sum()),
                         missing_or_infinite=0, npz_csv_equal=True,
                         csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'data_audit.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
