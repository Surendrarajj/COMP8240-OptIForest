"""Recreate the six synthetic evidence records and 12 explanation templates.

This script prepares inputs. Stored judgments were made by the assistant in
conversation, as disclosed in results/judge/execution.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    detail = pd.read_csv(ROOT / 'data_extended/synthetic_detailed.csv')
    scores = np.load(ROOT / 'results/synthetic_OptIForest_seed42.npz')['scores']
    rng = np.random.default_rng(123)
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    cases = []
    for kind in ['point', 'boundary', 'clustered']:
        choices = np.flatnonzero(detail.anomaly_type.eq(kind))
        for index in sorted(rng.choice(choices, 2, replace=False)):
            row = detail.iloc[index]
            vector = row[['feature_1', 'feature_2']].to_numpy(dtype=float)
            distances = np.linalg.norm(centers-vector, axis=1)
            center = centers[distances.argmin()]
            percentile = 100*(scores <= scores[index]).mean()
            evidence = dict(row_index=int(index), x1=round(float(vector[0]), 3),
                            x2=round(float(vector[1]), 3),
                            nearest_reference_center=center.tolist(),
                            euclidean_distance=round(float(distances.min()), 3),
                            anomaly_score=round(float(scores[index]), 6),
                            score_percentile=round(percentile, 2))
            grounded = (
                f"The observation ({evidence['x1']}, {evidence['x2']}) is "
                f"{evidence['euclidean_distance']} units from its nearest reference "
                f"center {center.tolist()}. Its anomaly score is at percentile "
                f"{percentile:.2f} in this dataset. This summarizes its position and "
                "model ranking; additional evidence would be needed to explain "
                "the detector's internal decision.")
            corrupted = (
                f"The observation ({evidence['x1']}, {evidence['x2']}) is exactly "
                f"0.000 units from its nearest reference center {center.tolist()}. "
                "The score proves with 100% certainty that this is a fraudulent "
                "financial transaction. The model has established the cause of the fraud.")
            for version, explanation in [('A', grounded), ('B', corrupted)]:
                cases.append(dict(case_id=f'R{index}{version}', evidence=evidence,
                                  explanation=explanation))
    target = ROOT / 'results/judge'
    target.mkdir(exist_ok=True)
    (target/'cases.json').write_text(json.dumps(cases, indent=2))
    (target/'prompt.txt').write_text((target/'rubric.txt').read_text()+'\n'+
                                   (target/'cases.json').read_text())
    print(f'Prepared {len(cases)} explanations. Existing judgments were not changed.')


if __name__ == '__main__':
    main()
