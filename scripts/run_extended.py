import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# Allow this script to import detectors from the project root
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from detectors import OptIForest


parser = argparse.ArgumentParser(
    description="Run OptIForest on extension datasets."
)

parser.add_argument(
    "--dataset",
    required=True,
    help="Dataset name, e.g. annthyroid or synthetic"
)

parser.add_argument(
    "--threshold",
    type=int,
    default=403,
    help="OptIForest threshold"
)

parser.add_argument(
    "--branch",
    type=int,
    default=0,
    help="OptIForest branch parameter"
)

parser.add_argument(
    "--runs",
    type=int,
    default=15,
    help="Number of repeated experiments"
)

args = parser.parse_args()

dataset_name = args.dataset
threshold = args.threshold
branch = args.branch
runs = args.runs

num_ensemblers = 100

data_path = Path("data_extended") / f"{dataset_name}.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(data_path, header=None)

X = df.iloc[:, :-1].values
ground_truth = df.iloc[:, -1].values

print("=" * 60)
print("OptIForest Extension Experiment")
print("=" * 60)

print("Dataset:", dataset_name)
print("Path:", data_path)
print("Observations:", X.shape[0])
print("Features:", X.shape[1])

print(
    "Normal:",
    np.sum(ground_truth == 0)
)

print(
    "Anomalies:",
    np.sum(ground_truth == 1)
)

print(
    "Anomaly rate: {:.2f}%".format(
        np.mean(ground_truth == 1) * 100
    )
)

print("Trees / ensemblers:", num_ensemblers)
print("Threshold:", threshold)
print("Branch:", branch)
print("Runs:", runs)

# --------------------------------------------------
# Run OptIForest
# --------------------------------------------------

AUC = []
PR_AUC = []
Traintime = []
Testtime = []

for run in range(runs):

    # Create a fresh model for each repetition
    detector = OptIForest(
        "L2OPT",
        num_ensemblers,
        threshold,
        branch
    )

    start_train = time.time()

    detector.fit(X)

    train_time = time.time() - start_train

    start_test = time.time()

    y_pred = detector.decision_function(X)

    test_time = time.time() - start_test

    # Preserve the scoring convention used in demo.py
    anomaly_scores = -1.0 * y_pred

    auc = roc_auc_score(
        ground_truth,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        ground_truth,
        anomaly_scores
    )

    AUC.append(auc)
    PR_AUC.append(pr_auc)
    Traintime.append(train_time)
    Testtime.append(test_time)

    print(
        f"Run {run + 1:02d}/{runs} | "
        f"ROC-AUC: {auc:.4f} | "
        f"PR-AUC: {pr_auc:.4f} | "
        f"Train: {train_time:.3f}s | "
        f"Test: {test_time:.3f}s"
    )

# --------------------------------------------------
# Final results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    "ROC-AUC mean: {:.6f}".format(
        np.mean(AUC)
    )
)

print(
    "ROC-AUC std:  {:.6f}".format(
        np.std(AUC)
    )
)

print(
    "PR-AUC mean:  {:.6f}".format(
        np.mean(PR_AUC)
    )
)

print(
    "PR-AUC std:   {:.6f}".format(
        np.std(PR_AUC)
    )
)

print(
    "Training time mean: {:.4f}s".format(
        np.mean(Traintime)
    )
)

print(
    "Testing time mean:  {:.4f}s".format(
        np.mean(Testtime)
    )
)

print("=" * 60)
