import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# --------------------------------------------------
# Output paths
# --------------------------------------------------
output_dir = Path("data_extended")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "synthetic.csv"
detailed_path = output_dir / "synthetic_detailed.csv"

# --------------------------------------------------
# 1. NORMAL OBSERVATIONS
# Three normal clusters, 600 observations each
# --------------------------------------------------

normal_a = rng.multivariate_normal(
    mean=[0, 0],
    cov=[[0.8, 0.1],
         [0.1, 0.8]],
    size=600
)

normal_b = rng.multivariate_normal(
    mean=[5, 5],
    cov=[[0.9, -0.1],
         [-0.1, 0.9]],
    size=600
)

normal_c = rng.multivariate_normal(
    mean=[-5, 5],
    cov=[[0.7, 0.0],
         [0.0, 0.7]],
    size=600
)

X_normal = np.vstack([
    normal_a,
    normal_b,
    normal_c
])

y_normal = np.zeros(len(X_normal), dtype=int)

normal_types = np.array(
    ["normal"] * len(X_normal),
    dtype=object
)

# --------------------------------------------------
# 2. POINT ANOMALIES
# Far away from the normal clusters
# --------------------------------------------------

point_1 = rng.uniform(
    low=[8, -8],
    high=[12, -4],
    size=(40, 2)
)

point_2 = rng.uniform(
    low=[-12, -8],
    high=[-8, -4],
    size=(40, 2)
)

X_point = np.vstack([point_1, point_2])

# --------------------------------------------------
# 3. BOUNDARY ANOMALIES
# Located close to the edges of normal regions
# --------------------------------------------------

boundary_1 = rng.multivariate_normal(
    mean=[2.7, 0],
    cov=[[0.12, 0.0],
         [0.0, 0.35]],
    size=35
)

boundary_2 = rng.multivariate_normal(
    mean=[2.5, 5],
    cov=[[0.15, 0.0],
         [0.0, 0.35]],
    size=35
)

X_boundary = np.vstack([
    boundary_1,
    boundary_2
])

# --------------------------------------------------
# 4. CLUSTERED ANOMALIES
# A compact abnormal group
# --------------------------------------------------

X_clustered = rng.multivariate_normal(
    mean=[0, 9],
    cov=[[0.25, 0.0],
         [0.0, 0.25]],
    size=50
)

# --------------------------------------------------
# Combine all anomaly types
# --------------------------------------------------

X_anomaly = np.vstack([
    X_point,
    X_boundary,
    X_clustered
])

y_anomaly = np.ones(len(X_anomaly), dtype=int)

anomaly_types = np.array(
    ["point"] * len(X_point)
    + ["boundary"] * len(X_boundary)
    + ["clustered"] * len(X_clustered),
    dtype=object
)

# --------------------------------------------------
# Combine normal and anomalous observations
# --------------------------------------------------

X = np.vstack([
    X_normal,
    X_anomaly
])

y = np.concatenate([
    y_normal,
    y_anomaly
])

types = np.concatenate([
    normal_types,
    anomaly_types
])

# --------------------------------------------------
# Shuffle observations
# --------------------------------------------------

indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]
types = types[indices]

# --------------------------------------------------
# OptIForest-ready dataset
#
# Columns:
# feature_1, feature_2, label
#
# No header because original OptIForest demo.py
# expects the final column to contain ground truth.
# --------------------------------------------------

optiforest_df = pd.DataFrame({
    "feature_1": X[:, 0],
    "feature_2": X[:, 1],
    "label": y
})

optiforest_df.to_csv(
    csv_path,
    index=False,
    header=False
)

# --------------------------------------------------
# Detailed version for analysis/documentation
# --------------------------------------------------

detailed_df = pd.DataFrame({
    "feature_1": X[:, 0],
    "feature_2": X[:, 1],
    "label": y,
    "anomaly_type": types
})

detailed_df.to_csv(
    detailed_path,
    index=False
)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=== Synthetic Dataset Created ===")

print("\nRandom seed:", SEED)

print("\nDataset shape:", optiforest_df.shape)
print("Feature columns: 2")

print("\nNormal:", np.sum(y == 0))
print("Anomalies:", np.sum(y == 1))

print(
    "Anomaly rate: {:.2f}%".format(
        np.mean(y == 1) * 100
    )
)

print("\nAnomaly type counts:")
print(detailed_df["anomaly_type"].value_counts())

print("\nMissing values:",
      optiforest_df.isna().sum().sum())

print("Infinite values:",
      np.isinf(X).sum())

print("\nOptIForest file:",
      csv_path)

print("Detailed file:",
      detailed_path)
