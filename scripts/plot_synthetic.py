import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_path = "data_extended/synthetic_detailed.csv"
output_path = "data_extended/synthetic_visualization.png"

df = pd.read_csv(input_path)

plt.figure(figsize=(10, 7))

markers = {
    "normal": "o",
    "point": "X",
    "boundary": "^",
    "clustered": "s"
}

sizes = {
    "normal": 18,
    "point": 70,
    "boundary": 55,
    "clustered": 55
}

for anomaly_type in ["normal", "point", "boundary", "clustered"]:
    subset = df[df["anomaly_type"] == anomaly_type]

    plt.scatter(
        subset["feature_1"],
        subset["feature_2"],
        label=f"{anomaly_type.capitalize()} ({len(subset)})",
        marker=markers[anomaly_type],
        s=sizes[anomaly_type],
        alpha=0.7
    )

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Constructed Synthetic Anomaly Detection Dataset")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(output_path, dpi=300)
plt.show()

print("Visualization saved to:", output_path)
