# Data sources and representation notes

These files are the exact benchmark representations supplied in the student's OptIForest archive. Their conversion has been checked against the corresponding supplied NPZ arrays. The final results use StandardScaler on extension features only. AD keeps its supplied representation. The raw CSV files are preserved.

| Dataset | Immediate source file | Underlying source | Representation note |
|---|---|---|---|
| AD | Authors' `data/ad.csv` | https://github.com/xiagll/OptIForest | 3,279 rows, 1,555 feature columns and label. This is the single original benchmark rerun here. |
| Annthyroid | `2_annthyroid.npz` | https://archive.ics.uci.edu/dataset/102/thyroid+disease | ADBench representation has six features. UCI lists several thyroid collections and a different feature summary. |
| BreastW | `4_breastw.npz` | https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original | Benchmark has 683 rows. Original UCI collection has 699. Do not attribute the difference to a processing step without source evidence. |
| Pima | `29_Pima.npz` | ADBench's supplied historical Pima representation | Exact original consent documentation and derivative redistribution terms were not recovered. The current UCI dataset ID 34 is a different diabetes dataset and must not be substituted as the Pima source. |
| PageBlocks | `27_PageBlocks.npz` | https://archive.ics.uci.edu/dataset/78/page+blocks+classification | Benchmark has 5,393 rows, compared with 5,473 in the original UCI collection. |
| Yeast | `47_yeast.npz` | https://archive.ics.uci.edu/dataset/110/yeast | Eight numerical feature columns. Protein identifiers are absent from this representation. |
| Synthetic | `scripts/generate_synthetic.py` | Project construction | Seed 42, 2,000 records, 1,800 normal, 80 point, 70 boundary, 50 clustered. Detailed type file supports diagnostics. |

The immediate benchmark collection is https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical . Its checked tree was `3dac8221081e190f157d78e93bfa8867f90d0965`. Files in this submission are the supplied local versions, identified by hashes in `results/data_audit.json`.

The current UCI entries for Thyroid Disease, Breast Cancer Wisconsin (Original), Page Blocks and Yeast state CC BY 4.0. Credit their creators and cite the associated dataset DOIs, which appear in the report bibliography. The ADBench software repository has a BSD-2-Clause licence. Software licensing and source-dataset permissions are distinct. There is no claim here that the repository licence resolves every historical data derivative or original participant-consent question.

Duplicates are retained. No additional imputation, zero-to-missing conversion, outlier removal, label reversal or class rebalancing is applied by the conversion script. Anomalies remain the supplied binary benchmark labels. Numeric clinical zeros, where present, are not assumed to be valid clinical measurements or imputed without an established source-specific protocol.

For broader research publication, verify unresolved source permissions and consent/governance requirements. Anomaly labels support this benchmark task and must not be interpreted as clinical judgments about individuals or populations.
