# Development log

## 2026-10-16: Imbalanced PathMNIST appendix demo

- **Change:** added a second class-balancing run on an artificially imbalanced PathMNIST variant (three classes cut to 10% of training images), alongside the natural-data run that becomes `generated_report_1`.
- **Why:** natural PathMNIST is only mildly imbalanced (max/min 1.63), so augmentation may show no gain over real-only or class-weighted training. The natural run demonstrates the agent judging whether generation is worthwhile; the variant demonstrates the case where it should help.
- **Consequences:** `scripts/make_imbalanced_pathmnist.py` builds the variant reproducibly; the `.npz` loader reads an optional `class_names` key. Validation and test splits are unchanged, so both runs are evaluated on the original distribution.
