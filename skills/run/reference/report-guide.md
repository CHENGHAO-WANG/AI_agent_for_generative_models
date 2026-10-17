# Writing the report narrative

`GA report` builds the factual sections (profile, allocation, candidates, configurations, metric tables, figures, downstream tables, decision log) from the run files. You write the interpretive sections as JSON; every key is required.

```json
{
  "summary": "3-5 sentences: objective(s), decision to generate or not, selected model(s), headline result, main caveat",
  "objective_interpretation": "how the user's words were read (primary and secondary objectives), what n means here, questions asked, what was declined",
  "model_choice": "why these candidates for this data profile and objective, and why others were excluded",
  "hyperparameters": "initial settings and why; what the search changed; any refinement and its effect",
  "evaluation": "per candidate: strengths and failures with metric values, including per-class results for rare classes; why the selected model(s) win, and why one or several models serve the objectives",
  "downstream": "per objective: what the downstream analysis shows, compared with the non-generative alternative, and how certain the bootstrap intervals make it",
  "recommendations": {
    "suitable_uses": ["..."],
    "avoid": ["..."],
    "caveats": ["..."],
    "synthetic_to_real_ratio": "e.g. 'at most 1:2 for augmentation; synthetic data alone is not a substitute (TSTR ratio 0.55)'"
  },
  "limitations": ["limitations of this run and of the system"]
}
```

Write for a data scientist deciding whether to use the synthetic data. Ground every claim in this run's numbers (cite the metric and value). Markdown inside strings is rendered.

Recommendation checklist (cover each that applies):
- **Augmentation**: did real+synthetic (with or without class weighting) beat real-only and the class-weighted alternative on balanced accuracy and minority-class recall? Claim a gain only when the bootstrap interval of the difference excludes 0; otherwise call it inconclusive, and recommend the cheaper alternative when it is at least as good. Name the classes whose test counts are too small to judge.
- **Stand-in / sharing**: C2ST and DCR decide it. High C2ST means analyses on synthetic data will not transfer; DCR flags privacy risk.
- **Training models on synthetic only**: TSTR ratio.
- **Exploratory use** (prototyping pipelines, teaching, testing code): acceptable even at modest fidelity if no memorization.
- **Representation**: is the latent better than the unsupervised baselines (kNN accuracy, ARI, silhouette)? Recommend the latent for clustering / visualization only if so; call the comparison inconclusive when the baseline is marked `circular_reference_labels`.
- **Anomaly detection**: the threshold used and expected flag rate; AUROC if hold-out classes were scored.
- **scRNA specifics**: which cell populations are under- or over-represented (proportions table, against the real-vs-real baseline), whether marker genes are preserved, and which classes' synthetic cells replay few real cells (rare classes; `scvi`'s class conditioning); synthetic counts are for method development and augmentation, not for biological discovery.
- **Per-class reliability**: for every class you recommend augmenting, cite its class consistency and memorization check.
- **Ratio**: suggest a synthetic:real ratio tied to fidelity (lower fidelity → lower ratio).
