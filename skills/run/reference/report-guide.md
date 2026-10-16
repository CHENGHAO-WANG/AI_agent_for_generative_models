# Writing the report narrative

`GA report` builds the factual sections (profile, allocation, candidates, configurations, metric tables, figures, downstream tables, decision log) from the run files. You write the interpretive sections as JSON; every key is required.

```json
{
  "summary": "3-5 sentences: objective, decision to generate or not, selected model, headline result, main caveat",
  "objective_interpretation": "how the user's words were read, what n means here, questions asked, what was declined",
  "model_choice": "why these candidates for this data profile and objective, and why others were excluded",
  "hyperparameters": "initial settings and why; what the search changed; any refinement and its effect",
  "evaluation": "per candidate: strengths and failures with metric values; why the selected one wins",
  "downstream": "what the downstream analysis shows, compared with the non-generative alternative",
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
- **Augmentation**: did real+synthetic beat real-only and the class-weighted alternative on balanced accuracy and minority-class recall, beyond the seed-to-seed spread? If not, recommend the alternative.
- **Stand-in / sharing**: C2ST and DCR decide it. High C2ST means analyses on synthetic data will not transfer; DCR flags privacy risk.
- **Training models on synthetic only**: TSTR ratio.
- **Exploratory use** (prototyping pipelines, teaching, testing code): acceptable even at modest fidelity if no memorization.
- **Representation**: is the latent better than the baseline embedding (kNN accuracy, ARI, silhouette)? Recommend the latent for clustering / visualization only if so.
- **Anomaly detection**: the threshold used and expected flag rate; AUROC if hold-out classes were scored.
- **scRNA specifics**: which cell populations are under- or over-represented (cluster proportions), whether marker genes are preserved; synthetic counts are for method development and augmentation, not for biological discovery.
- **Ratio**: suggest a synthetic:real ratio tied to fidelity (lower fidelity → lower ratio).
