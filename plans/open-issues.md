# Open issues

Unresolved defects and unsatisfactory implementations to return to. Entries 1-9 (found in the first `/genagent:run` demo runs, 2026-10-16/17) were resolved on 2026-10-17; see `docs/development-log.md`.

## 1. No truly class-conditional scVI-family model (scANVI not in the registry) — resolved 2026-10-17

- **Issue:** `scvi` has no label input; its "conditional" sampling encodes real cells of the requested class and decodes them with fresh noise (`SCVIModel._latent_samples`, `src/genagent/models/scrna.py`). For rare classes this replays a handful of real cells (PBMC 3k: 9 megakaryocytes in the 60% training split) and cannot produce unseen states. `latent_diffusion_scvi` is the only scVI-based candidate with a learned per-class prior.
- **Candidate:** scANVI (`scvi.model.SCANVI`, scvi-tools) is scVI's semi-supervised extension with a label-dependent latent prior, i.e. a generative model of counts that can sample cells of a requested type. Adding it needs a sampler around its generative internals (label → latent prior → NB decoder) and tests.
- **Unresolved:** Deferred for the 2026-10-19 deadline. Until then the registry and decision guide describe `scvi`'s class conditioning as posterior replay, suitable only as a baseline for class balancing.
- **Resolution (2026-10-17):** `scanvi` added; it samples from scANVI's label-dependent latent prior (see `docs/generative-models.md` and `docs/development-log.md`). `scvi` keeps posterior replay as its documented class conditioning.
