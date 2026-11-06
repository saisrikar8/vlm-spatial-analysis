# Experiment fixes — November 1, 2026

Implemented a controlled development workflow and validation boundaries. No archived ILAMB images, truth files, raw generations or predictions were changed. No GPU jobs were submitted.

| Finding | Resolution |
| --- | --- |
| Finite native values can lack visible support | Archived random-pair dataset rejected by default. New controlled renderer uses exact tile coordinates. ILAMB preparation requires explicit per-source registration evidence and a colored, stable local support patch. Pixel checks alone are not treated as registration evidence. |
| Fixed approximate registration | New primary experiment has no reconstructed geographic transform. Legacy ILAMB labels remain unvalidated; they are not automatically corrected or promoted. |
| Ground truth unbound / missing rows shrink scope | Truth checksum, exact unique membership, manifest counts, map assignments and CSV/reference agreement enforced. |
| Missing provenance / different prompts or code pooled | Mandatory metadata, model revision, prompts, environment, code hashes and protocol fingerprint checks. Resumed and fresh tasks may pool only under the same protocol. |
| Out-of-range trials / unexpected methods or maps | Per-file method, trial and map constraints; raw sample IDs/seeds and CSV/raw agreement checked for new-format runs. |
| Direct numerical outputs rejected for wrapper syntax | Single finite JSON scalar accepted, along with the requested object. Prose, nonfinite values, booleans as numbers and ambiguous objects rejected. Strict-wrapper compliance reported separately in response_quality.csv. Archived strict results are not silently reparsed. |
| Different valid subsets | Matched per-map comparisons use the same valid trials/pairs; separate unconditional completion/invalid counts remain visible. Dependence comparison uses identical retained complete readings. |
| Pseudoreplication and questionable per-map tests | New scores have no binomial/Holm p-values. Dependence summaries average within source, then equally across sources within condition. No confidence/calibration claim. |
| Ratios, mixed units and query drift | New dataset declares only difference and ordering. Difference is primary, with a frozen 17-unit supported-value range. Old B-relative-to-A protocols remain separate. |
| No known ambiguity or oracle | Three explicitly specified controlled conditions, exact feasible readings/probabilities, joint/independent oracle CRPS and reference-expected predictive CRPS. This is a rule-following diagnostic, not proof of natural visual ambiguity. |
| Unbounded downloads | Bounded chunks, size limits for HTML/PNG/NetCDF and elapsed-time checks. |
| Imported path traversal | Contained relative paths and safe IDs enforced by current dataset validation and legacy pilot command construction; plotting already rejects unsafe filename IDs. |
| Interrupted work cannot resume | Explicit provenance-checked --resume, raw sample/seed checks, trailing interrupted-line recovery, per-generation durable append, and CSV reconstruction. Old evaluations are preserved. Changed protocols require a new run. |
| README/batch disagreement | Short current README; old instructions archived. Current array has 18 tasks, A5000, four CPUs, 30 minutes/task. |

Validation: 22 unittest tests passed under an isolated Python 3.14 CPU environment with numpy, Pillow, netCDF4, pyproj and matplotlib; Python compilation, Slurm shell syntax and current dry-run planning passed. Mocked execution verifies the two-image path and completed-run resume with zero extra generations. Oracle integration confirms a shared-offset pairing benefit and no independent-offset pairing benefit. All 2,600 archived prediction rows were separately reproduced under their strict parser, with 900 frozen hash references, three source hashes and 600 native-export target values checked.

Limits: real CUDA inference, remote scheduler state and actual GPU interruption/recovery have not been exercised. Original NetCDF alignment for every ILAMB pixel remains unverified. The new 18-example corpus is development data; it does not establish a research claim or calibrated model uncertainty. Six related triplets are six source groups, not eighteen independent sources. A fresh frozen test collection is needed for confirmatory evaluation.
