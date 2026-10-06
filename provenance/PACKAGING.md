# Packaging record (2026-10-07)

This package selects the original dense CSMap workflow and the final manuscript
plots from the authors' working directory. `source_files.json` records source
filenames and hashes for copied inputs and source modules. Its source paths are
relative descriptions of the original workspace, not runtime dependencies.
`SHA256SUMS` records the packaged files (excluding itself).

Packaging changes:

- Replaced workstation-specific input paths with repository-relative paths.
- Added a mapping CLI, a synthetic end-to-end example and a benchmark audit.
- Kept the original solver logic and compatibility readers. Removed the unused,
  broken `core/__init__.py` import of a nonexistent `cytospace` package.
- Retained the study's negative-quota guard as an explicit wrapper option.
- Flattened the final Figure 5 builder chain into one plotting script. Exported
  numeric arrays without pickle objects and retained the source histology.
- Reconstructed LAR plotting values from raw held-out expression and mappings;
  both selected sections match the saved manuscript values within 3e-16.
- Removed superseded plotting helpers and unrelated tissue branches. Added
  `--gastric-only` to run the gastric paired evaluation independently of MOB.
- Left frozen manuscript tables separate from newly generated experiment output.

Verification in an isolated Python 3.12 environment:

- Synthetic mapping: 24 assignments, eight exact capacities of three, 24 unique
  reference cells.
- Benchmark audit: all 60 CMR/WCMR/WRR/CU/OP-Score values match saved tables.
- Five negative-quota guard tests pass, including 10,000 randomized quota cases.
- Figures 2–5 render to PDF, PNG and SVG; final layouts visually inspected.
- All packaged Python modules compile.

The full biological mapping and baseline experiments were not rerun during
packaging. The exact historical SMLE computations remain subject to the
traceability limitation described in the main README.
