# Code provenance and third-party dependencies

`analysis/core/` preserves the original CSMap solver implementation supplied for this study; a stale package initializer was removed during packaging.
Its input/output utilities and assignment scaffolding include adaptations of CytoSPACE.
The upstream CytoSPACE notice is retained in `licenses/CytoSPACE-LICENSE.txt` and is
not replaced by MIT, Apache, or another blanket license. CytoSPACE is described in
Vahid et al., Nature Biotechnology (2023), https://doi.org/10.1038/s41587-023-01697-9.
Upstream: https://github.com/digitalcytometry/cytospace.

No separate open-source license is granted here for the authors' additions.
Contact the corresponding author regarding reuse and licensing. Depositing this
repository does not alter third-party rights or grant a patent license.

NumPy, pandas, SciPy, matplotlib, seaborn, Pillow, PyMuPDF, LAP, OR-Tools, h5py,
setuptools and rdata are installed separately under their respective terms.
The LAPJV implementation is provided by the `lap` dependency; cite Jonker and
Volgenant (1987), https://doi.org/10.1007/BF02278710.

`analysis/shims/` contains the compatibility readers used for CSV/TSV and Matrix
Market workflows. The scanpy shim intentionally does not implement read_visium.
Tangram, CellTrek and CytoSPACE baseline source code are not bundled; only their
saved benchmark assignments needed to audit the paper's comparisons are included.
