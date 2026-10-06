"""Run original CSMap without modifying source files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from config import CSMAP_DIR, MOB_FILES, OUTPUT_DIR


def _setup_import_path() -> None:
    shim_dir = str(Path(__file__).resolve().parent / "shims")
    csmap_dir = str(CSMAP_DIR)
    if shim_dir not in sys.path:
        sys.path.insert(0, shim_dir)
    if csmap_dir not in sys.path:
        sys.path.insert(0, csmap_dir)
    _inject_ortools_pywrapgraph_stub()


def _inject_ortools_pywrapgraph_stub() -> None:
    """Newer ortools removed pywrapgraph; CSMap only needs it for lap_CSPR."""
    import types

    if "ortools.graph.pywrapgraph" in sys.modules:
        return

    try:
        from ortools.graph import pywrapgraph  # noqa: F401
        return
    except ImportError:
        pass

    from scipy.optimize import linear_sum_assignment

    class LinearSumAssignment:
        OPTIMAL = 0
        INFEASIBLE = 1
        POSSIBLE_OVERFLOW = 2

        def __init__(self):
            self._cost = None

        def AddArcWithCost(self, worker, task, cost):
            pass

        def Solve(self):
            row_ind, col_ind = linear_sum_assignment(self._cost)
            self._row_ind = row_ind
            self._col_ind = col_ind
            self._optimal_cost = int(self._cost[row_ind, col_ind].sum())
            return self.OPTIMAL

        def OptimalCost(self):
            return self._optimal_cost

        def NumNodes(self):
            return len(self._row_ind)

        def RightMate(self, i):
            return int(self._col_ind[i])

        def AssignmentCost(self, i):
            return int(self._cost[i, self._col_ind[i]])

    mod = types.ModuleType("pywrapgraph")
    mod.LinearSumAssignment = LinearSumAssignment

    import ortools.graph as og

    og.pywrapgraph = mod
    sys.modules["ortools.graph.pywrapgraph"] = mod


def run_csmap(
    output_folder: str | Path,
    output_prefix: str = "mob_",
    lambda_1: float | None = None,
    mean_cell_numbers: int = 10,
    seed: int = 1,
    plot_off: bool = True,
    solver_method: str = "lapjv_compat",
    scRNA_path: str | Path | None = None,
    cell_type_path: str | Path | None = None,
    st_path: str | Path | None = None,
    coordinates_path: str | Path | None = None,
    cellfracs_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    n_cells_per_spot_path: str | Path | None = None,
    geometry: str = "square",
) -> Path:
    _setup_import_path()
    from csmap import main_csmap

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if cellfracs_path is not None:
        frac_src = Path(cellfracs_path).resolve()
        weights_src = Path(weights_path or frac_src.with_name("Seurat_weights.txt")).resolve()
    else:
        frac_src = (Path.cwd() / "testresult" / "Seurat_cellfracs.txt").resolve()
        weights_src = (Path.cwd() / "testresult" / "Seurat_weights.txt").resolve()
        if not frac_src.exists():
            frac_src = (Path(__file__).resolve().parent / "testresult" / "Seurat_cellfracs.txt").resolve()
            weights_src = (Path(__file__).resolve().parent / "testresult" / "Seurat_weights.txt").resolve()
    if not frac_src.exists():
        raise FileNotFoundError(
            f"Missing Seurat_cellfracs.txt. Run prepare_mob_data.py first."
        )

    scRNA_path = Path(scRNA_path or MOB_FILES["scRNA"]).resolve()
    cell_type_path = Path(cell_type_path or MOB_FILES["cell_type"]).resolve()
    st_path = Path(st_path or MOB_FILES["st"]).resolve()
    coordinates_path = Path(coordinates_path or MOB_FILES["coordinates"]).resolve()

    if n_cells_per_spot_path is not None:
        n_cells_per_spot_path = Path(n_cells_per_spot_path).resolve()
    cwd = os.getcwd()
    try:
        os.chdir(output_folder)
        (Path("testresult")).mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(frac_src, Path("testresult") / "Seurat_cellfracs.txt")
        shutil.copy2(weights_src, Path("testresult") / "Seurat_weights.txt")

        main_csmap(
            scRNA_path=str(scRNA_path),
            cell_type_path=str(cell_type_path),
            st_path=str(st_path),
            coordinates_path=str(coordinates_path),
            n_cells_per_spot_path=(str(Path(n_cells_per_spot_path).resolve()) if n_cells_per_spot_path else None),
            st_cell_type_path=None,
            cell_type_fraction_estimation_path=str(weights_src),
            output_folder=".",
            output_prefix=output_prefix,
            mean_cell_numbers=mean_cell_numbers,
            downsample_off=False,
            scRNA_max_transcripts_per_cell=1500,
            solver_method=solver_method,
            distance_metric="Pearson_correlation",
            sampling_method="duplicates",
            single_cell=False,
            number_of_processors=1,
            seed=seed,
            plot_off=plot_off,
            geometry=geometry,
            max_num_cells_plot=50000,
            num_column=3,
            lambda_1=lambda_1,
        )
    finally:
        os.chdir(cwd)

    return output_folder / f"{output_prefix}assigned_locations.csv"
