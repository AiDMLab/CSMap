from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CSMAP_DIR = ROOT / "core"
MOB_DIR = PROJECT_ROOT / "inputs" / "mob"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
REF_DIR = ROOT / "reference"
MOB_RDATA = MOB_DIR / "20221023.CARD_Simulate_Data.Paired_Cell_Resource.sc_count.sc_meta.spatial_count.spatial_location.Rdata"
MOB_FILES = {"scRNA": MOB_DIR/"MOBsc.csv", "cell_type": MOB_DIR/"MOBcelltype.csv", "st": MOB_DIR/"MOBst.csv", "coordinates": MOB_DIR/"MOBstxy.csv"}
CELL_TYPES = ["Astrocytes", "Ependymal", "Immune", "Neurons", "Oligos", "Vascular"]
ALGORITHMS = ["CSMap", "CytoSPACE", "Tangram", "CellTrek"]
