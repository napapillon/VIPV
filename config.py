"""
Central path configuration for the VIPV Lisbon repository.

Every script in ``code/`` resolves its input and output locations from the
constants defined here; no script contains a machine-specific path.

To run the code on another machine, either:
  - place the datasets under ``data/`` following the layout described in
    ``data/README.md`` (default, nothing to edit), or
  - edit DATA_ROOT below to point to the folder holding the datasets, or
  - set the environment variable VIPV_DATA_ROOT to that folder.

Importing this module also makes the shared modules in ``code/common/``
importable, so that scripts can use e.g. ``import clc_mapping`` directly.

Usage at the top of any script located in code/<theme>/:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import STANDARD_DIR, OUTPUT_ROOT

Authors : see AUTHORS file
"""

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Repository root
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Data root  (the only line to edit if the datasets live elsewhere)
# --------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("VIPV_DATA_ROOT", REPO_ROOT / "data"))

# Measurement campaigns
STANDARD_DIR = DATA_ROOT / "campaign_standard"    # Solar Jinie sensors J30-J33
TUKTUK_DIR   = DATA_ROOT / "campaign_tuktuk"      # tuk-tuk sensor J33

# Ground reference
FCUL_DIR = DATA_ROOT / "fcul_pyranometer"         # FCUL rooftop GHI (Excel)

# Land cover
CLC_VECTOR_PATH = DATA_ROOT / "land_cover" / "clc2018" / "U2018_CLC2018_V2020_20u1.gpkg"
CLC_RASTER_PATH = DATA_ROOT / "land_cover" / "clc2018" / "U2018_CLC2018_V2020_20u1.tif"
COS23_PATH      = DATA_ROOT / "land_cover" / "cos2023" / "COS2023v1-S2.gpkg"

# Urban geometry
BUILDINGS_PATH = DATA_ROOT / "buildings" / "lisbon_metro_buildings_height.geojson"

# Administrative boundaries (GADM 4.1, Portugal)
BOUNDARIES_DIR = DATA_ROOT / "boundaries"

# --------------------------------------------------------------------------
# Generated files (not versioned)
# --------------------------------------------------------------------------
OUTPUT_ROOT = REPO_ROOT / "outputs"   # figures and result tables
CACHE_DIR   = REPO_ROOT / "cache"     # downloaded CAMS series, OSM networks

# --------------------------------------------------------------------------
# Shared modules
# --------------------------------------------------------------------------
COMMON_DIR = REPO_ROOT / "code" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
