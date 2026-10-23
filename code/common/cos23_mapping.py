"""
================================================================================
COS2023 Land-Use Mapping — Shared Module
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Reference for the level-4 classes of COS2023 (Carta de Uso e Ocupação do Solo
2023, Direção-Geral do Território — DGT): English labels, level-1 branches,
and lookup and validation helpers.

--------------------------------------------------------------------------------
Scope
--------------------------------------------------------------------------------
CLC (clc_mapping.py) is the main land-cover reference of the pipeline.
COS2023 is a finer, complementary layer used for the urban sub-categories
of the Lisbon metropolitan area (e.g. vertical vs horizontal residential).

--------------------------------------------------------------------------------
Nomenclature
--------------------------------------------------------------------------------
The DGT nomenclature is published in Portuguese only. The 93 level-4 labels
below are an English translation made for this project, taken from
docs/nomenclatures/COS_Classification_EN.xlsx (codes and structure of the
DGT nomenclature unchanged): do not rename, merge or drop entries. Any code
found in the data but absent from the table is reported by
validate_against_nomenclature() and should be investigated.
No colours are defined here: the DGT palette is unrelated to the colour
convention of the project figures.

The spatial joins themselves are done by join_cos_standard.py and
join_cos23_tuktuk.py (COS23_PATH in config.py).

Usage
-----
Scripts in code/<theme>/ import it through the project configuration, which
adds code/common/ to sys.path:

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import ...
    import cos23_mapping as cos23

Self-test
---------
    python cos23_mapping.py

Requirements
------------
    pip install pandas

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.1
================================================================================
"""

from __future__ import annotations

import pandas as pd

# ==============================================================================
# COS2023 LEVEL-4 NOMENCLATURE  —  93 classes, English translation
# Source: docs/nomenclatures/COS_Classification_EN.xlsx (project translation
# of the DGT Portuguese nomenclature).
# Do NOT rename, merge, or drop entries. Use validate_against_nomenclature()
# to handle any code found in real data that is absent from this table.
# ==============================================================================

COS23_NOMENCLATURE: dict[str, str] = {

    # --- 1. Artificial surfaces (37 classes) ---------------------------------
    "1.1.1.1": "Continuous residential built-up areas, predominantly vertical",
    "1.1.1.2": "Continuous residential built-up areas, predominantly horizontal",
    "1.1.2.1": "Discontinuous residential built-up areas",
    "1.1.2.2": "Sparse discontinuous residential built-up areas",
    "1.2.1.1": "Industry and logistics",
    "1.2.1.2": "Commerce and services",
    "1.2.2.1": "Agricultural and livestock facilities",
    "1.3.1.1": "Cultural facilities",
    "1.3.2.1": "Sport facilities",
    "1.3.2.2": "Leisure facilities",
    "1.3.2.3": "Golf courses",
    "1.3.2.4": "Camping and caravan sites",
    "1.3.3.1": "Cemeteries",
    "1.3.4.1": "Other tourist facilities and equipment",
    "1.4.1.1": "Hydropower production infrastructures",
    "1.4.1.2": "Solar energy production infrastructures",
    "1.4.1.3": "Other renewable energy production infrastructures",
    "1.4.2.1": "Fossil fuel energy production infrastructures",
    "1.4.3.1": "Substations and energy transformation stations",
    "1.4.4.1": "Water extraction and treatment infrastructures for consumption",
    "1.4.4.2": "Wastewater drainage and treatment infrastructures",
    "1.4.5.1": "Landfills",
    "1.4.5.2": "Other waste infrastructures",
    "1.4.6.1": "Other infrastructures",
    "1.5.1.1": "Road network",
    "1.5.1.2": "Rail network",
    "1.5.2.1": "Sea and river port terminals",
    "1.5.2.2": "Shipyards and dry docks",
    "1.5.2.3": "Marinas and fishing docks",
    "1.5.3.1": "Airports",
    "1.5.3.2": "Airfields",
    "1.5.4.1": "Parking areas",
    "1.6.1.1": "Open-cast mines",
    "1.6.1.2": "Quarries",
    "1.7.1.1": "Unbuilt vacant spaces",
    "1.7.1.2": "Construction sites",
    "1.8.1.1": "Green spaces",

    # --- 2. Agricultural areas (11 classes) ----------------------------------
    "2.1.1.1": "Rainfed and irrigated temporary crops",
    "2.1.1.2": "Rice fields",
    "2.2.1.1": "Vineyards",
    "2.2.2.1": "Orchards",
    "2.2.3.1": "Olive groves",
    "2.3.1.1": "Temporary crops and/or improved pastures associated with vineyards",
    "2.3.1.2": "Temporary crops and/or improved pastures associated with orchards",
    "2.3.1.3": "Temporary crops and/or improved pastures associated with olive groves",
    "2.3.2.1": "Complex cultivation and parcel mosaics",
    "2.3.3.1": "Agriculture with natural and semi-natural spaces",
    "2.4.1.1": "Protected agriculture and greenhouses",

    # --- 3. Pastures and natural grasslands (2 classes) ----------------------
    "3.1.1.1": "Improved pastures",
    "3.1.2.1": "Natural grasslands / Spontaneous pastures",

    # --- 4. Agro-forestry surfaces (12 classes) ------------------------------
    "4.1.1.1": "Agro-silvicultural surfaces with cork oak",
    "4.1.1.2": "Agro-silvicultural surfaces with holm oak",
    "4.1.1.3": "Agro-silvicultural surfaces with other oaks",
    "4.1.1.4": "Agro-silvicultural surfaces with other broad-leaved trees",
    "4.1.2.1": "Agro-silvicultural surfaces with stone pine",
    "4.1.2.2": "Agro-silvicultural surfaces with other conifers",
    "4.2.1.1": "Silvopastoral surfaces with cork oak",
    "4.2.1.2": "Silvopastoral surfaces with holm oak",
    "4.2.1.3": "Silvopastoral surfaces with other oaks",
    "4.2.1.4": "Silvopastoral surfaces with other broad-leaved trees",
    "4.2.2.1": "Silvopastoral surfaces with stone pine",
    "4.2.2.2": "Silvopastoral surfaces with other conifers",

    # --- 5. Forests (11 classes) ---------------------------------------------
    "5.1.1.1": "Cork oak forests",
    "5.1.1.2": "Holm oak forests",
    "5.1.1.3": "Other oak forests",
    "5.1.1.4": "Chestnut forests",
    "5.1.1.5": "Carob forests",
    "5.1.1.6": "Eucalyptus forests",
    "5.1.1.7": "Acacia forests",
    "5.1.1.8": "Other broad-leaved forests",
    "5.1.2.1": "Maritime pine forests",
    "5.1.2.2": "Stone pine forests",
    "5.1.2.3": "Other coniferous forests",

    # --- 6. Shrublands (1 class) ---------------------------------------------
    "6.1.1.1": "Shrublands",

    # --- 7. Open spaces with little or no vegetation (4 classes) ------------
    "7.1.1.1": "Inland beaches, dunes and sands",
    "7.1.1.2": "Coastal beaches, dunes and sands",
    "7.1.2.1": "Rocky areas",
    "7.1.3.1": "Sparsely vegetated areas",

    # --- 8. Wetlands (3 classes) ---------------------------------------------
    "8.1.1.1": "Peat bogs and fens",
    "8.1.2.1": "Salt marshes",
    "8.1.2.2": "Intertidal zones",

    # --- 9. Surface water bodies (12 classes) --------------------------------
    "9.1.1.1": "Natural watercourses",
    "9.1.1.2": "Modified or artificial watercourses",
    "9.1.2.1": "Artificial inland lakes and ponds",
    "9.1.2.2": "Natural inland lakes and ponds",
    "9.1.2.3": "Dam reservoirs",
    "9.1.2.4": "Weir reservoirs",
    "9.1.2.5": "Ponds",
    "9.2.1.1": "Aquaculture",
    "9.3.1.1": "Salinas",
    "9.3.2.1": "Coastal lagoons",
    "9.3.3.1": "Estuaries / River mouths",
    "9.3.4.1": "Ocean",
}

# Level-1 branch labels (9 branches).
# Key: first digit of a level-4 code (e.g. "1.1.1.1" -> "1").
# Source: docs/nomenclatures/COS_Classification_EN.xlsx, LVL1_LABEL column.
COS23_LEVEL1_LABELS: dict[str, str] = {
    "1": "Artificial surfaces",
    "2": "Agricultural areas",
    "3": "Pastures and natural grasslands",
    "4": "Agro-forestry surfaces",
    "5": "Forests",
    "6": "Shrublands",
    "7": "Open spaces with little or no vegetation",
    "8": "Wetlands",
    "9": "Surface water bodies",
}


# ==============================================================================
# LOOKUP HELPERS
# ==============================================================================

def get_level1_code(code: str) -> str:
    """Return the level-1 code (e.g. '1') for a level-4 code (e.g. '1.1.1.1')."""
    return str(code).split(".")[0]


def get_level1_label(code: str) -> str:
    """Return the English level-1 branch label for a level-4 code."""
    return COS23_LEVEL1_LABELS.get(get_level1_code(code), "Unknown level-1 branch")


def get_label(code, default: str = "Unknown COS2023 code") -> str:
    """
    Return the English COS2023 level-4 label of `code`.

    Returns `default` (not an exception) for any code absent from the table,
    so this function is safe to use directly inside a pandas .apply() over
    real data that may occasionally contain an unexpected code.

    Parameters
    ----------
    code : str or None or float (NaN)
        The level-4 code to look up (e.g. '1.1.1.1').
    default : str
        Value returned when the code is not found in COS23_NOMENCLATURE.

    Returns
    -------
    str
        English label, or `default`, or 'Unknown / missing' for None/NaN.
    """
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "Unknown / missing"
    return COS23_NOMENCLATURE.get(str(code).strip(), default)


def validate_against_nomenclature(
    df: pd.DataFrame, code_col: str = "cos23_n4_code"
) -> None:
    """
    Print every distinct code in df[code_col] that is NOT in COS23_NOMENCLATURE,
    with its observation count.

    An unexpected code points to a transcription error in COS23_NOMENCLATURE,
    a COS version mismatch with the polygon source, or a genuinely rare
    class; it should be investigated rather than silently grouped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the COS2023 join results.
    code_col : str
        Name of the column holding the level-4 codes (default: 'cos23_n4_code').
    """
    if code_col not in df.columns:
        print(f"Column '{code_col}' not found in DataFrame.")
        return

    counts = df.groupby(code_col, dropna=True).size()
    unknown = [
        (code, count) for code, count in counts.items()
        if str(code).strip() not in COS23_NOMENCLATURE
    ]
    if not unknown:
        print("All codes found in the data match the nomenclature.")
        return

    print("Codes NOT found in COS23_NOMENCLATURE (investigate before analysis):")
    for code, count in sorted(unknown, key=lambda x: -x[1]):
        print(f"  {code!r}: {count} observations")


# ==============================================================================
# SELF-TEST
# ==============================================================================

def _self_test() -> None:
    """Nomenclature size, spot-checked labels and lookup helpers."""
    assert len(COS23_NOMENCLATURE) == 93, (
        f"Expected 93 level-4 classes, found {len(COS23_NOMENCLATURE)}"
    )
    assert len(set(COS23_NOMENCLATURE.keys())) == 93, "Duplicate codes detected"
    assert len(COS23_LEVEL1_LABELS) == 9, "Expected 9 level-1 branches"

    spot_checks = {
        "1.1.1.1": "Continuous residential built-up areas, predominantly vertical",
        "1.1.1.2": "Continuous residential built-up areas, predominantly horizontal",
        "1.5.4.1": "Parking areas",
        "1.8.1.1": "Green spaces",
        "5.1.1.6": "Eucalyptus forests",
        "9.3.4.1": "Ocean",
        "2.4.1.1": "Protected agriculture and greenhouses",
        "6.1.1.1": "Shrublands",
    }
    for code, expected in spot_checks.items():
        actual = get_label(code)
        assert actual == expected, (
            f"get_label({code!r}) = {actual!r}, expected {expected!r}"
        )

    assert get_level1_code("4.2.1.3") == "4"
    assert get_level1_label("4.2.1.3") == "Agro-forestry surfaces"
    assert get_label("9.9.9.9") == "Unknown COS2023 code"
    assert get_label(None) == "Unknown / missing"

    print(
        f"Self-test passed: {len(COS23_NOMENCLATURE)} classes, "
        f"{len(spot_checks)} spot checks, level-1 lookups OK."
    )


if __name__ == "__main__":
    _self_test()