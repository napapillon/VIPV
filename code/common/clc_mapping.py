"""
================================================================================
CORINE Land Cover (CLC) Mapping — Shared Module
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Single reference for the CLC 2018 land-cover classes (44 classes) used across
the VIPV pipeline. Every script imports the labels and the Urban / Rural
grouping from here instead of defining its own copy.

--------------------------------------------------------------------------------
Index convention
--------------------------------------------------------------------------------
Keys are CLC grid_codes (integers 1–44), as written by the CLC joins:
    land_use_clcv  (join_clc_standard.py / join_clc_tuktuk.py, reference)
    land_use_clcr  (join_clc_raster_standard.py, cross-check)

The raw 'land_use' column delivered with the Solar Jinie files also holds the
grid_code (no offset), but its 'land_use_str' label is shifted by one class
and must not be used: always take labels from this module.

    land_use_clcv  ==  grid_code  ==  CLC_MAPPING key   (no arithmetic)

-999 marks a missing value (no polygon, invalid row). clean_data.py keeps
only 1 <= land_use_clcv <= 44; 44 is a genuine class (Sea and ocean, 523).

--------------------------------------------------------------------------------
CLC_MAPPING
--------------------------------------------------------------------------------
Key   : grid_code (int, 1–44)
Value : CLCEntry named tuple
    clc_code    (int)  : 3-digit CLC code     e.g. 111
    lvl1_code   (int)  : Level-1 code         e.g. 1
    lvl1_label  (str)  : Level-1 label        e.g. 'Artificial surfaces'
    lvl2_code   (int)  : Level-2 code         e.g. 11
    lvl2_label  (str)  : Level-2 label        e.g. 'Urban fabric'
    lvl3_label  (str)  : Level-3 label        e.g. 'Continuous urban fabric'
    urban_rural (str)  : 'Urban' or 'Rural'   (project grouping)

Derived lookups (grid_code -> value)
    CLC_LVL3_LABEL, CLC_LVL2_LABEL, CLC_LVL1_LABEL : labels
    CLC_URBAN_RURAL                                : 'Urban' / 'Rural'
    CLC_LVL1_CODE, CLC_LVL2_CODE, CLC_CODE         : numeric codes

Usage
-----
    df['label'] = df['land_use_clcv'].map(CLC_LVL3_LABEL)
    df['group'] = df['land_use_clcv'].map(CLC_URBAN_RURAL)

Source
------
docs/nomenclatures/CLC_Classification.xlsx: Corine Land Cover official legend
(44 classes) and Urban / Rural grouping.

Self-test
---------
    python clc_mapping.py

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

from typing import NamedTuple


# ==============================================================================
# NAMED TUPLE — one entry per CLC class
# ==============================================================================

class CLCEntry(NamedTuple):
    """All CLC attributes for a single land-cover class."""
    clc_code:    int   # 3-digit CLC code          (e.g. 111)
    lvl1_code:   int   # Level-1 numeric code       (e.g. 1)
    lvl1_label:  str   # Level-1 label              (e.g. 'Artificial surfaces')
    lvl2_code:   int   # Level-2 numeric code       (e.g. 11)
    lvl2_label:  str   # Level-2 label              (e.g. 'Urban fabric')
    lvl3_label:  str   # Level-3 label (LVL3)       (e.g. 'Continuous urban fabric')
    urban_rural: str   # Aggregated category        ('Urban' or 'Rural')


# ==============================================================================
# CLC_MAPPING  —  key: grid_code (1–44), value: CLCEntry
# Source: docs/nomenclatures/CLC_Classification.xlsx
# ==============================================================================

CLC_MAPPING: dict[int, CLCEntry] = {

    # ── LVL1: Artificial surfaces ─────────────────────────────────────────────
    #    LVL2: Urban fabric
     1: CLCEntry(111, 1, 'Artificial surfaces', 11, 'Urban fabric',
                 'Continuous urban fabric',                                  'Urban'),
     2: CLCEntry(112, 1, 'Artificial surfaces', 11, 'Urban fabric',
                 'Discontinuous urban fabric',                               'Urban'),
    #    LVL2: Industrial, commercial and transport units
     3: CLCEntry(121, 1, 'Artificial surfaces', 12, 'Industrial, commercial and transport units',
                 'Industrial or commercial units',                           'Urban'),
     4: CLCEntry(122, 1, 'Artificial surfaces', 12, 'Industrial, commercial and transport units',
                 'Road and rail networks and associated land',               'Urban'),
     5: CLCEntry(123, 1, 'Artificial surfaces', 12, 'Industrial, commercial and transport units',
                 'Port areas',                                               'Urban'),
     6: CLCEntry(124, 1, 'Artificial surfaces', 12, 'Industrial, commercial and transport units',
                 'Airports',                                                 'Urban'),
    #    LVL2: Mine, dump and construction sites
     7: CLCEntry(131, 1, 'Artificial surfaces', 13, 'Mine, dump and construction sites',
                 'Mineral extraction sites',                                 'Rural'),
     8: CLCEntry(132, 1, 'Artificial surfaces', 13, 'Mine, dump and construction sites',
                 'Dump sites',                                               'Rural'),
     9: CLCEntry(133, 1, 'Artificial surfaces', 13, 'Mine, dump and construction sites',
                 'Construction sites',                                       'Urban'),
    #    LVL2: Artificial, non-agricultural vegetated areas
    10: CLCEntry(141, 1, 'Artificial surfaces', 14, 'Artificial, non-agricultural vegetated areas',
                 'Green urban areas',                                        'Urban'),
    11: CLCEntry(142, 1, 'Artificial surfaces', 14, 'Artificial, non-agricultural vegetated areas',
                 'Sport and leisure facilities',                             'Urban'),

    # ── LVL1: Agricultural areas ──────────────────────────────────────────────
    #    LVL2: Arable land
    12: CLCEntry(211, 2, 'Agricultural areas', 21, 'Arable land',
                 'Non-irrigated arable land',                                'Rural'),
    13: CLCEntry(212, 2, 'Agricultural areas', 21, 'Arable land',
                 'Permanently irrigated land',                               'Rural'),
    14: CLCEntry(213, 2, 'Agricultural areas', 21, 'Arable land',
                 'Rice fields',                                              'Rural'),
    #    LVL2: Permanent crops
    15: CLCEntry(221, 2, 'Agricultural areas', 22, 'Permanent crops',
                 'Vineyards',                                                'Rural'),
    16: CLCEntry(222, 2, 'Agricultural areas', 22, 'Permanent crops',
                 'Fruit trees and berry plantations',                        'Rural'),
    17: CLCEntry(223, 2, 'Agricultural areas', 22, 'Permanent crops',
                 'Olive groves',                                             'Rural'),
    #    LVL2: Pastures
    18: CLCEntry(231, 2, 'Agricultural areas', 23, 'Pastures',
                 'Pastures',                                                 'Rural'),
    #    LVL2: Heterogeneous agricultural areas
    19: CLCEntry(241, 2, 'Agricultural areas', 24, 'Heterogeneous agricultural areas',
                 'Annual crops associated with permanent crops',             'Rural'),
    20: CLCEntry(242, 2, 'Agricultural areas', 24, 'Heterogeneous agricultural areas',
                 'Complex cultivation patterns',                             'Rural'),
    21: CLCEntry(243, 2, 'Agricultural areas', 24, 'Heterogeneous agricultural areas',
                 'Land principally occupied by agriculture, '
                 'with significant areas of natural vegetation',             'Rural'),
    22: CLCEntry(244, 2, 'Agricultural areas', 24, 'Heterogeneous agricultural areas',
                 'Agro-forestry areas',                                      'Rural'),

    # ── LVL1: Forest and semi-natural areas ───────────────────────────────────
    #    LVL2: Forests
    23: CLCEntry(311, 3, 'Forest and semi-natural areas', 31, 'Forests',
                 'Broad-leaved forest',                                      'Rural'),
    24: CLCEntry(312, 3, 'Forest and semi-natural areas', 31, 'Forests',
                 'Coniferous forest',                                        'Rural'),
    25: CLCEntry(313, 3, 'Forest and semi-natural areas', 31, 'Forests',
                 'Mixed forest',                                             'Rural'),
    #    LVL2: Shrub and/or herbaceous vegetation associations
    26: CLCEntry(321, 3, 'Forest and semi-natural areas', 32,
                 'Shrub and/or herbaceous vegetation associations',
                 'Natural grasslands',                                       'Rural'),
    27: CLCEntry(322, 3, 'Forest and semi-natural areas', 32,
                 'Shrub and/or herbaceous vegetation associations',
                 'Moors and heathland',                                      'Rural'),
    28: CLCEntry(323, 3, 'Forest and semi-natural areas', 32,
                 'Shrub and/or herbaceous vegetation associations',
                 'Sclerophyllous vegetation',                                'Rural'),
    29: CLCEntry(324, 3, 'Forest and semi-natural areas', 32,
                 'Shrub and/or herbaceous vegetation associations',
                 'Transitional woodland-shrub',                              'Rural'),
    #    LVL2: Open spaces with little or no vegetation
    30: CLCEntry(331, 3, 'Forest and semi-natural areas', 33,
                 'Open spaces with little or no vegetation',
                 'Beaches, dunes, sands',                                    'Rural'),
    31: CLCEntry(332, 3, 'Forest and semi-natural areas', 33,
                 'Open spaces with little or no vegetation',
                 'Bare rocks',                                               'Rural'),
    32: CLCEntry(333, 3, 'Forest and semi-natural areas', 33,
                 'Open spaces with little or no vegetation',
                 'Sparsely vegetated areas',                                 'Rural'),
    33: CLCEntry(334, 3, 'Forest and semi-natural areas', 33,
                 'Open spaces with little or no vegetation',
                 'Burnt areas',                                              'Rural'),
    34: CLCEntry(335, 3, 'Forest and semi-natural areas', 33,
                 'Open spaces with little or no vegetation',
                 'Glaciers and perpetual snow',                              'Rural'),

    # ── LVL1: Wetlands ────────────────────────────────────────────────────────
    #    LVL2: Inland wetlands
    35: CLCEntry(411, 4, 'Wetlands', 41, 'Inland wetlands',
                 'Inland marshes',                                           'Rural'),
    36: CLCEntry(412, 4, 'Wetlands', 41, 'Inland wetlands',
                 'Peat bogs',                                                'Rural'),
    #    LVL2: Maritime wetlands
    37: CLCEntry(421, 4, 'Wetlands', 42, 'Maritime wetlands',
                 'Salt marshes',                                             'Rural'),
    38: CLCEntry(422, 4, 'Wetlands', 42, 'Maritime wetlands',
                 'Salines',                                                  'Rural'),
    39: CLCEntry(423, 4, 'Wetlands', 42, 'Maritime wetlands',
                 'Intertidal flats',                                         'Rural'),

    # ── LVL1: Water bodies ────────────────────────────────────────────────────
    #    LVL2: Inland waters
    40: CLCEntry(511, 5, 'Water bodies', 51, 'Inland waters',
                 'Water courses',                                            'Rural'),
    41: CLCEntry(512, 5, 'Water bodies', 51, 'Inland waters',
                 'Water bodies',                                             'Rural'),
    #    LVL2: Marine waters
    42: CLCEntry(521, 5, 'Water bodies', 52, 'Marine waters',
                 'Coastal lagoons',                                          'Rural'),
    43: CLCEntry(522, 5, 'Water bodies', 52, 'Marine waters',
                 'Estuaries',                                                'Rural'),
    44: CLCEntry(523, 5, 'Water bodies', 52, 'Marine waters',
                 'Sea and ocean',                                            'Rural'),
}


# ==============================================================================
# DERIVED LOOKUP DICTIONARIES  (computed once at import time)
# ==============================================================================

# grid_code → LVL3 label  (most specific, e.g. 'Continuous urban fabric')
CLC_LVL3_LABEL: dict[int, str] = {k: v.lvl3_label  for k, v in CLC_MAPPING.items()}

# grid_code → LVL2 label  (intermediate, e.g. 'Urban fabric')
CLC_LVL2_LABEL: dict[int, str] = {k: v.lvl2_label  for k, v in CLC_MAPPING.items()}

# grid_code → LVL1 label  (broad, e.g. 'Artificial surfaces')
CLC_LVL1_LABEL: dict[int, str] = {k: v.lvl1_label  for k, v in CLC_MAPPING.items()}

# grid_code → 'Urban' or 'Rural'
CLC_URBAN_RURAL: dict[int, str] = {k: v.urban_rural for k, v in CLC_MAPPING.items()}

# grid_code → LVL1 numeric code  (1, 2, 3, 4, 5)
CLC_LVL1_CODE: dict[int, int]   = {k: v.lvl1_code   for k, v in CLC_MAPPING.items()}

# grid_code → LVL2 numeric code  (11, 12, 13 … 52)
CLC_LVL2_CODE: dict[int, int]   = {k: v.lvl2_code   for k, v in CLC_MAPPING.items()}

# grid_code → 3-digit CLC code  (111, 112, 121 … 523)
CLC_CODE: dict[int, int]        = {k: v.clc_code    for k, v in CLC_MAPPING.items()}


# ==============================================================================
# SENTINEL VALUES
# ==============================================================================

LAND_USE_NO_SATELLITE: int = -999   # missing value — removed by clean_data.py

# Valid grid_code range (clean_data.py keeps 1 <= land_use_clcv <= 44)
LAND_USE_MIN: int = 1
LAND_USE_MAX: int = 44


# ==============================================================================
# QUICK SELF-TEST  (python clc_mapping.py)
# ==============================================================================

if __name__ == '__main__':
    print('CLC_MAPPING self-test')
    print('=' * 72)

    assert len(CLC_MAPPING) == 44, f"Expected 44 entries, got {len(CLC_MAPPING)}"
    assert set(CLC_MAPPING.keys()) == set(range(1, 45)), \
        "Keys must cover exactly 1–44"

    valid_ur = {'Urban', 'Rural'}
    for lu, entry in CLC_MAPPING.items():
        assert entry.urban_rural in valid_ur, \
            f"land_use={lu}: invalid urban_rural '{entry.urban_rural}'"

    for name, d in [('CLC_LVL3_LABEL',  CLC_LVL3_LABEL),
                    ('CLC_LVL2_LABEL',  CLC_LVL2_LABEL),
                    ('CLC_LVL1_LABEL',  CLC_LVL1_LABEL),
                    ('CLC_URBAN_RURAL', CLC_URBAN_RURAL),
                    ('CLC_LVL1_CODE',   CLC_LVL1_CODE),
                    ('CLC_LVL2_CODE',   CLC_LVL2_CODE),
                    ('CLC_CODE',        CLC_CODE)]:
        assert len(d) == 44, f"{name}: expected 44 entries, got {len(d)}"

    # Verify valid range constants
    assert LAND_USE_MIN == 1  and LAND_USE_MAX == 44

    print(f"  {'land_use':>10}  {'clc_code':>8}  {'urban_rural':>10}  "
          f"{'lvl1':^22}  {'lvl2':^45}  {'lvl3'}")
    print('  ' + '-' * 140)
    for lu, e in CLC_MAPPING.items():
        print(f"  {lu:>10}  {e.clc_code:>8}  {e.urban_rural:>10}  "
              f"{e.lvl1_label:^22}  {e.lvl2_label:^45}  {e.lvl3_label}")

    print()
    n_urban = sum(1 for e in CLC_MAPPING.values() if e.urban_rural == 'Urban')
    n_rural = sum(1 for e in CLC_MAPPING.values() if e.urban_rural == 'Rural')
    lvl1s = sorted(set(e.lvl1_label for e in CLC_MAPPING.values()))
    lvl2s = sorted(set(e.lvl2_label for e in CLC_MAPPING.values()))
    print(f"  Urban classes  : {n_urban}")
    print(f"  Rural classes  : {n_rural}")
    print(f"  LVL1 groups    : {len(lvl1s)}  → {lvl1s}")
    print(f"  LVL2 groups    : {len(lvl2s)}")
    print()
    print('  All assertions passed.')
