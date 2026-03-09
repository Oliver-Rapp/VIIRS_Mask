# VIIRS Cloud Mask Edge Analysis

This project analyzes the radiometric and textural properties of the **Marginal Ice Zone (MIZ)** and cloud boundaries using data from the **NOAA-21 VIIRS** instrument and the **NWC SAF Cloud Mask (CMA)** product.

It automatically processes satellite scenes to classify pixels based on their morphological context (e.g., "Ice touching Water", "Pure Interior Ice", "Cloud touching Water") and generates statistical reports and visual maps.

## 📂 Project Structure

```text
.
├── main.py               # Entry point for the application
├── lib/
│   ├── __init__.py
│   ├── config.py         # Configuration (Paths, Thresholds, Plot settings)
│   ├── core_logic.py     # Mathematical morphology & neighbor classification
│   ├── data_io.py        # NetCDF file handling (safe loading, scaling)
│   └── plotting.py       # Matplotlib/Cartopy map and report generation
└── README.md             # This file
```

## 🛠️ Prerequisites

The project requires **Python 3** and the following scientific libraries:

```bash
pip install numpy scipy matplotlib netCDF4 cartopy
```
## ⚙️ Configuration

All settings are managed in `lib/config.py`.

### Key Settings
*   **`DATA_DIR`**: Path to the folder containing the grouped NetCDF files (`S_NWC_viirs...`, `S_NWC_CMA...`, etc.).
*   **`OUTPUT_DIR`**: Destination for the PDF report and debug images.
*   **`NUM_WORKERS`**: Number of CPU cores to use. Set to `None` to use all available cores.
*   **`MAX_SOLAR_ZENITH`**: Default `85.0`. Pixels with a solar zenith angle higher than this (night/deep twilight) are discarded to ensure valid visible channel data.

### Geographic Crop
To save memory and focus the analysis, select a predefined region or add your own to `PREDEFINED_REGIONS`:
```python
SELECTED_REGION = 'barents_and_fram'  # 'svalbard' | 'barents_sea' | 'barents_and_fram'
USE_CROP = True
```

### MIZ Subset
A second set of histograms can be generated filtering pixels by T11 brightness temperature — targeting the temperature range characteristic of thin or melting sea ice:
```python
ENABLE_MIZ_HISTOGRAMS = True
MIZ_T11_RANGE = (268.5, 271.5)  # Kelvin
```

### Satellite Zenith Angle Analysis
To investigate whether the satellite viewing angle affects brightness temperature differences, enable the satzen analysis section:
```python
ENABLE_SATZEN_ANALYSIS = True
```

The relevant settings are:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `SATZEN_ANALYSIS_CLASSES` | `['IN', 'WN']` | Which pixel classes to include |
| `SATZEN_RANGES` | 3 ranges | List of `(min°, max°, label)` tuples defining the viewing angle bins |
| `SATZEN_DIFF_VARS` | `['diff1','diff2','diff3']` | Which brightness temperature differences to analyse |
| `SATZEN_PLOT_MODE` | `'both'` | `'histogram'` — density curves per range; `'scatter'` — 2D heatmap; `'both'` |
| `SATZEN_RANGE_COLORS` | blue/orange/green | One matplotlib colour per satzen range |

The default ranges split the VIIRS swath into three equal-area thirds:
*   **0–36.2°** — near-nadir
*   **36.2–52.5°** — mid-swath
*   **52.5–70°** — outer swath

## 🚀 Usage

Run the main script from the project root:

```bash
python main.py
```

1.  The script scans `DATA_DIR` for valid groups of 4 files (L1B, CMA, Geo, Texture).
2.  It processes scenes in parallel (unless `NUM_WORKERS=1`).
3.  It generates a **PDF Report** in `OUTPUT_DIR`.
4.  It generates **Debug Maps** (PNGs) in `OUTPUT_DIR/debug_maps/<TIMESTAMP>/`.

## 🔬 Class Definitions

The core logic uses **Binary Morphological Dilation** (3x3 structuring element) to identify how pixels interact with their neighbors.

### Input Data
The script relies on the **NWC SAF Cloud Mask Extended (`cma_extended`)** flags:
*   **0:** Cloud Free (Used for Water, filtered by Land Use).
*   **1 & 2:** Cloudy / Contaminated (Used for Cloud).
*   **3:** Snow/Ice (Used for Ice).

### Classification Logic
Pixels are classified into **Pure Interiors**, **Edge Neighbors**, or **Mixed** states.

#### 1. Interior Classes (Pure)
Pixels that are strictly one class and have **zero** neighbors of any other class within a 3-pixel radius.
*   **`IN` (Interior Ice):** Solid pack ice.
*   **`WN` (Interior Water):** Open ocean.
*   **`CN` (Interior Cloud):** Deep cloud deck.

#### 2. Neighbor Classes (Edges)
Pixels of one class that are $N$ steps away from a different class.
*   *Notation:* `[Class][Degree]n[Neighbor]` (e.g., `InW` = Ice 1st neighbor to Water).

| Notation | Class | Distance | Neighboring | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`InW`** | Ice | 1 px | Water | The physical Ice Edge. |
| **`I2nW`** | Ice | 2 px | Water | Just inside the Ice Edge. |
| **`I3nW`** | Ice | 3 px | Water | Further inside. |
| **`WnI`** | Water | 1 px | Ice | The water touching the ice. |
| **`InC`** | Ice | 1 px | Cloud | Ice obscured/shadowed by cloud edges. |
| **`WnC`** | Water | 1 px | Cloud | Water obscured/shadowed by cloud edges. |

*Note: Cloud neighbors (`CnI`, `CnW`) are calculated to define the pure `CN` class but are generally excluded from surface analysis.*

#### 3. Mixed Class
*   **`Mixed`**: Any pixel that satisfies **two or more** neighbor conditions simultaneously.
    *   *Example:* An ice pixel touching both Water and Cloud is `Mixed`.
    *   These pixels are isolated to prevent "dirty" data from skewing the pure edge statistics.

## 📊 Outputs

### PDF Report (`VIIRS_Report_<region>_YYYYMMDD_HHMMSS.pdf`)
*   **Page 1:** Run metadata (including active configuration) and a geographic heatmap showing scene coverage density.
*   **Part 1 — Standard histograms:** Density plots for all variables (T11, diffs, textures, angles) across all pixel classes.
*   **Part 2 — MIZ subset** *(if enabled)*: Same histograms restricted to pixels within `MIZ_T11_RANGE`.
*   **Part 3 — Satzen analysis** *(if enabled)*: For `IN` and `WN` pixels, brightness temperature differences (diff1/diff2/diff3) broken down by satellite viewing angle range.
    *   **Histogram mode:** One density curve per satzen range, colour-coded, for each diff variable and class.
    *   **Scatter mode:** 2D log-scale heatmap with satellite zenith angle on the X-axis and diff on the Y-axis; range boundaries marked with dashed lines.
    *   Both all-data and MIZ-subset versions are produced.

### Debug Maps
Generated for every scene (up to `NUM_DEBUG_MAPS`) to verify logic visually:
1.  **`01_base_class.png`**: Raw NWC SAF classification (Ice/Water/Cloud/Land).
2.  **`02_combined_analysis.png`**: A stacked visualization showing:
    *   **Background:** Interior Classes (Dark Violet Ice, Blue Water, Grey Cloud).
    *   **Midground:** 3rd $\to$ 2nd $\to$ 1st Neighbors.
    *   **Foreground:** Mixed pixels (Deep Pink).
3.  **`05_miz_diagnostic.png`** *(if MIZ enabled)*: Base classification with MIZ-range pixels highlighted in gold.

## 📝 Notes on Edge Cases
*   **Coastlines:** Clouds over land are masked out before neighbor calculation. This prevents "False Edges" where Ocean touches a Cloud-over-Land.
*   **Land Interaction:** Ice touching Land is **not** considered an edge (neighbors are only calculated against Water and Cloud).
*   **Fill Values:** The data loader automatically handles NetCDF `_FillValue` (e.g., -32767) by converting them to `NaN` to prevent histogram corruption.