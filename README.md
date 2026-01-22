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
To save memory and focus the analysis (e.g., on Svalbard), use the crop settings:
```python
USE_CROP = True
CROP_BOUNDS = {
    'lat_min': 74.0, 'lat_max': 81.0,
    'lon_min': 10.0, 'lon_max': 35.0
}
```

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

### PDF Report (`VIIRS_Report_YYYYMMDD_HHMMSS.pdf`)
*   **Page 1:** Run metadata and a geographic heatmap showing scene coverage density.
*   **Page 2+:** Histograms comparing radiometric properties (T11, T11-T12, Textures) across the different classes defined above.

### Debug Maps
Generated for every scene (up to `NUM_DEBUG_MAPS`) to verify logic visually:
1.  **`01_base_class.png`**: Raw NWC SAF classification (Ice/Water/Cloud/Land).
2.  **`02_combined_analysis.png`**: A stacked visualization showing:
    *   **Background:** Interior Classes (Dark Violet Ice, Blue Water, Grey Cloud).
    *   **Midground:** 3rd $\to$ 2nd $\to$ 1st Neighbors.
    *   **Foreground:** Mixed pixels (Deep Pink).

## 📝 Notes on Edge Cases
*   **Coastlines:** Clouds over land are masked out before neighbor calculation. This prevents "False Edges" where Ocean touches a Cloud-over-Land.
*   **Land Interaction:** Ice touching Land is **not** considered an edge (neighbors are only calculated against Water and Cloud).
*   **Fill Values:** The data loader automatically handles NetCDF `_FillValue` (e.g., -32767) by converting them to `NaN` to prevent histogram corruption.