# VLM Spatial Analysis

Studying whether vision-language models preserve relationships among uncertain numerical readings of figures when answering different calculations. Initial data comes from the [ILAMB CMIP6 historical report](https://www.ilamb.org/CMIP6/historical/).

## Progress

- Downloaded 30 global `timeint` maps, their legends, and corresponding NetCDF files.
- Combined each map and legend into one RGB image and exported matching numerical arrays in manifest order.
- Verified all 30 pairs against the original image pixels, numerical values, and missing-data masks.
- VLM evaluation has not started; no experimental gap or method improvement has been established.

## Usage

`get_ilamb_data.py` supports category filters, seeded sampling, NetCDF downloads, and NumPy export. Downloading requires Python 3.9+ with no extra packages; exporting requires NumPy, Pillow, and netCDF4.

```bash
# List datasets or download a reproducible sample
python3 get_ilamb_data.py --list
python3 get_ilamb_data.py --random --seed 42 --count 30

# Add NetCDF files to an older image-only download
python3 get_ilamb_data.py --add-netcdf

# Export existing map–legend images and matching numerical arrays
python3 -m pip install numpy pillow netCDF4
python3 get_ilamb_data.py --align

python3 get_ilamb_data.py --help
```

## Outputs

Files are saved to `ilamb_maps/` by default (`--output DIR` to change it).

| File | Contents |
| --- | --- |
| `assets/`, `data/` | Original map/legend PNGs and shared NetCDF files |
| `catalog.json`, `manifest.json` | Discovered figures, selected sources, checksums, and download errors |
| `preview.html` | Downloaded map and legend preview |
| `images.npy` | Combined RGB images: `uint8`, currently `(30, 475, 750, 3)` |
| `numerical.npy` | Native numerical arrays flattened into rows: `float64`, currently `(30, 259200)` |
| `alignment.json` | Row mappings, original shapes, coordinates, units, selected variables, and image placement |

Matching row indices identify the same sample. Legends are centered below maps without resizing; image padding is white. Numerical padding and masked values are NaN. Station vectors and grids retain their native resolution and units. Alignment pairs images with source arrays; it does not register geographic cells to image pixels. Only global maps are currently supported by `--align`.

```python
import json
import numpy as np

images = np.load('ilamb_maps/images.npy')
values = np.load('ilamb_maps/numerical.npy')
with open('ilamb_maps/alignment.json') as f:
    records = json.load(f)['records']
i = 0
image = images[i]
numerical = values[i, :records[i]['length']].reshape(records[i]['shape'])
```

## Next step

Build a small evaluation pilot: a controlled example with known shared ambiguity, plus verified locations in real maps. Compare calculations from complete VLM readings with independently recombined readings, using the same numerical executor. Establish the gap before developing a new method.
