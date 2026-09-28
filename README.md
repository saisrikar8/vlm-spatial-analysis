# VLM Spatial Analysis

Investigating how well vision-language models interpret scientific maps and reason about numerical values and uncertainty. Initial experiments use figures from the [ILAMB CMIP6 historical report](https://www.ilamb.org/CMIP6/historical/).

## Current structure

- `get_ilamb_data.py` — Downloads ILAMB maps and their colorbars, with category filters and seeded random sampling. Saves source metadata and an HTML preview.

## Usage

Requires Python 3.9+. No additional packages are needed.

```bash
# List available categories and datasets
python3 get_ilamb_data.py --list

# Download a reproducible sample of 10 maps
python3 get_ilamb_data.py --random --seed 42 --count 10

# See all options
python3 get_ilamb_data.py --help
```

Outputs are saved to `ilamb_maps/` by default. Open `ilamb_maps/preview.html` to inspect them.

## Next steps

Add VLM inference and evaluation against reference data.
