# Data directory

**No real patient data is stored or tracked in this repository.**

In line with the project data-governance policy, raw, input, interim,
processed, and output data files are excluded by the top-level
[`.gitignore`](../.gitignore) and must never be committed or pushed to any
public remote. Only source code and non-sensitive schema metadata are public.

The cohort analysed in the manuscript is the **Charité PROM Baseline** dataset,
publicly available from Mendeley Data (DOI: `10.17632/wrhr5862cb.4`). It is
**not** redistributed here.

## What lives here

| Item | Tracked? | Notes |
|------|----------|-------|
| `codebook_crosswalk.csv` | yes | Variable schema only (names, roles, types, allowed values). Contains **no** patient records. |
| `example/generate_example.py` | yes | Generates a example cohort that reproduces the schema and plausible marginal distributions used by the pipeline. |
| `example/cohort.xlsx` | **no** | Generated locally on demand; git-ignored. |
| `raw/`, `input/`, `interim/`, `processed/`, `external/` | **no** | Reserved for private local data; git-ignored. |

## Running on example data (default)

```bash
python data/example/generate_example.py --n 600 --out data/example/cohort.xlsx
python run_pipeline.py            # PAPER5_RAW defaults to the example cohort
```

## Running on the real cohort

Download the dataset from Mendeley Data (DOI: `10.17632/wrhr5862cb.4`), place it
at a local path **outside** the tracked tree, and point the pipeline at it:

```bash
# Windows PowerShell
$env:PAPER5_RAW = "C:\path\to\Data_PROM_Baseline_updateCF.xlsx"; python run_pipeline.py
# bash
PAPER5_RAW=/path/to/Data_PROM_Baseline_updateCF.xlsx python run_pipeline.py
```

Verify with `git status` that no data file is staged before any commit or push.
