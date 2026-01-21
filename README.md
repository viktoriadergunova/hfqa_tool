# Heat Flow Quality Analysis Tool (hfqa_tool)
`hfqa_tool` is a Python package for validating heat-flow data against the IHFC Global Heat Flow Database (GHFDB) schema and assessing data quality according to International Heat Flow Commission standards, including quality scoring based on the methodology described in doi:10.1016/j.tecto.2023.229976.

## Table of contents 
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Vocabulary Validation](#vocabulary-validation)
- [Quality Scoring](#quality-scoring)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)


## Overview

`Vocabulary Validation` ensures that data entries comply with the GHFDB database schema by verifying required fields, value ranges, allowed vocabularies, multi-value fields, conditional rules, and by reporting, field-specific errors.

`Quality Score` calculates U-scores (uncertainty quantification, U1-U4), M-scores (methodology assessment based on measurement techniques, M1-M4), and P-flags (perturbations effect such as topographic, paleoclimatic etc. )


## Installation:

### Prerequisites
- Python 3.8 or higher
- pip, conda package manager

### Install from Github 
```bash
git clone https://github.com/viktoriadergunova/hfqa_tool.git
cd hfqa_tool
pip install -r requirements.txt # install dependencies
```
### Dependencies
```
pandas>=2.0
pyarrow
openpyxl
pyyaml
```
### Data Preparation

Your Excel file must follow offical GHFDB structure: #TODO add image
```
Row 1-7:   Metadata rows (configurable)
Row 8:     Column headers (ID, Level, Obligation, P1, P2, P3, ...)
Row 9+:    Data rows
```


## Quick Start - Vocabulary Validation
```bash
python main.py --vocab-check --input your_data.xlsx 
```
**Parameters:**
- `--vocab-check`: Run vocabulary validation mode
- `--input`: Path to your Excel file
- `--out-json`: Option to get .json files, saved in /voc
- `--out-excel`: Option to get .xlsx files (default), saved in /voc folder
- `--meta-rows`: Number of metadata rows at the top of your Excel sheet (default: 7)
- `--sheet`: Sheet index to validate (default: 1)
- 
**Full example with all options:**
```bash
python main.py --input your_data.xlsx --vocab-check --out-json --sheet 1 --meta-rows 7 --debug-prefix debug_run 
```
### Understanding the Output 
#### JSON Report Structure
```json
{
  "violations": [
    {
      "row_excel_number": 8,
      "column": "P2",
      "site_name": "[Site-001]",
      "flag": "missing",
      "comment": "HF Uncertainty"
    }
  ],
  "summary": {
    "total_rows_in_sheet": 1000,
    "data_rows_validated": 993,
    "rows_with_any_error": 45,
    "functional_error_count": 78,
    "conditional_error_count": 12,
    "runtime_seconds": 2.34
  }
}
```
## Excel Output

The Excel output file contains all original input data and two additional columns that describe validation results and row classification:

- **`row_type`** – indicates whether the row represents metadata or data
- **`validation_comments`** – contains column dependent error descriptions
- Rows without validation issues have an empty `validation_comments` field.


| … | P1 | P2 | P3 | row_type | validation_comments |
|---|----|----|----|----------|---------------------|
| … | 65.5 |    | Site-001 | data | `[MISSING] C47 (Thermal conductivity number): Required field is empty; [INVALID VALUE] C26 (Stratigraphy): Value 'late paleozoic orogeny' is not in allowed list` |
| … | 72.3 | 5.2 | Site-002 | data |  |


### Validation Types

#### 1. Functional Validation
Basic data quality checks performed on all data:

**Missing Value Checks**
- Validates that all mandatory (M) fields contain values
- Example: `P2` (HF Uncertainty) must not be empty

**Range Validation**
- Ensures numeric values fall within acceptable bounds
- Example: Latitude (`P4`) must be between -90.0 and 90.0

**Allowed Value Validation**
- Checks that values match controlled vocabularies
- Example: `P7` (Location Type) must be one of `[Onshore (continental)]`, `[Offshore (marine)]`, etc.

#### 2. Conditional Validation
Context-dependent rules based on other field values:

**Method-Specific Requirements**
- If exploration method (`P12`) is `[Probing (...)]`, then probe tilt (`C23`) is mandatory
- If thermal conductivity measurement type is specified, corresponding method columns must be valid

**Cross-Field Logic**
- Temperature measurement method restrictions based on measurement count
- Thermal conductivity source restrictions based on location type

### Error Categories

| Category | Flag Suffix | Description | Example |
|----------|-------------|-------------|---------|
| Missing | `__missing` | Required field is empty | `P2__missing` |
| Range | `__out_of_range` | Value outside valid range | `P4__out_of_range` |
| Invalid | `__invalid` | Value not in vocabulary | `P7__invalid` |
| Conditional | `__cond_*` | Context-dependent rule violated | `C23__cond_probing_requires_c23` |




## Quality Scoring

The quality scoring module evaluates heat-flow data according to IHFC standards, providing a comprehensive assessment through three independent components:

### Quick Start - Quality Scoring
```bash
python main.py --quality-score --input your_data.xlsx 
```

**Parameters:**
- `--quality-score`: Run quality scoring mode
- `--input`: Path to your Excel file
- `--out-json`: Option to get .json files, saved in /voc
- `--out-excel`: Option to get .xlsx files (default), saved in /qc folder
- `--meta-rows`: Number of metadata rows (default: 7)
- `--sheet`: Sheet index to process (default: 0)
- `--debug-prefix`: Optional prefix for debug output files

**Full example with all options:**
```bash
python main.py --input your_data.xlsx --quality-score --out-json --sheet 1 --meta-rows 7 --debug-prefix debug_run 
```

### Quality Score Components

The quality assessment consists of three independent scores that are combined into a final quality code:

#### 1. U-Score (Uncertainty Quantification)
Evaluates the numerical uncertainty of heat-flow determinations based on the coefficient of variation (COV):

| Score | COV Range | Description |
|-------|-----------|-------------|
| U1 | < 5% | Excellent - Very low uncertainty |
| U2 | 5-15% | Good - Low uncertainty |
| U3 | 15-25% | Acceptable - Moderate uncertainty |
| U4 | > 25% | Poor - High uncertainty |
| Ux | N/A | Not determined / missing data |

**Calculation:**
```
COV(%) = (Heat Flow Uncertainty / Heat Flow Mean) × 100
```

#### 2. M-Score (Methodological Quality)
Assesses the quality of measurement methods for both temperature gradient and thermal conductivity determinations. The evaluation differs for borehole/mine data versus probe-sensing data.

**Borehole/Mine Data:**
- Evaluates temperature method (equilibrium vs. perturbed measurements)
- Assesses thermal conductivity source and measurement conditions
- Considers number of measurements and in-situ conditions

**Probe-Sensing Data:**
- Evaluates penetration depth and number of temperature points
- Assesses water depth and probe tilt
- Considers thermal conductivity measurement location and method

| Score | Quality Range | Description |
|-------|---------------|-------------|
| M1 | ≥ 0.75 | Excellent methodology |
| M2 | 0.50-0.74 | Good methodology |
| M3 | 0.25-0.49 | Acceptable methodology |
| M4 | < 0.25 | Poor methodology |
| M*x | Any | Incomplete metadata (x-suffix indicates missing data) |

**Example M-Score Calculation:**
Starting from base value 1.0, penalties are applied based on:
- Temperature measurement type and quality
- Thermal conductivity source and measurement conditions
- Number of measurements
- Pressure/temperature corrections

#### 3. P-Flags (Perturbation Effects)
A 7-character code indicating the status of environmental perturbations that may affect heat-flow measurements:

**Format:** `SxxxCxh` (example)

Each position represents a specific perturbation:
1. **S/s** - Sedimentation effects
2. **E/e** - Erosion effects
3. **T/t** - Topography/Bathymetry effects
4. **P/p** - Paleoclimate effects
5. **V/v** - Surface/Bottom water temperature variation
6. **C/c** - Convection/Fluid flow
7. **R/r** - Heat refraction effects

**Character meanings:**
- **Uppercase letter** (e.g., `S`, `E`) = Present and **corrected**
- **Lowercase letter** (e.g., `s`, `e`) = Present but **not corrected**
- **Uppercase `X`** = Present but **not significant**
- **Lowercase `x`** = **Not recognized** or not present
- **Dash `-`** = Unspecified/missing data

**Examples:**
- `SxxxCxh` = Sedimentation corrected, convection not recognized, heat refraction present but not corrected
- `SETPV--` = Multiple effects corrected, convection and heat refraction unspecified
- `-------` = All perturbations unspecified

### Combined Quality Score

The final quality score combines all three components:

**Format:** `U[1-4]M[1-4][x].P-FLAGS`

**Examples:**
- `U1M1.SxxxCxh` - Excellent uncertainty, excellent methodology, specific perturbations
- `U2M3x.-------` - Good uncertainty, acceptable methodology with missing metadata, unspecified perturbations
- `U3M2.SETPVXR` - Acceptable uncertainty, good methodology, multiple perturbations addressed

### Output Files

#### JSON Quality Report
```json
{
  "summary": {
    "total_rows_processed": 150,
    "quality_distribution": {
      "U1": 45,
      "U2": 67,
      "U3": 28,
      "U4": 10,
      "M1": 34,
      "M2": 56,
      "M3": 42,
      "M4": 18
    },
    "runtime_seconds": 3.42
  },
  "data": [
    {
      "row": 9,
      "site_name": "Site-001",
      "quality_U": "U1",
      "quality_M": "M2",
      "quality_P": "SxxxCxh",
      "quality_combined": "U1M2.SxxxCxh"
    }
  ]
}
```

#### Excel Output
The Excel output includes all original data plus an additional column:
- **`quality_score`** - Combined quality code (e.g., `U1M2.SxxxCxh`)

### Quality Score Schema

Quality scoring behavior is defined in `quality_score_schema.yaml`:

**U-Score Configuration:**
- Threshold values for uncertainty classes
- Calculation method (coefficient of variation)

**M-Score Configuration:**
- Penalties for different measurement methods
- Borehole vs. probe-sensing evaluation criteria
- Required metadata fields

**P-Flags Configuration:**
- Perturbation field mappings (C13-C19)
- Encoding rules for each perturbation type
- Letter assignments for each effect

### Interpreting Results

**High-Quality Determinations:**
- U1 or U2 (low uncertainty)
- M1 or M2 (good methodology)
- Perturbations corrected (uppercase letters in P-flags)

**Use with Caution:**
- U3 or U4 (high uncertainty)
- M3 or M4 (questionable methodology)
- M-score with 'x' suffix (incomplete metadata)
- Uncorrected perturbations (lowercase letters in P-flags)

**Example Quality Assessment:**

| Score | Interpretation |
|-------|----------------|
| `U1M1.SxxxCxh` | **Excellent** - Low uncertainty, best methodology, sedimentation corrected |
| `U2M2.SETPVXR` | **Very Good** - Low uncertainty, good methodology, most perturbations addressed |
| `U3M3x.-------` | **Questionable** - Moderate uncertainty, acceptable but incomplete methodology, perturbations unknown |
| `U4M4.sePvch` | **Poor** - High uncertainty, poor methodology, uncorrected perturbations |


## Documentation

### Schema Files

The validation behavior is defined in YAML schema files:

- **`hf_schema.yaml`**: Main data structure and functional validation rules
- **`conditional_rules.yaml`**: Context-dependent validation logic
- **`quality_score_schema.yaml`**: Quality score calculation rules

### Column Reference

Full documentation of all data fields is available in the schema files. Key field groups:

**Parent Level (P1-P13)**: Site-level information
- Location coordinates, elevation, exploration method, etc.

**Child Level (C1-C49)**: Determination-level information
- Heat flow value, uncertainty, measurement methods, thermal conductivity, etc.

**Admin Level (A1-A8)**: Administrative metadata
- Reviewer information, geographic classification, etc.

## Testing

Run the built-in test suite to verify functionality:
```bash
python main.py --run-tests
```

This executes:
- Range validation tests
- Obligation/mandatory field tests
- Allowed value tests
- Conditional rule tests
- U-score calculation tests
- M-score calculation tests
- P-score calculation tests

## Contributing

We welcome contributions! Please see [ISSUEs.md](ISSUEs.md) 

## Citation 

If you use this tool in your research, please cite:
```bibtex
@software{chishti2025hfqa,
  author = {Chishti, Saman F. and Balkan-Pazvantoğlu, Elif and Norden, Ben and 
            Neumann, Florian and Elbarbary, Samah and Gross, Eskil S. and 
            Petrunin, Alexey G. and Fuchs, Sven},
  title = {Heat Flow Quality Analysis Toolbox (hfqa_tool)},
  year = {2025},
  publisher = {GFZ Data Services},
  doi = {10.5880/fidgeo.2025.043},
  url = {https://doi.org/10.5880/fidgeo.2025.043}
}
```

**Reference paper:**
```bibtex
@article{fuchs2023quality,
  title = {Quality-assurance of heat-flow data: The new structure and evaluation 
           scheme of the IHFC Global Heat Flow Database},
  author = {Fuchs, Sven and others},
  journal = {Tectonophysics},
  volume = {863},
  pages = {229976},
  year = {2023},
  doi = {10.1016/j.tecto.2023.229976}
}
```

## Authors

- Viktoria Dergunova (GFZ)
- Elif  Balkan-Pazvantoğlu (GFZ)
- Saman Firdaus Chishti (GFZ / University of Potsdam)
- Ben Norden (GFZ)
- Florian Neumann (MARUM, University of Bremen)
- Samah Elbarbary (GFZ)
- Eskil Salis Gross (GFZ)
- Alexey G. Petrunin (GFZ)
- Sven Fuchs (GFZ)

See [AUTHORs.md](AUTHORs.md) for detailed affiliations and contributions.

## License

This project is dual-licensed:

- **Source code**: [MIT License](license/MIT.txt)
- **Documentation and images**: [CC-BY-4.0](license/CC-BY-4.0.txt)

---

**Maintained by the Section Geoenergy, GFZ Helmholtz Centre for Geosciences**






