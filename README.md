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
python main.py --vocab-check \
  --input your_data.xlsx \
  --out-json validation_report.json \
  --out-excel validated_data.xlsx \
  --meta-rows 7
```
**Parameters:**
- `--vocab-check`: Run vocabulary validation mode
- `--input`: Path to your Excel file
- `--out-json`: Output path for JSON validation report
- `--out-excel`: Output path for Excel file with validation comments
- `--meta-rows`: Number of metadata rows at the top of your Excel sheet (default: 7)
- `--sheet`: Sheet index to validate (default: 0)

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
TO BE CONTINUED


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



