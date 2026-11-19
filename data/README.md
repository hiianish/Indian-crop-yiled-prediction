# Data acquisition and preparation

## Starter panel currently in this workspace

`data/raw/crop_yield_dataset.csv` was downloaded from [sid09008mane/crop-yield](https://github.com/sid09008mane/crop-yield), with its [dataset notes](https://github.com/sid09008mane/crop-yield/blob/main/DATASET_README.md). The source documents ICRISAT District Level Database (yield, fertilizer, soil and rainfall), NASA POWER weather, and FAOSTAT pesticide inputs. It contains 198,369 district-year-crop rows from 1966–2017. `python -m src.prepare_data` selects Rice, Wheat, Maize, Cotton and Sugarcane, normalizes the geographic keys, and converts the source's yield from kg/ha to tonnes/ha. Raw file is excluded from git due to its size; rerun the source download if needed.

Source caveats: annual weather is not crop-season weather; early temperature/humidity values are climatology backfills; pesticide values are national rather than district observations; and soil pH is taxonomy-derived. Those limitations need to appear in the final report. This dataset supplies yield directly, not production and area, so it is a practical starting panel rather than the final preferred official production/area foundation. Keep attribution to ICRISAT, NASA POWER, FAOSTAT and the dataset compiler in any derived publication.

CropBench is another promising dataset named in the brief, but its Hugging Face repository currently requires account access approval and acceptance of CC BY-NC-SA terms. The open starter panel was used to make progress without accepting those terms on the user's behalf.

Keep downloaded source files in `data/raw/` and only publish data you are permitted to redistribute. Record source URL, download date, license/terms, units, geographic coding system and transformations in a provenance log before analysis.

Suggested sources in the project brief are DES (crop area/production), IMD gridded rainfall and ISRIC SoilGrids. Check the source's current access terms and spatial/temporal resolution before using it. District boundaries and names change over time; retain source identifiers and a documented crosswalk rather than joining by unreviewed names alone. Soil values should be aggregated to the district/cropland footprint with a documented method. Rainfall should cover the crop's relevant season, not an arbitrary calendar window.

Prepare one record per `state,district,crop,season,year`. Required numeric fields: `production_tonnes`, `area_hectares`, both positive. Optional numeric weather/soil features are included in the corresponding feature ablation. Normalize spelling, units, season labels and district identifiers. Resolve duplicates at source level; do not average conflicting production records without a defensible rule. Missing feature values are imputed within each training fold by the model pipeline.

Do not include production or area as predictors: yield is computed from them and their inclusion would leak the target. The historical feature is built by sorting each group and shifting yield one row, so it uses the previous *available* observation; inspect gaps and consider requiring consecutive years in the final analysis.
