This repository includes selected modules adapted from repo IWD_graph_analysis (DOI:10.5281/zenodo.5015072), branch [workflow_implementation](https://github.com/trettelbach/IWD_graph_analysis/tree/workflow_implementation), under the terms of the LICENSE file.
Only the components required for the analyses in the study 
**Post-disturbance ice-wedge degradation in Alaskan tundra fire scars using space-for-time substitution remote sensing** 
Rettelbach, T.; Bader, J;, Groenke, B.; Helm, V.; Langer, M.; Freytag, J.-C.; Grosse, G. 
_in review for ERL_ 
are included here.


The workflow is implemented using [Nextflow](https://www.nextflow.io/) and can be run as
`nextflow run main.nf -with-docker fondahub/iwd:latest -with-report --version 3`


The repository is organized into several key components:

`main.nf`
The main Nextflow pipeline definition.
This file orchestrates all major analysis steps, defines execution logic, and coordinates data flow between processes.

`nextflow.config`
Configuration file specifying:
- default parameters
- resource requirements
- software environment configuration
- executor settings


### main Nextflow processes (*.nf)
`scripts/`
- `demToGraph.nf`: Reads trough locations from DEMs and converts them to graphs.
- `extractTroughTransects.nf`: Extracts transects along trough or channel features for geomorphic analysis.
- `graphToShapefile.nf`: Converts graph outputs into GIS-compatible shapefiles.
- `mergeAnalysisCSVs.nf`: Aggregates CSV outputs from multiple runs into one summary CSV.
- `networkAnalysis.nf`: Performs graph analysis on the extracted trough networks.
- `transectAnalysis.nf`: Computes metrics and statistics along extracted transects.

### Helper Directories and files
`bin/`
Contains the executable python scripts and utilities that support are called by the Nextflow processes (in `scripts/`).

`prep_scripts/`
Preprocessing scripts for preparing input datasets prior to running the main pipeline, as well as plotting scripts for the result analysis and visualization.

`nextflow_reports/`
Auto-generated workflow execution reports, trace files, and runtime statistics produced by Nextflow.

The repository is released under the MIT license.
See the LICENSE file for details.

Data at the base of this analysis can be found via: [https://doi.org/10.5281/zenodo.17866631](https://doi.org/10.5281/zenodo.17866631)


