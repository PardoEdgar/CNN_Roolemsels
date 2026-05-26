# Metrics from crossectional root images through trained CNN-segmentation models, Tropical Ecology Lab, 2026
## Universidad del Rosario

------

## Overview
This repository provides a reproducible workflow for Spatial Mapping from 3D models and Orthomosacs built in Agisoft Metashape.
The approach combines Python and R scrips using Metashape-API to extract internal locations and convert them to Real World position with transformation matrix.
We plotted the real world positions from tagged Coral and Poles and further edited the maps with Inkscape for better aesthetic visualization.
Here we also acquired positions of new possible tagged coral colonies using the Orthomosaics as a map and plotted them for further and easy search in fieldtrip.

---
## Contents
- `Images/`: Crossectional root segmentation Real, ROI and Masks images.
  - `ROIs_Real/`: Internal and Real World colonies and poles position datasets
  - `ROIs_Real/`: Internal and Real World colonies and poles position datasets
  - `Masks/`: Internal and Real World colonies and poles position datasets
    - `Xylem/`: Internal and Real World colonies and poles position datasets 
    - `Stele/`: Internal and Real World colonies and poles position datasets
    - `Aerenchyme/`: Internal and Real World colonies and poles position datasets
    - `Xylem/`: Internal and Real World colonies and poles position datasets
- `scripts/`: Python scripts for segmentation model and Graphical User Interface construction

---
## Reproducibility
All scripts conducted in Python and R are provided in sequential order:
 1. `Extract_data_colonies.py`
 2. `Mud_Map.R`
 3. `New_colonies_data_extraction.py`
 4. `Mud_Map_New_Colonies.R`
 5. `Mud_Map_New_Colonies_plus_size_table.R`
    
---------------
## Requirements
### Python
  - Torchvision
  - Albumentations
  - OpenCV
  - Tkinter
  - Matplotlib
  - Pandas

-------
## Data availability
All data and code required to reproduce the crossectional root area recognition presented here are included in this repository.

------------
## Author
Edgar Alejandro Pardo Sarmiento, Universidad del Rosario


--------
## License
The source code in this repository is licensed under the MIT License.
All data and figures are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
