# Metrics from crossectional root images through trained CNN-segmentation models, Tropical Ecology Lab, 2026
## Universidad del Rosario

------

## Overview
This repository provides a reproducible workflow for crossctional root images segmentation.
The approach combines Python and R scrips using Metashape-API to extract internal locations and convert them to Real World position with transformation matrix.
We plotted the real world positions from tagged Coral and Poles and further edited the maps with Inkscape for better aesthetic visualization.
Here we also acquired positions of new possible tagged coral colonies using the Orthomosaics as a map and plotted them for further and easy search in fieldtrip.

---
## Contents
- `Images/`: Crossectional root segmentation Real, ROI and Masks images.
  - `Raw/`: Non-modified pictures
  - `ROIs_Real/`: Selected region of interest from images for segmetation
  - `Masks/`: Masks created from ROIs
    - `Total/`: Total Masks created from ROIs
    - `Stele/`: Stele Masks created from ROIs
    - `Aerenchyma/`: Aerenchyme Masks created from ROIs
    - `Xylem/`: Xylem Masks created from ROIs
- `scripts/`: Python scripts for segmentation model and Graphical User Interface construction

---
## Reproducibility
All scripts conducted in Python and R are provided in sequential order:
 1. `Segmentation_model_total.py`
 2. `Segmentation_model_aerenchyme.py`
 3. `Segmentation_model_xylem.py`
 4. `Segmentation_model_stele.py`
 5. `GUI_Construction.py`
    
---------------
## Requirements
### Python
  - Torchvision
  - Albumentations
  - OpenCV
  - Tkinter
  - Matplotlib
  - Pandas
  - pathlib
  - Os
  - Sys
  - PIL
  - Skimage
  - Threading
    
### FIji/ImageJ

-------
## Data availability
All data and code required to reproduce the crossectional root area recognition presented here are included in this repository.

------------
## Authors
Edgar Alejandro Pardo Sarmiento & Yessica Hoyos,
Universidad del Rosario

--------
## License
The source code in this repository is licensed under the MIT License.
All data and figures are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
