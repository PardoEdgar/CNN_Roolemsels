# Metrics from crossectional root images through trained CNN-segmentation models, Tropical Ecology Lab, 2026
## Universidad del Rosario, Colombia

------

## Overview
This repository provides a reproducible workflow for crossctional root images segmentation.
The approach uses Python to build Graphical user interface for root vessels segmentation from transversal sections in monocots native plants. Using convolutional neural networks (CNN) U-net deep learning and design of Graphical User Interface.

Built models are found in this drive folder 
https://drive.google.com/drive/folders/18fueIGTQ1Sue_zXBPNpd9VDGy7fLwvji?usp=sharing

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
