# Python Algorithmic Implementations

This directory contains the Python 3.11 implementation of the self-supervised end-to-end framework.

* **01_preprocessing_hdf5.py**: Resizes scans to 224x224 px and performs z-standardisation.
* **02_simclr_pretraining.py**: Implements SimCLR v1 with a ResNet-50 backbone to learn feature representations.
* **03_manifold_validation.py**: Executes UMAP dimensionality reduction and Grad-CAM heatmap generation.
* **04_unet_refinement.py**: Integrates SSL weights and deterministic pseudo-labels via knowledge fusion.
* **05_mesh_reconstruction.py**: Performs surface extraction using the marching cubes algorithm.
