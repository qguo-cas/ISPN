# ISPN: Interactive Self-Supervised Pansharpening Network

This repository contains the pansharpening research code associated with the following paper, with PyTorch workflows for training, per-epoch validation, and standalone validation:

> Q. Guo, Q. Zhou, Y. Tian and L. Yu, “Interactive Self-Supervised Pansharpening Network Guided by Joint Satellite Data and Intermediate Loss,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 19, pp. 19289–19306, 2026, doi: 10.1109/JSTARS.2026.3699231.

[Paper DOI](https://doi.org/10.1109/JSTARS.2026.3699231) · [Author Manuscript and Abstract](https://www.researchgate.net/publication/405779320_Interactive_Self-Supervised_Pansharpening_Network_Guided_by_Joint_Satellite_Data_and_Intermediate_Loss)

**Keywords:** Pansharpening, remote sensing, self-supervised learning, interactive feature learning, joint satellite data, intermediate loss, image enhancement.


## Method Overview

Pansharpening combines complementary information from high-resolution panchromatic (PAN) images and low-resolution multispectral (MS) images to produce high-resolution multispectral images. ISPN enables feature interaction among spatial, spectral, and prior streams, constructs prior images using inter-pixel correlations (IPC), and incorporates intermediate losses and joint satellite data training. See the [author manuscript](https://www.researchgate.net/publication/405779320_Interactive_Self-Supervised_Pansharpening_Network_Guided_by_Joint_Satellite_Data_and_Intermediate_Loss) for the method description. The implementation details below refer to the current repository.

| Method Component | Current Implementation |
| --- | --- |
| Spatial feature extraction and injection | `pan_encoder` and `SpatialFeatureBlock` extract PAN features that interact with the spectral stream. |
| Spectral feature extraction | `ms_encoder` and `SpectralFeatureBlock` extract spectral features from upsampled MS images (UPMS). |
| Prior guidance | `prior_encoder` and `JointFeatureBlock` extract features from precomputed prior images (PRIOR) and inject them into the fusion process. |
| Stream supervision | The spatial and spectral streams reconstruct PAN and MS images, supervised by UIQI and SAM losses. Fusion residuals are constrained by a KL loss. |
| Joint data training | Preprocessed samples from different satellites can share a dataset directory. The current loader samples individual images without grouping them by satellite source. |

## Implementation Scope and Reproducibility

This version includes configurable paths, training with all spectral bands, and standalone validation as part of the preparation for public release. To reproduce the paper's experiments, prepare the data and prior images according to its experimental settings:

- **Prior images:** The training entry point reads precomputed PRIOR images. The paper's IPC prior generation pipeline is not included; prepare these inputs using the method described in the paper.
- **Intermediate losses:** The current entry point uses the four outputs of `fusion_net.py` and a shared self-supervised objective to constrain spatial and spectral reconstructions. Additional stage-output variants and loss aggregation across stages are not included in this version.
- **Joint satellite data:** Users must prepare the data sources, scene splits, and preprocessing. Each model requires a consistent number of MS channels, and images within a batch must have matching dimensions. Band alignment and radiometric correction across sensors are not performed automatically.
- **Results:** Synthetic-data checks verify program behavior. The paper's real-data metrics and ablation results have not been re-evaluated with this revised version; refer to the paper for reported results.

`dataload/` contains only `dataset.py` and the package entry point `__init__.py`, used for training and validation. Import the dataset with `from dataload import FusionDataset`.

## Citation

If this project is useful for your research, please cite the following paper:

```bibtex
@article{guo2026ispn,
  author  = {Guo, Q. and Zhou, Q. and Tian, Y. and Yu, L.},
  title   = {Interactive Self-Supervised Pansharpening Network Guided by Joint Satellite Data and Intermediate Loss},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  volume  = {19},
  pages   = {19289--19306},
  doi     = {10.1109/JSTARS.2026.3699231},
  url     = {https://doi.org/10.1109/JSTARS.2026.3699231}
}
```

## Release Notes

`.gitignore` excludes data, model weights, output directories, and Python caches. A code license file has not yet been added; include the chosen project license before publication.
