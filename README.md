# TumorMassEffect
Pytorch implementation of the displacement model from "Generation of Annotated Brain Tumor MRIs with Tumor-induced Tissue Deformations for Training and Assessment of Neural Networks".

## Prerequisites 
- Linux or MacOS
- Python 3
- PyTorch>0.4
- NVIDIA GPU + CUDA CuDNN
- MATLAB (Optional, but you will need to write a new edge extraction function)

## Getting started
### Instalation
- Clone this repo
- Install the requirements

### Datasets
Here we use simple threshhold-based shape-describing deformed and non-deformed versions of images. See Folder "images" for examples. 


## Citation
This work has been accepted to the MICCAI 2020. If you use this code, please cite as follows:

```
@inproceedings{MEGAN,
	title = {Generation of Annotated Brain Tumor MRIs with Tumor-induced Tissue Deformations for Training and Assessment of Neural Networks},
	booktitle = {International Conference on  Medical Image Computing and Computer Assisted Intervention, MICCAI 2020},
	year = {In Press},
	author = {Uzunova, Hristina and Ehrhardt, Jan and Handels, Heinz}
}
```
