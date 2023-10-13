# Medical-image-reconstruction-and-synthesis

## Paper:
Please see: A Generalized Dual-Domain Generative Framework with Hierarchical Consistency for Medical Image Reconstruction and Synthesis (Communications Engineering 2023)

Paper Link: https://doi.org/10.1038/s44172-023-00121-z

## Introduction:
This is the PyTorch implementation for low-dose PET reconstruction and PET-CT synthesis.

![Image](https://github.com/ZhangJD-ong/Iterative-Cycle-consistent-Semi-supervised-Learning-for-fibroglandular-tissue-segmentation/blob/main/img/Framework1.png)

## Requirements:
* python 3.10
* pytorch 1.12.1
* tensorboard 2.10.1
* simpleitk 2.1.1.1
* scipy 1.9.1
* odl

## Setup

### Dataset
* Paritial data are released in this project for debugging and testing, including paired low-dose/standard-dose PET images, and paired PET/CT images. 
* To test the reconstruction and synhtesis models, you need to put the data in ./data/Datasets/:

```
./data
├─test.txt
./data/Datasets
├─1
  ├─CT.nii.gz
  └─PET.nii.gz
├─2
  ├─CT.nii.gz
  └─PET.nii.gz
      ...
```
* The format of the test.txt is as follow：
```
./data/test.txt
├─'1_0'
├─'1_1'
├─'1_2'
...
├─'1_19'
├─'2_0'
...
```

### Well-trained Model
* The well trained model can be downloaded via: https://drive.google.com/drive/folders/1zwQkCnctDeEh60hnkDDqROaRcRz7ycr8?usp=sharing
* The well-trained model should be placed in ./Test/Saved_MODEL


## Citation
If you find the code useful, please consider citing the following papers:
* Zhang et al., A Generalized Dual-Domain Generative Framework with Hierarchical Consistency for Medical Image Reconstruction and Synthesis, Communications Engineering (2023), https://doi.org/10.1038/s44172-023-00121-z
* Zhang et al., Mapping in Cycles: Dual-Domain PET-CT Synthesis Framework with Cycle-Consistent Constraints, International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2022), https://doi.org/10.1007/978-3-031-16446-0_72






