# Medical-image-reconstruction-and-synthesis

## Paper:
Please see: A Generalized Dual-Domain Generative Framework with Hierarchical Consistency for Medical Image Reconstruction and Synthesis (Communications Engineering 2023)

Paper Link: https://doi.org/10.1038/s44172-023-00121-z

## Introduction:
This is the PyTorch implementation for low-dose PET reconstruction and PET-CT synthesis.

![Image](https://github.com/ZhangJD-ong/Iterative-Cycle-consistent-Semi-supervised-Learning-for-fibroglandular-tissue-segmentation/blob/main/img/Framework.png)

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
./data/Datasets
├─test.txt
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
├─'1_1'
├─'1_2'
├─'1_3'
...
├─'1_21'
├─'2_1'
...
```

### Whole Breast Segmentation Model
* The whole breast segmentation process is required to locate the breast ROI first.
* Partial images and whole breast annotations are available at: https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation

### Tumor Segmentation Model
* The tumor segmentation process is required to remove tumor enhancement for accurate BPE (Background Parenchymal Enhancement) quantification.
* A well-designed tumor segmentation assistant is available at: https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation

## Citation
If you find the code useful, please consider citing the following papers:
* Zhang et al., Breast Fibroglandular Tissue Segmentation for Automated BPE Quantification with Iterative Cycle-consistent Semi-supervised Learning, IEEE Transactions on Medical Imaging (2023), https://doi.org/10.1109/TMI.2023.3319646
* Zhang et al., A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework, Patterns (2023), https://doi.org/10.1016/j.patter.2023.100826
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001






The code is available at https://drive.google.com/drive/folders/1zwQkCnctDeEh60hnkDDqROaRcRz7ycr8?usp=sharing
This code is developed for medical image reconstruction and synthesis based on proposed framework.
We include two tasks in the file, i.e., low-dose PET reconstruction and PET-CT synthesis.
Small dataset are provided to demo the code.

System requiements:
We need system intalled Pytorch and ODL package.

Training:
In each tasak, there are three stage as mentioned in the paper. User needs to train each model seperatively. We also provide our well-trained models.


Testing:
Users need to copy the best model to the '.\Test\Saved_Model', and run test.py to inference. 
We also provide the well-trained model in the folder.
User can download our provided data and model. By filling the right path of data, user can get test resutls in '.\Test\Saved_Model' by simply run 'test.py'.

