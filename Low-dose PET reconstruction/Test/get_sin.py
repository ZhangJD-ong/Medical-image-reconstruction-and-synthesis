import torch
import numpy as np
import SimpleITK as sitk
from dataset_lits_test import Test_Datasets
from Model.Projection_operator import Forward_projection,Backward_projection
from Model.model import ResUNet
from Model.Sin_model import ResNet
from torch.utils.data import DataLoader
import logger
import torch.nn as nn
from skimage.metrics import structural_similarity,normalized_root_mse,peak_signal_noise_ratio
import os
import cv2
import util
from common import get_psnr
from collections import OrderedDict

def normalization(img):
    out=(img - np.min(img))/(np.max(img) - np.min(img) + 0.000001 )
    return out

def test_result(task_id,modality):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    REAL = np.zeros([250,285,183]).astype(np.float32)
    FAKE = np.zeros([250, 285, 183]).astype(np.float32)


    sin_model_pet_ct = ResNet(1,1).to(device)
    sin_model_ct_pet = ResNet(1,1).to(device)
    FP = Forward_projection('astra_cuda').to(device)
    BP = Backward_projection('astra_cuda').to(device)


    ckpt = torch.load(os.path.join(r'C:\Users\user\3D Objects\PET2CT\Results', task_id, 'PETsin_CTsin.pth'),map_location= device)
    sin_model_pet_ct.load_state_dict(ckpt['model'])
    ckpt = torch.load(os.path.join(r'C:\Users\user\3D Objects\PET2CT\Results', task_id, 'CTsin_PETsin.pth'),map_location= device)
    sin_model_ct_pet.load_state_dict(ckpt['model'])

    num = 0
    datasets = Test_Datasets(r'C:\Users\user\3D Objects\PET2CT\Data')
    for img_dataset, file_idx in datasets:
        PET, CT= img_dataset[0].to(device), img_dataset[1].to(device)
        PET,CT = PET.unsqueeze(0).type(torch.FloatTensor).to(device), CT.unsqueeze(0).type(torch.FloatTensor).to(device)
        if modality == 'CT':
            sin = FP(PET).type(torch.FloatTensor).to(device)
            sin = sin_model_pet_ct(sin)
            a = FP(CT).type(torch.FloatTensor).to(device)
            Pred = sin.cpu().detach().numpy().squeeze()
            Real = a.cpu().detach().numpy().squeeze()

        elif modality == 'PET':
            sin = FP(CT).type(torch.FloatTensor).to(device)
            sin = sin_model_ct_pet(sin)
            a = FP(PET).type(torch.FloatTensor).to(device)
            Pred = sin.cpu().detach().numpy().squeeze()
            Real = a.cpu().detach().numpy().squeeze()

        else:
            print('jiliguala')




        FAKE[num,:,:] = Pred
        REAL[num,:,:] = Real
        num +=1


    aa = sitk.GetImageFromArray(FAKE)
    sitk.WriteImage(aa,os.path.join(r'C:\Users\user\3D Objects\PET2CT\Results',task_id,'Fake_sin_'+modality+'.nii.gz'))
    bb = sitk.GetImageFromArray(REAL)
    sitk.WriteImage(bb,os.path.join(r'C:\Users\user\3D Objects\PET2CT\Results',task_id,'Real_sin_'+modality+'.nii.gz'))


if __name__ == '__main__':
    test_result('MultiCycle','CT')
