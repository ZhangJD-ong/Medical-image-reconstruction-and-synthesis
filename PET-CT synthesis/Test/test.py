import torch
import numpy as np
import SimpleITK as sitk
from dataset_lits_test import Test_Datasets
#from Model.Projection_operator import Forward_projection,Backward_projection
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

def test_result(modality):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    REAL = np.zeros([40,128,128]).astype(np.float32)
    FAKE = np.zeros([40, 128, 128]).astype(np.float32)

    model = ResUNet(1, 1).to(device)
    if modality == 'CT':
        ckpt = torch.load(os.path.join(r'D:\AAAAAAA\Framework\Code\code for Low-dose PET reconstruction\PET-CT synthesis\Test\Saved_MODEL', 'PETimg_CTimg.pth'),map_location= device)
        model.load_state_dict(ckpt['model'])
    elif modality == 'PET':
        ckpt = torch.load(os.path.join(r'D:\AAAAAAA\Framework\Code\code for Low-dose PET reconstruction\PET-CT synthesis\Test\Saved_MODEL', 'CTimg_PETimg.pth'),map_location= device)
        model.load_state_dict(ckpt['model'])
    else:
        print('Error!')

    log_test = logger.Test_Logger(os.path.join(r'D:\AAAAAAA\Framework\Code\code for Low-dose PET reconstruction\PET-CT synthesis\Test\Saved_MODEL'), modality)
    save_result_path = os.path.join(r'D:\AAAAAAA\Framework\Code\code for Low-dose PET reconstruction\PET-CT synthesis\Test\Saved_MODEL', modality+'_Pictures')
    util.mkdir(save_result_path)
    num = 0
    datasets = Test_Datasets(r'D:\AAAAAAA\Framework\Code\code for Low-dose PET reconstruction\PET-CT synthesis\data')
    for img_dataset, file_idx in datasets:
        PET, CT= img_dataset[0].to(device), img_dataset[1].to(device)
        PET,CT = PET.unsqueeze(0).type(torch.FloatTensor).to(device), CT.unsqueeze(0).type(torch.FloatTensor).to(device)
        if modality == 'CT':
            Input_dex = PET
            Output_dex = CT
        elif modality == 'PET':
            Input_dex = CT
            Output_dex = PET
        else:
            print('Error')

        pred = model(Input_dex)

        Pred = pred.cpu().detach().numpy().squeeze()
        Real = Output_dex.cpu().detach().numpy().squeeze()

        FAKE[num,:,:] = Pred
        REAL[num,:,:] = Real
        num +=1
        NRMSE = normalized_root_mse(Real,Pred)
        PSNR = get_psnr(Real,Pred,1)
        SSIM = structural_similarity(Real,Pred)

        log_test.update(file_idx,OrderedDict({'PSNR': PSNR,'SSIM': SSIM,'NRMSE': NRMSE}))
        cv2.imwrite(os.path.join(save_result_path,file_idx+'_fake.png'),Pred*255)
        cv2.imwrite(os.path.join(save_result_path,file_idx+'_real.png'),Real*255)
    mkdir(os.path.join(r'D:\AAAAAAA\Framework\Code\code for Low-dose PET reconstruction\PET-CT synthesis\Test\Saved_MODEL','Results'))



if __name__ == '__main__':
    test_result('CT')
