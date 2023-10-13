import torch
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from dataset.dataset_lits_test import Test_Datasets
from Model.model import ResUNet
from Model.Sin_model import ResNet
from Model.Projection_operator import Forward_projection,Backward_projection
from torch.utils.data import DataLoader
from utils import logger,util
import torch.nn as nn
from utils.common import get_ssim, get_psnr,adjust_learning_rate
from skimage.metrics import normalized_root_mse
from utils.metrics import LossAverage
import os
from test import *
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


def train (train_dataloader,epoch):
    print("=======Epoch:{}===================================".format(epoch))

    PET_CT_Loss = LossAverage()
    PET_CT_img_supervise = LossAverage()
    PET_CT_img_self = LossAverage()
    PET_CT_sin_self = LossAverage()
    CT_PET_Loss = LossAverage()
    CT_PET_img_supervise = LossAverage()
    CT_PET_img_self = LossAverage()
    CT_PET_sin_self = LossAverage()


    CT_NRMSE = LossAverage()
    CT_PSNR = LossAverage()
    CT_SSIM = LossAverage()
    PET_NRMSE = LossAverage()
    PET_PSNR = LossAverage()
    PET_SSIM = LossAverage()


    sin_model_PET_CT.eval()
    sin_model_CT_PET.eval()

    for i, (PET,CT) in enumerate(train_dataloader):  # inner loop within one epoch
        ##main model update param
        PET,CT = PET.to(device),CT.to(device)

        #### PET TO CT
        img_model_CT_PET.eval()
        img_model_PET_CT.train()

        fake_CT = img_model_PET_CT(PET)
        fake_PET = img_model_CT_PET(fake_CT)

        CT_sin = FP(fake_CT).type(torch.FloatTensor).to(device)
        PET_sin = sin_model_CT_PET(CT_sin)
        fake_PET_sin = BP(PET_sin).type(torch.FloatTensor).to(device)


        PET_CT_loss1 = nn.L1Loss()(CT,fake_CT)  #image_loss_supervise
        PET_CT_loss2 = nn.L1Loss()(fake_PET,PET)  # image_loss_self
        PET_CT_loss3 = nn.L1Loss()(fake_PET_sin,PET)  #sin_loss_self
        PET_CT_loss = PET_CT_loss1+opt.lamada_PET[0]*PET_CT_loss2+opt.lamada_PET[1]*PET_CT_loss3

        optimizer_PET_CT.zero_grad()
        PET_CT_loss.backward()
        optimizer_PET_CT.step()
        adjust_learning_rate(optimizer_PET_CT,epoch,opt.lr_PET_CT,opt.step_PET_CT)


        #### CT TO PET
        img_model_CT_PET.train()
        img_model_PET_CT.eval()

        fake_PET = img_model_CT_PET(CT)
        fake_CT = img_model_PET_CT(fake_PET)

        PET_sin = FP(fake_PET).type(torch.FloatTensor).to(device)
        CT_sin = sin_model_PET_CT(PET_sin)
        fake_CT_sin = BP(CT_sin).type(torch.FloatTensor).to(device)


        CT_PET_loss1 = nn.L1Loss()(PET,fake_PET)  #image_loss_supervise
        CT_PET_loss2 = nn.L1Loss()(fake_CT,CT)  # image_loss_self
        CT_PET_loss3 = nn.L1Loss()(fake_CT_sin,CT)  #sin_loss_self
        CT_PET_loss = CT_PET_loss1+opt.lamada_CT[0]*CT_PET_loss2+opt.lamada_CT[1]*CT_PET_loss3

        optimizer_CT_PET.zero_grad()
        CT_PET_loss.backward()
        optimizer_CT_PET.step()
        adjust_learning_rate(optimizer_CT_PET,epoch,opt.lr_CT_PET,opt.step_CT_PET)


        fake_CT = img_model_PET_CT(PET)
        fake_PET = img_model_CT_PET(CT)

        Real_ct = CT.cpu().detach().numpy().squeeze()*255
        Fake_ct = fake_CT.cpu().detach().numpy().squeeze()*255
        Real_pet = PET.cpu().detach().numpy().squeeze()*255
        Fake_pet = fake_PET.cpu().detach().numpy().squeeze()*255

        ct_nrmse = normalized_root_mse(Real_ct, Fake_ct)
        ct_psnr = get_psnr(Real_ct, Fake_ct,255)
        ct_ssim = get_ssim(Real_ct, Fake_ct)
        pet_nrmse = normalized_root_mse(Real_pet, Fake_pet)
        pet_psnr = get_psnr(Real_pet, Fake_pet,255)
        pet_ssim = get_ssim(Real_pet, Fake_pet)

        CT_PET_Loss.update(CT_PET_loss.item(),1)
        CT_PET_img_supervise.update(CT_PET_loss1.item(),1)
        CT_PET_img_self.update(CT_PET_loss2.item(),1)
        CT_PET_sin_self.update(CT_PET_loss3.item(),1)
        PET_CT_Loss.update(PET_CT_loss.item(),1)
        PET_CT_img_supervise.update(PET_CT_loss1.item(),1)
        PET_CT_img_self.update(PET_CT_loss2.item(),1)
        PET_CT_sin_self.update(PET_CT_loss3.item(),1)

        CT_NRMSE.update(ct_nrmse,1)
        CT_PSNR.update(ct_psnr,1)
        CT_SSIM.update(ct_ssim,1)
        PET_NRMSE.update(pet_nrmse,1)
        PET_PSNR.update(pet_psnr,1)
        PET_SSIM.update(pet_ssim,1)

    return OrderedDict({'CT_PET_Loss': CT_PET_Loss.avg, 'PET_CT_loss': PET_CT_Loss.avg,
                        'CT_PSNR': CT_PSNR.avg,'CT_SSIM': CT_SSIM.avg,'CT_NRMSE': CT_NRMSE.avg,
                        'PET_PSNR': PET_PSNR.avg,'PET_SSIM': PET_SSIM.avg,'PET_NRMSE': PET_NRMSE.avg})


def val(val_dataloader):

    Loss = LossAverage()

    CT_NRMSE = LossAverage()
    CT_PSNR = LossAverage()
    CT_SSIM = LossAverage()
    PET_NRMSE = LossAverage()
    PET_PSNR = LossAverage()
    PET_SSIM = LossAverage()

    img_model_PET_CT.eval()
    img_model_CT_PET.eval()

    for i, (PET, CT) in enumerate(val_dataloader):  # inner loop within one epoch
        ##main model update param

        PET = PET.to(device)
        CT = CT.to(device)

        fake_CT = img_model_PET_CT(PET)
        fake_PET = img_model_CT_PET(CT)

        loss = nn.L1Loss()(PET,fake_PET) + nn.L1Loss()(CT,fake_CT)

        Real_ct = CT.cpu().detach().numpy().squeeze()*255
        Fake_ct = fake_CT.cpu().detach().numpy().squeeze()*255
        Real_pet = PET.cpu().detach().numpy().squeeze()*255
        Fake_pet = fake_PET.cpu().detach().numpy().squeeze()*255

        ct_nrmse = normalized_root_mse(Real_ct, Fake_ct)
        ct_psnr = get_psnr(Real_ct, Fake_ct,255)
        ct_ssim = get_ssim(Real_ct, Fake_ct)
        pet_nrmse = normalized_root_mse(Real_pet, Fake_pet)
        pet_psnr = get_psnr(Real_pet, Fake_pet,255)
        pet_ssim = get_ssim(Real_pet, Fake_pet)

        Loss.update(loss.item(), 1)
        CT_NRMSE.update(ct_nrmse,1)
        CT_PSNR.update(ct_psnr,1)
        CT_SSIM.update(ct_ssim,1)
        PET_NRMSE.update(pet_nrmse,1)
        PET_PSNR.update(pet_psnr,1)
        PET_SSIM.update(pet_ssim,1)

    return OrderedDict({'Val_Loss': Loss.avg,
                        'CT_PSNR': CT_PSNR.avg, 'CT_SSIM': CT_SSIM.avg, 'CT_NRMSE': CT_NRMSE.avg,
                        'PET_PSNR': PET_PSNR.avg, 'PET_SSIM': PET_SSIM.avg, 'PET_NRMSE': PET_NRMSE.avg})

if __name__ == '__main__':
    opt = Options_x().parse()   # get training options
    device = torch.device('cuda:'+opt.gpu_ids if torch.cuda.is_available() else "cpu")
    print(device)

    sin_model_PET_CT = ResNet(1,1).to(device)
    ckpt = torch.load('./Pretrained_model/L_Ssin.pth',map_location=device)
    #ckpt = torch.load('./Multistage13/PETsin_CTsin.pth', map_location=device)
    sin_model_PET_CT.load_state_dict(ckpt['model'])

    sin_model_CT_PET = ResNet(1,1).to(device)
    ckpt = torch.load('./Pretrained_model/S_Lsin.pth',map_location=device)
    #ckpt = torch.load('./Multistage13/CTsin_PETsin.pth', map_location=device)
    sin_model_CT_PET.load_state_dict(ckpt['model'])

    img_model_PET_CT = ResUNet(1,1).to(device)
    ckpt = torch.load('./Pretrained_model/L_Simg.pth',map_location=device)
    #ckpt = torch.load('./Multistage13/PETimg_CTimg.pth', map_location=device)
    img_model_PET_CT.load_state_dict(ckpt['model'])

    img_model_CT_PET = ResUNet(1,1).to(device)
    ckpt = torch.load('./Pretrained_model/S_Limg.pth',map_location=device)
    #ckpt = torch.load('./Multistage13/CTimg_PETimg.pth', map_location=device)
    img_model_CT_PET.load_state_dict(ckpt['model'])


    FP = Forward_projection('astra_cuda').to(device)
    BP = Backward_projection('astra_cuda').to(device)



    train_dataset = Lits_DataSet(opt.datapath,'train')
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=opt.batch_size,\
                                  num_workers=opt.num_threads, shuffle=True)
    val_dataset = Lits_DataSet(opt.datapath,'val')
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=opt.batch_size,\
                                  num_workers=opt.num_threads, shuffle=True)
        
    save_result_path = os.path.join(opt.checkpoints_dir,opt.task_name)
    util.mkdir(save_result_path)

    optimizer_PET_CT = optim.Adam(img_model_PET_CT.parameters(), lr=opt.lr_PET_CT)
    optimizer_CT_PET = optim.Adam(img_model_CT_PET.parameters(), lr=opt.lr_CT_PET)
    best = [0,np.inf] 

    model_save_path = os.path.join(save_result_path,'model')
    util.mkdir(model_save_path)
    logger_save_path = os.path.join(save_result_path,'logger')
    util.mkdir(logger_save_path)

    log_train = logger.Train_Logger(logger_save_path,"train_log")


    for epoch in range(opt.epoch):
        epoch = epoch +1
        train_log= train(train_dataloader,epoch)
        val_log = val(val_dataloader)

        log_train.update(epoch,train_log,val_log)
        
        PET_CT_state = {'model': img_model_PET_CT.state_dict(), 'epoch': epoch}
        torch.save(PET_CT_state, os.path.join(model_save_path, 'latest_model_PET_CT.pth'))
        CT_PET_state = {'model': img_model_CT_PET.state_dict(), 'epoch': epoch}
        torch.save(CT_PET_state, os.path.join(model_save_path, 'latest_model_CT_PET.pth'))

        if val_log['Val_Loss'] < best[1]:
           print('Saving best model')
           torch.save(PET_CT_state, os.path.join(model_save_path, 'best_model_PET_CT.pth'))
           torch.save(CT_PET_state, os.path.join(model_save_path, 'best_model_CT_PET.pth'))
           best[0] = epoch
           best[1] = val_log['Val_Loss']

        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))
        
        if epoch%opt.model_save_fre ==0:
            torch.save(PET_CT_state, os.path.join(model_save_path, 'model_PET_CT_'+np.str(epoch)+'.pth'))
            torch.save(CT_PET_state, os.path.join(model_save_path, 'model_CT_PET_' + np.str(epoch) + '.pth'))

        torch.cuda.empty_cache()

    #test_result('best_model_PET_CT.pth')
    

 
        

            
            

            
            
            
            
            
            
