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

    Img_Loss = LossAverage()
    Sin_Loss = LossAverage()
    NRMSE = LossAverage()
    PSNR = LossAverage()
    SSIM = LossAverage()

    sin_model.train()
    img_model.train()

    for i, (PET,CT) in enumerate(train_dataloader):  # inner loop within one epoch
        ##main model update param

        PET,CT = PET.to(device),CT.to(device)
        PET_sin = FP(PET).type(torch.FloatTensor).to(device)
        CT_sin = FP(CT).type(torch.FloatTensor).to(device)


        ####image_space
        sin_model.eval()
        img_model.train()
        fake_CT = img_model(PET)
        PET_CT_sin = sin_model(PET_sin)
        PET_CT = BP(PET_CT_sin).type(torch.FloatTensor)
        PET_CT = PET_CT.detach().to(device)

        img_loss1 = nn.L1Loss()(CT,fake_CT)  #image_loss
        img_loss2 = nn.L1Loss()(PET_CT,fake_CT)  # real_sin_loss
        img_loss = img_loss1+opt.lamada_img*img_loss2

        optimizer_img.zero_grad()
        img_loss.backward()
        optimizer_img.step()
        adjust_learning_rate(optimizer_img,epoch,opt.lr_img,opt.step_img)

        ####sin_space
        sin_model.train()
        img_model.eval()
        fake_CT_sin = sin_model(PET_sin)
        PET_CT_img = img_model(PET)
        PET_CT = FP(PET_CT_img).type(torch.FloatTensor)
        PET_CT = PET_CT.detach().to(device)

        sin_loss1 = nn.L1Loss()(CT_sin.unsqueeze(1), fake_CT_sin)
        sin_loss2 = nn.L1Loss()(PET_CT, fake_CT_sin)
        sin_loss = sin_loss1 + opt.lamada_sin*sin_loss2

        optimizer_sin.zero_grad()
        sin_loss.backward()
        optimizer_sin.step()
        adjust_learning_rate(optimizer_sin,epoch,opt.lr_sin,opt.step_sin)

        img_model.eval()
        sin_model.eval()
        fake_CT = img_model(PET)

        Real_ct = CT.cpu().detach().numpy().squeeze()*255
        Fake_ct = fake_CT.cpu().detach().numpy().squeeze()*255

        nrmse = normalized_root_mse(Real_ct, Fake_ct)
        psnr = get_psnr(Real_ct, Fake_ct,255)
        ssim = get_ssim(Real_ct, Fake_ct)

        Img_Loss.update(img_loss.item(),1)
        Sin_Loss.update(sin_loss.item(), 1)

        NRMSE.update(nrmse,1)
        PSNR.update(psnr,1)
        SSIM.update(ssim,1)

    return OrderedDict({'Train_IMG_Loss': Img_Loss.avg,'Train_Sin_Loss': Sin_Loss.avg,'Train_PSNR': PSNR.avg,'Train_SSIM': SSIM.avg,'Train_NRMSE': NRMSE.avg })


def val(val_dataloader):

    Loss = LossAverage()
    NRMSE = LossAverage()
    PSNR = LossAverage()
    SSIM = LossAverage()
    img_model.eval()

    for i, (PET, CT) in enumerate(val_dataloader):  # inner loop within one epoch
        ##main model update param

        PET = PET.to(device)
        CT = CT.to(device)

        fake_CT = img_model(PET)
        loss = nn.L1Loss()(CT, fake_CT)

        Real_ct = CT.cpu().detach().numpy().squeeze()*255
        Fake_ct = fake_CT.cpu().detach().numpy().squeeze()*255

        nrmse = normalized_root_mse(Real_ct, Fake_ct)
        psnr = get_psnr(Real_ct, Fake_ct, 255)
        ssim = get_ssim(Real_ct, Fake_ct)

        Loss.update(loss.item(), 1)
        NRMSE.update(nrmse, 1)
        PSNR.update(psnr, 1)
        SSIM.update(ssim, 1)

    return OrderedDict({'Val_Loss': Loss.avg,'Val_PSNR': PSNR.avg, 'Val_SSIM': SSIM.avg, 'Val_NRMSE': NRMSE.avg})

if __name__ == '__main__':
    opt = Options_x().parse()   # get training options
    device = torch.device('cuda:'+opt.gpu_ids if torch.cuda.is_available() else "cpu")
    print(device)

    sin_model = ResNet(1,1).to(device)
    ckpt = torch.load('./Pretrained_model/S_Lsin.pth',map_location=device)
    sin_model.load_state_dict(ckpt['model'])

    img_model = ResUNet(1,1).to(device)
    ckpt = torch.load('./Pretrained_model/S_Limg.pth',map_location=device)
    img_model.load_state_dict(ckpt['model'])

    FP = Forward_projection('astra_cuda').to(device)
    BP = Backward_projection('astra_cuda').to(device)

    #ckpt = torch.load('latest_1600.pth')
    #model.load_state_dict(ckpt['model'])


    train_dataset = Lits_DataSet(opt.datapath,'train')
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=opt.batch_size,\
                                  num_workers=opt.num_threads, shuffle=True)
    val_dataset = Lits_DataSet(opt.datapath,'val')
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=opt.batch_size,\
                                  num_workers=opt.num_threads, shuffle=True)
        
    save_result_path = os.path.join(opt.checkpoints_dir,opt.task_name)
    util.mkdir(save_result_path)

    optimizer_sin = optim.Adam(sin_model.parameters(), lr=opt.lr_sin)
    optimizer_img = optim.Adam(img_model.parameters(), lr=opt.lr_img)
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
        
        sin_state = {'model': sin_model.state_dict(), 'epoch': epoch}
        torch.save(sin_state, os.path.join(model_save_path, 'latest_model_sin.pth'))
        img_state = {'model': img_model.state_dict(), 'epoch': epoch}
        torch.save(img_state, os.path.join(model_save_path, 'latest_model_img.pth'))

        if val_log['Val_Loss'] < best[1]:
           print('Saving best model')
           torch.save(img_state, os.path.join(model_save_path, 'best_model_img.pth'))
           torch.save(sin_state, os.path.join(model_save_path, 'best_model_sin.pth'))
           best[0] = epoch
           best[1] = val_log['Val_Loss']

        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))
        
        if epoch%opt.model_save_fre ==0:
            torch.save(sin_state, os.path.join(model_save_path, 'model_sin_'+np.str(epoch)+'.pth'))
            torch.save(img_state, os.path.join(model_save_path, 'model_img_' + np.str(epoch) + '.pth'))

        torch.cuda.empty_cache()

    #test_result('best_model_img.pth')
    

 
        

            
            

            
            
            
            
            
            
