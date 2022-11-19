import random
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Lits_DataSet(Dataset):
    def __init__(self,root,stage='train'):
        self.root = root
        self.stage = stage
        f = open(os.path.join(self.root,self.stage+'.txt'))
        self.filename = f.read().splitlines()


    def __getitem__(self, index):
        objects = self.filename[index].replace("'","")
        sample, slice = objects.split('_')[0], np.uint16(objects.split('_')[1])
        PET = self.load(os.path.join(self.root,'Datasets',sample,'std_dose.nii.gz'))
        CT = self.load(os.path.join(self.root,'Datasets', sample, 'low_dose.nii.gz'))
        pet, ct = self.normalization(PET[slice, :, :]), self.normalization(CT[slice, :, :])

        return pet[np.newaxis,:],ct[np.newaxis,:]

    def __len__(self):
        return len(self.filename)

    def normalization(self,img):
        out=(img - np.min(img))/(np.max(img) - np.min(img) + 0.000001 )
        return out    
        
    def load(self,file):
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image    
    

if __name__ =='__main__':
    path = r'C:\Users\user\3D Objects\PET2CT\Data'
    a = Lits_DataSet(path,'val')
    train_dataloader = DataLoader(dataset=a, batch_size=30, \
                                  num_workers=1, shuffle=True)
    for i, (PET,CT) in enumerate(train_dataloader):
        print(i,PET.shape,CT.shape)




