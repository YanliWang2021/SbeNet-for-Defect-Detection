import glob
import os
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset 
from PIL import Image
import torchvision.transforms as transforms

import torchvision.transforms.functional as VF

def Divide(dataroot,datatype,selection):
    if datatype == 'KSDD':
        imgFiles = sorted(glob.glob(dataroot+ "/*/*.jpg"))
        labelFiles = sorted(glob.glob(dataroot+ "/*/*.bmp"))
        if selection =="ALL":
            sample = imgFiles
            mask = labelFiles
        else:
            NGsample,NGlabel,OKsample,OKlabel, = [],[],[],[]
            for i in range(len(labelFiles)):
                labelimg = cv2.imread(labelFiles[i])
                indicater = bool(np.sum(np.sum(labelimg)))
                if indicater:
                    NGsample.append(imgFiles[i])
                    NGlabel.append(labelFiles[i])
                else:
                    OKsample.append(imgFiles[i])
                    OKlabel.append(labelFiles[i])
            if selection == "OK":
                sample = OKsample
                mask = OKlabel
            elif selection == "NG":
                sample = NGsample
                mask = NGlabel
    elif datatype == 'DAGM':
        if selection =='NG':
            sample = sorted(glob.glob(dataroot+ "/NG/*.PNG"))
            mask = sorted(glob.glob(dataroot+ "/NG/*.bmp"))
        elif selection =='OK':
            sample = sorted(glob.glob(dataroot+ "/OK/*.PNG"))
            mask = sorted(glob.glob(dataroot+ "/OK/*.bmp"))
        else:
            sample = sorted(glob.glob(dataroot+ "/*/*.PNG"))
            mask = sorted(glob.glob(dataroot+ "/*/*.bmp"))
    elif datatype == 'SSD':
        if selection =='NG':
            sample = sorted(glob.glob(dataroot+ "/NG/*.jpg"))
            mask = sorted(glob.glob(dataroot+ "/NG/*.bmp"))
        elif selection =='OK':
            sample = sorted(glob.glob(dataroot+ "/OK/*.jpg"))
            mask = sorted(glob.glob(dataroot+ "/OK/*.bmp"))
        else:
            sample = sorted(glob.glob(dataroot+ "/*/*.jpg"))
            mask = sorted(glob.glob(dataroot+ "/*/*.bmp"))
    return {'sample':sample,"label":mask}

class Dataset(Dataset):
    def __init__(self, dataRoot, datatype, transforms_sample= None, transforms_mask = None, subFold=None, isTrain=True,label = None):

        if transforms_sample== None:
            self.sampletransform = transforms.Compose([transforms.ToTensor()])
        else:
            self.sampletransform = transforms_sample

        if transforms_mask == None:
            self.maskTransform = transforms_sample
        else:
            self.maskTransform = transforms_mask
        
        

        if subFold == None:
            file_dir = dataRoot
        else:
            file_dir = os.path.join(dataRoot,subFold)
        
        fileDict = Divide(file_dir,datatype,label)

        self.isTrain = isTrain
        self.imgFiles = fileDict["sample"]
        self.labelFiles = fileDict["label"]#bmp
        self.len = len(self.imgFiles)

    def __getitem__(self, index):
        idx = index % self.len
        img = Image.open(self.imgFiles[idx]).convert("L")
        mat = cv2.imread(self.labelFiles[idx], cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((3, 3), np.uint8)
        mat = cv2.dilate(mat, kernel)
        mask = Image.fromarray(mat)
        label = np.array([float(bool(np.sum(np.sum(mat))))]).astype('float32')
        if self.isTrain==True:
            if np.random.rand(1) > 0.5:
                mask = VF.hflip(mask)
                img  = VF.hflip(img)
            if np.random.rand(1) > 0.5:
                mask = VF.vflip(mask)
                img  = VF.vflip(img)
    
        img = self.sampletransform(img)
        mask = self.maskTransform(mask)
        label = torch.from_numpy(label)

        return {"img":img,"mask":mask, "label":label}

    def __len__(self):
        return len(self.imgFiles)
        




if __name__ == "__main__":
    t = Divide("Data/SSD/Test",'SSD','NG')
    pass
