from models import SegmentNet
# from models import SegmentNet, DecisionNet, weights_init_kaiming, weights_init_normal
from dataset import Dataset
import cv2
import torch.nn as nn
import torch

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

import os
import sys
import argparse
import time
import PIL.Image as Image

from ptflops import get_model_complexity_info

import random
import numpy as np

#from measures import compute_ave_MAE_of_methods

from shutil import copyfile
import datetime as dt
from Temp.tools.SendEmail import SendEmail
from auxfunc import setup_seed

'''
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
'''
def dice_loss(pred, mask):
    
    # zeros = torch.zeros_like(pred)
    # ones = torch.ones_like(pred)
    pred = torch.sigmoid(pred)
    # pred = torch.where(pred>0.51,ones,zeros)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    #wiou = 1 - ((inter + 1)/(union - inter+1)).mean()
    wiou = 1 - ((inter + 1)/(union+1)).mean()

    return wiou

def computeIOU(pred,mask):
    zeros = torch.zeros_like(pred)
    ones = torch.ones_like(pred)
    pred = torch.where(pred>0,ones,zeros)
    inter = ((pred * mask)).sum(dim=(0,1,2,3))
    union = ((pred + mask)).sum(dim=(0,1,2,3))
    wiou = ((inter + 1)/(union - inter+1)).mean()
    return wiou

def showloss(pred, mask,opt):
    zeros = torch.zeros_like(pred)
    ones = torch.ones_like(pred)
    pred = torch.sigmoid(pred)
    pred = torch.where(pred>0.5,ones,zeros)

    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    dice = 1 - ((inter + 1)/(union+1)).mean()

    bce = torch.nn.BCELoss()(pred,mask)

    loss = opt.Lambda*dice + (1-opt.Lambda)*bce

    return {'dice':dice.item(),'bce':bce.item(),'loss':loss.item()}


def train_Segment(opt):
    try:
        Error = False       
        print(opt)
        now_time = dt.datetime.now().strftime('%b.%d %T')

        dataSetRoot = f"Data/{opt.Dataset}/F{opt.Fold}"

        OutputPath = f"Model/{opt.Dataset}_F{opt.Fold}_lambda{opt.Lambda}_{opt.Remark}"

        model_name = f'Seg_lr{opt.Lr}_Batch{opt.Batch_size}_Epoch{opt.End_epoch}'
        
        saveModelDir = os.path.join(OutputPath,"Segment_Model",model_name)
        verifyDir = os.path.join(OutputPath,"Segment_Varify",model_name)
        logpath = os.path.join(OutputPath,f'Log/{model_name}/{now_time}')



        writer = SummaryWriter(logpath)

        os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_ids

        gpu_num = len(opt.gpu_ids.split(','))

        milestones = []
        for i in opt.Milestones.split(','):
            milestones.append(int(i))

        if os.path.exists(saveModelDir) == False:
            os.makedirs(saveModelDir, exist_ok=True)

        if os.path.exists(verifyDir) == False:
            os.makedirs(verifyDir, exist_ok=True)

        # Build nets
        segment_net = SegmentNet()
                
        # Loss functions
        criterion_segment_bce = torch.nn.BCEWithLogitsLoss()
        criterion_segment_dice  = dice_loss

        if opt.cuda:
            segment_net = segment_net.cuda()

        if gpu_num > 1:
            segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(gpu_num)))

        if opt.Need_load_segment_model:
            segment_net.load_state_dict(torch.load(opt.Load_segment_model_dir))

        # Optimizers
        optimizer_seg = torch.optim.Adam(segment_net.parameters(), lr=opt.Lr, betas=(opt.b1, opt.b2))
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_seg, milestones=milestones, gamma=opt.Gamma)

        transforms_sample = transforms.Compose([
            transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        transforms_mask = transforms.Compose([
            transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        trainOKloader = DataLoader(
            Dataset(dataSetRoot, opt.Dataset, transforms_sample=transforms_sample, transforms_mask= transforms_mask,subFold="Train",isTrain=True,label="OK"),
            batch_size=opt.Batch_size,
            shuffle=True,
            num_workers=opt.worker_num
        )
        trainNGloader = DataLoader(
            Dataset(dataSetRoot, opt.Dataset, transforms_sample=transforms_sample, transforms_mask= transforms_mask,subFold="Train",isTrain=True,label="NG"),
            batch_size=opt.Batch_size,
            shuffle=True,
            num_workers=opt.worker_num
        )

        if opt.Train_with_test:
            testloader = DataLoader(
                Dataset(dataSetRoot,opt. Dataset, transforms_sample=transforms_sample, transforms_mask= transforms_mask,  subFold="Test", isTrain=True,label="ALL"), #subFold="normal_valid"
                batch_size=1,
                shuffle=False,
                num_workers=opt.worker_num,
            )

        for epoch in range(opt.Begin_epoch, opt.End_epoch):
            loss_epoch = 0
            iterOK = trainOKloader.__iter__()
            iterNG = trainNGloader.__iter__()

            lenNum = len(trainOKloader)*2

            segment_net.train()

            # train *****************************************************************
            for i in range(0, lenNum):
                try:
                    if i%2 == 0:
                        batchData = iterOK.__next__()
                    else:
                        batchData = iterNG.__next__()
                except StopIteration:
                    iterNG = trainNGloader.__iter__()
                    continue

                if opt.cuda:
                    img = batchData["img"].cuda()
                    mask = batchData["mask"].cuda()
                else:
                    img = batchData["img"]
                    mask = batchData["mask"]

                optimizer_seg.zero_grad()
                
                seg = segment_net(img)
                res = seg['seg']

                loss_seg_dice = criterion_segment_dice(res, mask)
                loss_seg_bce  = criterion_segment_bce(res, mask)
            
                loss_seg = opt.Lambda*loss_seg_dice + (1-opt.Lambda)*loss_seg_bce

                loss_seg.backward()
                optimizer_seg.step()

                print('"\rEpoch: {%d} batch: {%d}/{%d} loss: {%.10f} Bce:{%.10f} Dice:{%.10f}"'%(epoch,i,lenNum,loss_seg.item(),loss_seg_bce.item(),loss_seg_dice.item()),end='',flush=False)
                writer.add_scalar('Train_Loss', loss_seg.item(),global_step=(epoch*lenNum+i))
                writer.add_scalar('Dice_Loss', loss_seg_dice.item(),global_step=(epoch*lenNum+i))
                writer.add_scalar('Bce_Loss', loss_seg_bce.item(),global_step=(epoch*lenNum+i))
                
                loss_epoch = loss_epoch + loss_seg.item()
                
                
            exp_lr_scheduler.step()
            now_time = dt.datetime.now().strftime('%b.%d %T')
            writer.add_scalar('Loss_Epoch', (loss_epoch/lenNum),global_step=epoch)
            print('"\n %s \r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"'%(
                        now_time,
                        epoch,
                        opt.End_epoch,
                        i,
                        lenNum,
                        loss_epoch
                    ))
        
            # test ****************************************************************************
            if opt.Train_with_test and epoch % opt.Test_interval == 0 and epoch >= opt.Test_interval:
            #if opt.need_test:
                test_loss = 0
                segment_net.eval()
                now_time = dt.datetime.now().strftime('%b.%d %T')
                print(now_time+' Begin Test!')
                
                iou_list = []
                pos_iou_list = []
                for i, testBatch in enumerate(testloader):
                    if opt.cuda:
                        imgTest = testBatch["img"].cuda()
                        mask = testBatch["mask"].cuda()
                    else:
                        imgTest = testBatch["img"]
                        mask = testBatch["mask"]
                    
                    rstTest = segment_net(imgTest)

                    segTest = rstTest["seg"]

                    loss_seg_dice  = criterion_segment_dice(segTest, mask)

                    test_loss = test_loss + loss_seg_dice.item()
                
                    save_path_str = os.path.join(verifyDir, "epoch_%d"%epoch)

                    if os.path.exists(save_path_str) == False:
                        os.makedirs(save_path_str, exist_ok=True)
       
                    save_image(imgTest.data, "%s/img_%d.jpg"% (save_path_str, i))
                    save_image(segTest.data, "%s/img_%d_seg.jpg"% (save_path_str, i))

                    iou = computeIOU(segTest, mask).item()
                    iou_list.append(iou)

                    writer.add_scalar('Seg_Test_Loss', loss_seg_dice.item(),global_step=i+epoch*len(testloader))
                    
                meaniou = np.mean(iou_list)
                writer.add_scalar('Seg_Test_EpochLoss', test_loss/(len(testloader)-1),global_step=epoch)
                writer.add_scalar('IOU', meaniou,global_step=epoch)
                segment_net.train()

            # save parameters *****************************************************************
            if opt.Train_with_save and epoch % opt.Save_interval == 0 and epoch >= opt.Save_interval:
                segment_net.eval()
                
                save_path_str =saveModelDir
                if os.path.exists(save_path_str) == False:
                    os.makedirs(save_path_str, exist_ok=True)

                if gpu_num>1:
                    torch.save(segment_net.module.state_dict(), "%s/Segment_Net_%d.pth" % (save_path_str, epoch))
                else:
                    torch.save(segment_net.state_dict(), "%s/Segment_Net_%d.pth" % (save_path_str, epoch))

                segment_net.train()

    except FileExistsError as e:
        Error = True
        emailcontent = f"Segment Network Error have some problem: {e}"
        emailsubject = "Segment Network Error"
        print(emailcontent)