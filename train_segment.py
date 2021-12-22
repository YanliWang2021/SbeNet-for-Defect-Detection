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
    pred = torch.where(pred>0.51,ones,zeros)

    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    dice = 1 - ((inter + 1)/(union - inter+1)).mean()

    bce = torch.nn.BCELoss()(pred,mask)

    loss = opt.Dice*dice + (1-opt.Dice)*bce

    return {'dice':dice.item(),'bce':bce.item(),'loss':loss.item()}


def train_Segment(opt):
    try:
        Error = False       
        print(opt)
        now_time = dt.datetime.now().strftime('%b.%d %T')

        OutputPath = f"Model/F{opt.Fold}_Dice{opt.Dice}_Bce{format(float(1-opt.Dice),'.1f')}_Dilate{opt.Dilate}_{opt.remark}"
        model_name = f'Seg_lr{opt.seg_lr}_Batch{opt.seg_batch_size}_Epoch{opt.seg_end_epoch}'
        
        dataSetRoot = f"Data/Fold_{opt.Fold}" #"/home/sean/Data/KolektorSDD_sean"  # 

        save_model_dir = os.path.join(OutputPath,"Segment_Model",model_name)
        verify_dir = os.path.join(OutputPath,"Segment_Varify",model_name)
        logpath = os.path.join(OutputPath,f'Log/{model_name}/{now_time}')

        writer = SummaryWriter(logpath)

        os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_ids
        gpu_num = len(opt.gpu_ids.split(','))

        seg_milestones = []
        for i in opt.seg_milestones.split(','):
            seg_milestones.append(int(i))

        writer.add_text("Basic_Config",str(opt))


        if os.path.exists(save_model_dir) == False:
            os.makedirs(save_model_dir, exist_ok=True)

        if os.path.exists(verify_dir) == False:
            os.makedirs(verify_dir, exist_ok=True)

        # str_ids = opt.gpu_ids.split(',')
        # if len(str_ids)>0:
        #     gid = 0
        #     if 
        # gpu_ids = []
        # for str_id in str_ids:
            
        #     if gid >=0:
        #         gpu_ids.append(gid)
        # # set gpu ids
        # if len(gpu_ids)>0:
        #     torch.cuda.set_device(gpu_ids[0])


        # ***********************************************************************

        # Build nets
        segment_net = SegmentNet(init_weights=True)
        segment_net.eval()
                
        #save_path_str = "./saved_models"
        # save_path_str =save_model_dir
        # if os.path.exists(save_path_str) == False:
        #     os.makedirs(save_path_str, exist_ok=True)

        # if gpu_num>1:
        #     torch.save(segment_net.module.state_dict(), "%s/HUnet_0.pth" % (save_path_str))
        # else:
        #     torch.save(segment_net.state_dict(), "%s/HUnet_0.pth" % (save_path_str))

        now_time = dt.datetime.now().strftime('%b.%d %T')
        print(now_time + " save weights! epoch = 0")
        segment_net.train()
        #unet = UNet(n_channels=1, n_classes=1, bilinear=True,init_weights=True)

        # Loss functions
        criterion_segment_mse  = torch.nn.MSELoss()
        criterion_segment_bce = torch.nn.BCEWithLogitsLoss()
        criterion_segment_dice  = dice_loss
        #criterion_segment  =torch.nn.BCEWithLogitsLoss()
        #criterion_segment = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.cuda.FloatTensor([1]))

        if opt.cuda:
            segment_net = segment_net.cuda()


        if gpu_num > 1:
            segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(gpu_num)))

        if opt.seg_begin_epoch != 0:
            # Load pretrained models
            segment_net.load_state_dict(torch.load(".%s/Seg_%d.pth" % (save_model_dir,opt.begin_epoch)))
        else:
            #segment_net.load_state_dict(torch.load("Model/F%d_init.pth" % (opt.Fold)))
            pass
        # Optimizers
        optimizer_seg = torch.optim.Adam(segment_net.parameters(), lr=opt.seg_lr, betas=(opt.seg_b1, opt.seg_b2))
        #optimizer_seg = optim.RMSprop(unet.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_seg, milestones=seg_milestones, gamma=opt.seg_gamma)

        transforms_sample = transforms.Compose([
            transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
            #transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            #
        ])

        transforms_mask = transforms.Compose([
            #transforms.Resize((opt.img_height//8, opt.img_width//8)),
            transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
            #transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            #transforms.Normalize(*mean_std),
        ])


        trainOKloader = DataLoader(
            Dataset(dataSetRoot, transforms_sample=transforms_sample, transforms_mask= transforms_mask,subFold="Train",isTrain=True,label="OK"),
            batch_size=opt.seg_batch_size,
            shuffle=True,
            num_workers=opt.worker_num,
        )
        trainNGloader = DataLoader(
            Dataset(dataSetRoot, transforms_sample=transforms_sample, transforms_mask= transforms_mask,subFold="Train",isTrain=True,label="NG"),
            batch_size=opt.seg_batch_size,
            shuffle=True,
            num_workers=opt.worker_num,
        )


        testloader = DataLoader(
            Dataset(dataSetRoot, transforms_sample=transforms_sample, transforms_mask= transforms_mask,  subFold="Test", isTrain=True,label="ALL"), #subFold="normal_valid"
            batch_size=1,
            shuffle=False,
            num_workers=opt.worker_num,
        )



        copyfile('train_segment.py',save_model_dir+'/segment.py')
        copyfile('models.py',save_model_dir+'/models.py')
        copyfile('dataset.py',save_model_dir+'/dataset.py')

        for epoch in range(opt.seg_begin_epoch, opt.seg_end_epoch):
            print(now_time+" Begin Training!")
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

               
    
                #loss_seg_dice = criterion_segment_dice(res, res)
                loss_seg_dice = criterion_segment_dice(res, mask)
                loss_seg_bce  = criterion_segment_bce(res, mask)
            
                loss_seg = opt.Dice*loss_seg_dice + (1-opt.Dice)*loss_seg_bce

                loss_seg.backward()
                optimizer_seg.step()

                show_segloss = showloss(res,mask,opt)
                print('"\rEpoch: {%d} batch: {%d}/{%d} loss: {%.10f} Bce:{%.10f} Dice:{%.10f}"'%(epoch,i,lenNum,show_segloss['loss'],show_segloss['bce'],show_segloss['dice']),end='',flush=False)
                writer.add_scalar('Train_Loss', show_segloss['loss'],global_step=(epoch*lenNum+i))
                writer.add_scalar('Dice_Loss', show_segloss['dice'],global_step=(epoch*lenNum+i))
                writer.add_scalar('Bce_Loss', show_segloss['bce'],global_step=(epoch*lenNum+i))
                #loss_seg = 0.3*loss_seg_dice + 0.7*loss_seg_mse
                loss_epoch = loss_epoch + show_segloss['loss']
                
                
            exp_lr_scheduler.step()
            now_time = dt.datetime.now().strftime('%b.%d %T')
            writer.add_scalar('Loss_Epoch', (loss_epoch/lenNum),global_step=epoch)
            print('"\n %s \r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"'%(
                        now_time,
                        epoch,
                        opt.seg_end_epoch,
                        i,
                        lenNum,
                        loss_epoch
                    ))
        
            # test ****************************************************************************
            if opt.seg_need_test and epoch % opt.seg_test_interval == 0 and epoch >= opt.seg_test_interval:
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
                    
                    label = testBatch["label"].item()

                    #print(imgTest)
                    rstTest = segment_net(imgTest)

                    segTest = rstTest["seg"]

                    loss_seg_dice  = criterion_segment_dice(segTest, mask)
                    writer.add_scalar('Seg_Test_Loss', loss_seg_dice.item(),global_step=i+epoch*len(testloader))

                    test_loss = test_loss + loss_seg_dice.item()
                
                    save_path_str = os.path.join(verify_dir, "epoch_%d"%epoch)
                    #print(save_path_str)
                    if os.path.exists(save_path_str) == False:
                        os.makedirs(save_path_str, exist_ok=True)
                        #os.mkdir(save_path_str)
                    
                    #print("processing image NO %d, time comsuption %fs"%(i, t2 - t1))
                    save_image(imgTest.data, "%s/img_%d.jpg"% (save_path_str, i))
                    save_image(segTest.data, "%s/img_%d_seg.jpg"% (save_path_str, i))

                    iou = computeIOU(segTest, mask).item()
                    iou_list.append(iou)
                    if label>0:
                        
                        pos_iou_list.append(iou)
                    
                meaniou = np.mean(iou_list)
                pos_meaniou = np.mean(pos_iou_list)
                writer.add_scalar('Seg_Test_EpochLoss', test_loss/(len(testloader)-1),global_step=epoch)
                writer.add_scalar('IOU', meaniou,global_step=epoch)
                writer.add_scalar('POSIOU', pos_meaniou,global_step=epoch)
                segment_net.train()

            # save parameters *****************************************************************
            if opt.seg_need_save and epoch % opt.seg_save_interval == 0 and epoch >= opt.seg_save_interval:
            #if opt.need_save:
                segment_net.eval()
                
                #save_path_str = "./saved_models"
                save_path_str =save_model_dir
                if os.path.exists(save_path_str) == False:
                    os.makedirs(save_path_str, exist_ok=True)

                if gpu_num>1:
                    torch.save(segment_net.module.state_dict(), "%s/HUnet_%d.pth" % (save_path_str, epoch))
                else:
                    torch.save(segment_net.state_dict(), "%s/HUnet_%d.pth" % (save_path_str, epoch))

                now_time = dt.datetime.now().strftime('%b.%d %T')
                print(now_time + " save weights! epoch = %d"%epoch)
                segment_net.train()
    except Exception as e:
        Error = True
        emailcontent = f"Segment Network Error have some problem: {e}"
        emailsubject = "Segment Network Error"
        print(emailcontent)
        SendEmail(emailcontent,emailsubject)
    
    if not Error:
        emailcontent = f'''F{opt.Fold}_Dice{opt.Dice}_Bce{float(1-opt.Dice)}_Dilate{opt.Dilate}_Seg_lr{opt.seg_lr}_Batch{opt.seg_batch_size}_Epoch{opt.seg_end_epoch}_{opt.remark} Trained!!!'''
        emailsubject = "Segment Network Finished"
        print(emailcontent)
        SendEmail(emailcontent,emailsubject)
    
    
    
 
if __name__ == "__main__":
    setup_seed()
    parser = argparse.ArgumentParser()
    #public argument
    parser.add_argument("--Fold", type=int, default=2, help="number of fold")
    parser.add_argument("--Dice", type=float, default=0.7, help="precent of dice loss")
    parser.add_argument("--Dilate", type=int, default=3, help="kernel size of dilate")
    parser.add_argument("--gpu_ids",default='2',type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
    parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")

    parser.add_argument("--img_height", type=int, default=1408, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")

    #segment argument
    parser.add_argument("--seg_b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--seg_b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--seg_batch_size", type=int, default=2, help="batch size of input")
    parser.add_argument("--seg_lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--seg_milestones", type=str, default="50,140,180", help="adam: milestone")
    parser.add_argument("--seg_gamma", type=float, default=0.1, help="adam: gamma")
    parser.add_argument("--seg_begin_epoch", type=int, default=0, help="begin_epoch")
    parser.add_argument("--seg_end_epoch", type=int, default=201, help="end_epoch")

    parser.add_argument("--seg_need_test", type=bool, default=True, help="need to test")
    parser.add_argument("--seg_test_interval", type=int, default=5, help="interval of test")
    parser.add_argument("--seg_need_save", type=bool, default=True, help="need to save")
    parser.add_argument("--seg_save_interval", type=int, default=5, help="interval of save weights")
    parser.add_argument("--remark", type=str, default='B', help="remark")

    opt = parser.parse_args()

    train_Segment(opt)