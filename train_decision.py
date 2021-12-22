
from models import SegmentNet, DecisionNet
from dataset import Dataset
import numpy as np
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter


import os
import sys
import argparse
import time
import PIL.Image as Image
from shutil import copyfile
import datetime as dt
from Temp.tools.SendEmail import SendEmail
from auxfunc import compute_roc,compute_ap,setup_seed

def train_Decision(opt):
    try:
        Error = False
        print(opt)
        now_time = dt.datetime.now().strftime('%b.%d %T')

        OutputPath = f"Model/F{opt.Fold}_Dice{opt.Dice}_Bce{format(float(1-opt.Dice),'.1f')}_Dilate{opt.Dilate}_{opt.remark}"
        model_name = f'Dec_seg{opt.dec_trainseg_epoch}_lr{opt.dec_lr}_Batch{opt.dec_batch_size}_Epoch{opt.dec_end_epoch}'

        dataSetRoot = f"Data/Fold_{opt.Fold}" 

        saveModelDir = os.path.join(OutputPath,"Decision_Model",now_time,model_name)
        verifyDir = os.path.join(OutputPath,"Decision_Varify",now_time,model_name)
        logPath = os.path.join(OutputPath,f'Log/{model_name}/{now_time}')

        SegModelName = f'Seg_lr{opt.seg_lr}_Batch{opt.seg_batch_size}_Epoch{opt.seg_end_epoch}'

        
        Segment_ModelPath = os.path.join(OutputPath,"Segment_Model",SegModelName,f"HUnet_{opt.dec_trainseg_epoch}.pth")
        gpu_num = len(opt.gpu_ids.split(','))

        dec_milestones = []
        for i in opt.dec_milestones.split(','):
            dec_milestones.append(int(i))

        writer = SummaryWriter(logPath)

        os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_ids

        dec_milestones = []
        for i in opt.dec_milestones.split(','):
            dec_milestones.append(int(i))

        writer.add_text("Basic_Config",str(opt))

        if os.path.exists(saveModelDir) == False:
            os.makedirs(saveModelDir, exist_ok=True)

        if os.path.exists(verifyDir) == False:
            os.makedirs(verifyDir, exist_ok=True)
        # str_ids = opt.gpu_ids.split(',')
        # gpu_ids = []
        # for str_id in str_ids:
        #     gid = int(str_id)
        #     if gid >=0:
        #         gpu_ids.append(gid)
        # # set gpu ids
        # if len(gpu_ids)>0:
        #     torch.cuda.set_device(gpu_ids[0])

        #dataSetRoot = "/home/ylwang/TinySegNet-master/Data/hair_dtect/" # "/home/sean/Data/KolektorSDD_sean"

        


        # ***********************************************************************

        # Build nets
        segment_net = SegmentNet(init_weights=True)
        decision_net = DecisionNet(init_weights=True)
        writer.add_text("DecisionNet stucture",str(decision_net))
        writer.add_text("Decision using SegmentNet stucture",str(segment_net))
        # Loss functions
        #criterion_segment  = torch.nn.MSELoss()
        #criterion_decision = torch.nn.MSELoss()
        criterion_decision = torch.nn.BCELoss()

        if opt.cuda:
            segment_net = segment_net.cuda()
            decision_net = decision_net.cuda()
            #criterion_segment.cuda()
            criterion_decision.cuda()

        if gpu_num > 1:
            segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(opt.gpu_num)))
            decision_net = torch.nn.DataParallel(decision_net, device_ids=list(range(opt.gpu_num)))

        if opt.dec_begin_epoch != 0:
            # Load pretrained models
            decision_net.load_state_dict(torch.load(os.path.join(saveModelDir,"Decision_net_%d.pth" % (opt.dec_begin_epoch))))
            #decision_net.load_state_dict(torch.load("Model/F0_Dice0.3_Bce0.7_Dilate5/Decision_Model/Feb.09 17:04:11/Dec_seg200_lr0.001_Batch1_Epoch101/Decision_net_85.pth"))
        # load pretrained segment parameters
        segment_net.load_state_dict(torch.load(Segment_ModelPath))
        segment_net.eval()

        # Optimizers
        optimizer_dec = torch.optim.Adam(decision_net.parameters(), lr=opt.dec_lr, betas=(opt.dec_b1, opt.dec_b2))
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_dec, milestones=dec_milestones, gamma=opt.dec_gamma)

        transforms_sample = transforms.Compose([
            transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainOKloader = DataLoader(
            Dataset(dataSetRoot, transforms_sample=transforms_sample, subFold="Train", isTrain=True,label="OK",dilate = opt.Dilate),
            batch_size=opt.dec_batch_size,
            shuffle=True,
            num_workers=opt.worker_num,
        )
        trainNGloader = DataLoader(
            Dataset(dataSetRoot, transforms_sample=transforms_sample, subFold="Train", isTrain=True,label="NG",dilate = opt.Dilate),
            batch_size=opt.dec_batch_size,
            shuffle=True,
            num_workers=opt.worker_num,
        )


        testloader = DataLoader(
            Dataset(dataSetRoot, transforms_sample=transforms_sample,  subFold="Test", isTrain=False,label="ALL",dilate = opt.Dilate),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        copyfile('train_decision.py',saveModelDir+'/segment.py')
        copyfile('models.py',saveModelDir+'/models.py')
        copyfile('dataset.py',saveModelDir+'/dataset.py')

        for epoch in range(opt.dec_begin_epoch, opt.dec_end_epoch):
            
            iterOK = trainOKloader.__iter__()
            iterNG = trainNGloader.__iter__()

            lenNum = len(trainOKloader)*2

            
            epoch_loss = 0
            # train *****************************************************************
            for i in range(0, lenNum):
                decision_net.train()
                #segment_net.train()
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
                    gt_c = batchData["label"].cuda()
                else:
                    img = batchData["img"]
                    gt_c = batchData["label"]

                rst = segment_net(img)

                f = rst["f"]
                seg = rst["seg"]

                optimizer_dec.zero_grad()

                rst_d = decision_net(f, seg)
                # rst_d = torch.Tensor.long(rst_d)

                loss_dec = criterion_decision(rst_d, gt_c)
                
                epoch_loss = epoch_loss + loss_dec.item()
                # print(dec.shape,mask.shape)
                loss_dec.backward()
                optimizer_dec.step()
                
                writer.add_scalar('Train_Loss', loss_dec.item(),global_step=(epoch*lenNum+i))
                print("\r [Epoch %d/%d]  [Batch %d/%d] [loss %.10f]"%(epoch,opt.dec_end_epoch,i,lenNum,loss_dec.item()),end='',flush=False)
            
            exp_lr_scheduler.step()
            writer.add_scalar('Epoch_Loss', epoch_loss,global_step=(epoch))
            print("[Epoch {%d}/{%d}] [Epoch loss {%.10f}]"%(epoch,opt.dec_end_epoch, epoch_loss))

            # save parameters *****************************************************************
            if opt.dec_need_save and epoch % opt.dec_save_interval == 0 and epoch >= opt.dec_save_interval:
                decision_net.eval()

                if gpu_num>1:
                    torch.save(decision_net.module.state_dict(), os.path.join(saveModelDir,"Decision_net_%d.pth" % (epoch)) )
                else:
                    torch.save(decision_net.state_dict(), os.path.join(saveModelDir,"Decision_net_%d.pth" % (epoch)))

                print("save weights ! epoch = %d"%epoch)
                decision_net.train()

            # test ****************************************************************************
            if opt.dec_need_test and epoch % opt.dec_test_interval == 0 and epoch >= opt.dec_test_interval:
            #if True:
                test_epoch_loss = 0
                decision_net.eval()
                error_item = 0
     
                prob_socre=[]
                label_score = []
                lenNum = len(testloader)

                testiter = testloader.__iter__()

                for i in range(0, lenNum):

                    testBatch = testiter.__next__()

                    
                    torch.cuda.synchronize()

                    imgTest = testBatch["img"].cuda()
                    label = testBatch["label"].item()

                    with torch.no_grad():
                        rstTest = segment_net(imgTest)

                    fTest = rstTest["f"]
                    segTest = rstTest["seg"]

                    with torch.no_grad():
                        cTest = decision_net(fTest, segTest)
                    
                    prob_socre.append(cTest.item())
                    label_score.append(label)

                    test_loss = criterion_decision(cTest,testBatch["label"].cuda())
                    writer.add_scalar('Test_Loss', test_loss,global_step=(i+lenNum*epoch))
                    test_epoch_loss = test_epoch_loss+test_loss

                    torch.cuda.synchronize()

                    if cTest.item() > 0.5:
                        labelStr = "NG"
                        if not label:
                            error_item = error_item +1
                    else: 
                        labelStr = "OK"
                        if label:
                            error_item = error_item +1
                    if not label :
                        save_path_str = os.path.join(verifyDir,f"epoch{epoch}","fromOK", labelStr)
                    else:
                        save_path_str = os.path.join(verifyDir,f"epoch{epoch}","fromNG", labelStr)


                    if os.path.exists(save_path_str) == False:
                        os.makedirs(save_path_str, exist_ok=True)
                    
                    save_image(imgTest.data, "%s/img_%d_.jpg"% (save_path_str, i)) # (save_path_str, i, labelStr))
                    save_image(segTest.data, "%s/img_%d_seg_%.6f.jpg"% (save_path_str, i,cTest.item()))#(save_path_str, i, labelStr))
                
                
                ROC = compute_roc(np.array(label_score), np.array(prob_socre))
                auc,tpr,fpr,thresholds,fig = ROC['AUC'],ROC['TPR'],ROC['FPR'],ROC['Thresholds'],ROC['Fig']
                AP = compute_ap(np.array(label_score), np.array(prob_socre))
                ap,pre,rec,apthre,PRcurve = AP['AP'],AP['PRE'],AP['REC'],AP['Thresholds'],AP['Fig']

                writer.add_scalar('AUC',auc,global_step=(epoch))
                writer.add_figure('ROC',fig,global_step=(epoch))
                writer.add_scalar('AP',ap,global_step=(epoch))
                writer.add_figure('PR',PRcurve,global_step=(epoch))
                for istep in range(len(tpr)):
                    writer.add_scalar('TPR',tpr[istep],global_step=(istep))
                    writer.add_scalar('FPR',fpr[istep],global_step=(istep))
                    writer.add_scalar('ROCThresholds',thresholds[istep],global_step=(istep))
                for istep in range(len(pre)):
                    writer.add_scalar('PRE',pre[istep],global_step=(istep))
                    writer.add_scalar('REC',rec[istep],global_step=(istep))
                    
                if istep < len(pre)-1:
                    writer.add_scalar('APThresholds',apthre[istep],global_step=(istep))
                if ap >= 1:
                    break

                decision_net.train()

    except Exception as e:
        Error = True
        emailcontent = f"Decision Network Error have some problem: {e}"
        emailsubject = "Decision Network Error"
        print(emailcontent)
        SendEmail(emailcontent,emailsubject)
    
    if not Error:
        emailcontent = f'''F{opt.Fold}_Dice{opt.Dice}_Bec{float(1-opt.Dice)}_Dilate{opt.Dilate}_Dec_seg{opt.dec_trainseg_epoch}_lr{opt.dec_lr}_Batch{opt.dec_batch_size}_Epoch{opt.dec_end_epoch}_{opt.remark} Trained!!!'''
        emailsubject = "Decision Network Finished"
        print(emailcontent)
        SendEmail(emailcontent,emailsubject)


    
if __name__ == "__main__":

    setup_seed()
    parser = argparse.ArgumentParser()

    #public argument
    parser.add_argument("--Fold", type=int, default=0, help="number of fold")
    parser.add_argument("--Dice", type=float, default=0.7, help="precent of dice loss")
    parser.add_argument("--Dilate", type=int, default=3, help="kernel size of dilate")
    parser.add_argument("--gpu_ids",default='0',type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
    parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")

    parser.add_argument("--img_height", type=int, default=1408, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")

    #Decision argument
    parser.add_argument("--dec_b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--dec_b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--dec_lr", type=float, default=0.001, help="adam: learning rate")#0.001、0.0005
    parser.add_argument("--dec_batch_size", type=int, default=1, help="batch size of input")
    parser.add_argument("--dec_milestones", type=str, default="50,140,180", help="adam: milestone")
    parser.add_argument("--dec_gamma", type=float, default=0.1, help="adam: gamma")

    parser.add_argument("--dec_begin_epoch", type=int, default=0, help="begin_epoch")
    parser.add_argument("--dec_end_epoch", type=int, default=101, help="end_epoch")
    parser.add_argument("--dec_trainseg_epoch", type=int, default=55, help="pretrained segment epoch")

    parser.add_argument("--dec_need_test", type=bool, default=True, help="need to test")
    parser.add_argument("--dec_test_interval", type=int, default=5, help="interval of test")
    parser.add_argument("--dec_need_save", type=bool, default=True, help="need to save")
    parser.add_argument("--dec_save_interval", type=int, default=5, help="interval of save weights")

    #segment argument
    parser.add_argument("--seg_batch_size", type=int, default=2, help="batch size of input")
    parser.add_argument("--seg_lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--seg_end_epoch", type=int, default=201, help="end_epoch")
    parser.add_argument("--remark", type=str, default='A', help="remark")

    opt = parser.parse_args()

    train_Decision(opt)