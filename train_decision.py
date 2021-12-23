
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
        
        now_time = dt.datetime.now().strftime('%b.%d %T')

        dataSetRoot = f"Data/{opt.Dataset}/F{opt.Fold}"

        OutputPath = f"Model/{opt.Dataset}_F{opt.Fold}_lambda{opt.Lambda}_{opt.Remark}"
        
        model_name = f'Dec_lr{opt.Lr}_Batch{opt.Batch_size}_Epoch{opt.End_epoch}'

        saveModelDir = os.path.join(OutputPath,"Decision_Model",model_name)        
        verifyDir = os.path.join(OutputPath,"Decision_Varify",model_name)
        logPath = os.path.join(OutputPath,f'Log/{model_name}/{now_time}')

        gpu_num = len(opt.gpu_ids.split(','))

        writer = SummaryWriter(logPath)

        os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_ids

        milestones = []
        for i in opt.Milestones.split(','):
            milestones.append(int(i))

        writer.add_text("Basic_Config",str(opt))

        if os.path.exists(saveModelDir) == False:
            os.makedirs(saveModelDir, exist_ok=True)

        if os.path.exists(verifyDir) == False:
            os.makedirs(verifyDir, exist_ok=True)

        


        # ***********************************************************************

        # Build nets
        segment_net = SegmentNet(init_weights=True)
        decision_net = DecisionNet(init_weights=True)

        criterion_decision = torch.nn.BCELoss()

        if opt.cuda:
            segment_net = segment_net.cuda()
            decision_net = decision_net.cuda()
            criterion_decision.cuda()

        if gpu_num > 1:
            segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(opt.gpu_num)))
            decision_net = torch.nn.DataParallel(decision_net, device_ids=list(range(opt.gpu_num)))

        if opt.Need_load_decision_model:
            decision_net.load_state_dict(torch.load(opt.Load_decision_model_dir))


        segment_net.load_state_dict(torch.load(opt.Load_segment_model_dir))
        segment_net.eval()

        # Optimizers
        optimizer_dec = torch.optim.Adam(decision_net.parameters(), lr=opt.Lr, betas=(opt.b1, opt.b2))
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_dec, milestones=milestones, gamma=opt.Gamma)

        transforms_sample = transforms.Compose([
            transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        trainOKloader = DataLoader(
            Dataset(dataSetRoot, opt.Dataset, transforms_sample=transforms_sample, subFold="Train", isTrain=True,label="OK"),
            batch_size=opt.Batch_size,
            shuffle=True,
            num_workers=opt.worker_num,
        )
        trainNGloader = DataLoader(
            Dataset(dataSetRoot, opt.Dataset, transforms_sample=transforms_sample, subFold="Train", isTrain=True,label="NG"),
            batch_size=opt.Batch_size,
            shuffle=True,
            num_workers=opt.worker_num,
        )

        if opt.Train_with_test:
            testloader = DataLoader(
                Dataset(dataSetRoot, opt.Dataset, transforms_sample=transforms_sample,  subFold="Test", isTrain=False,label="ALL"),
                batch_size=1,
                shuffle=False,
                num_workers=1,
            )

        for epoch in range(opt.Begin_epoch, opt.End_epoch+1):
            iterOK = trainOKloader.__iter__()
            iterNG = trainNGloader.__iter__()
            lenNum = len(trainOKloader)*2            
            epoch_loss = 0

            # train *****************************************************************
            for i in range(0, lenNum):
                decision_net.train()
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
                    label = batchData["label"].cuda()
                else:
                    img = batchData["img"]
                    label = batchData["label"]

                rst_seg = segment_net(img)
                seg = rst_seg["seg"]

                optimizer_dec.zero_grad()
                
                prediction = decision_net(seg)
                loss_dec = criterion_decision(prediction, label)
                
                epoch_loss = epoch_loss + loss_dec.item()
                

                loss_dec.backward()
                optimizer_dec.step()
                
                writer.add_scalar('Train_Loss', loss_dec.item(),global_step=(epoch*lenNum+i))
                print("\r [Epoch %d/%d]  [Batch %d/%d] [loss %.10f]"%(epoch,opt.End_epoch,i,lenNum,loss_dec.item()),end='',flush=False)
            
            exp_lr_scheduler.step()
            writer.add_scalar('Epoch_Loss', epoch_loss,global_step=(epoch))
            print("[Epoch {%d}/{%d}] [Epoch loss {%.10f}]"%(epoch,opt.End_epoch, epoch_loss))

            # save parameters *****************************************************************
            if opt.Train_with_save and epoch % opt.Save_interval == 0 and epoch >= opt.Save_interval:
                decision_net.eval()

                if gpu_num>1:
                    torch.save(decision_net.module.state_dict(), os.path.join(saveModelDir,"Decision_Net_%d.pth" % (epoch)) )
                else:
                    torch.save(decision_net.state_dict(), os.path.join(saveModelDir,"Decision_Net_%d.pth" % (epoch)))

                print("save weights ! epoch = %d"%epoch)
                decision_net.train()

            # test ****************************************************************************
            if opt.Train_with_test and epoch % opt.Test_interval == 0 and epoch >= opt.Test_interval:
            #if True:
                decision_net.eval()
                test_epoch_loss = 0
                
                pred_socre=[]
                label_score = []
                lenNum = len(testloader)
                testiter = testloader.__iter__()

                for i in range(0, lenNum):
                    testBatch = testiter.__next__()

                    torch.cuda.synchronize()

                    imgTest = testBatch["img"].cuda()
                    labelTest = testBatch["label"].cuda()

                    with torch.no_grad():
                        rst_segTest = segment_net(imgTest)

                    segTest = rst_segTest["seg"]

                    with torch.no_grad():
                        predictionTest = decision_net(segTest)
                    
                    pred_socre.append(predictionTest.item())
                    label_score.append(labelTest.cpu().item())

                    test_loss = criterion_decision(predictionTest,labelTest)
                    
                    test_epoch_loss = test_epoch_loss+test_loss
                    writer.add_scalar('Test_Loss', test_loss,global_step=(i+lenNum*epoch))
                    

                    torch.cuda.synchronize()


                    if not label :
                        save_path_str = os.path.join(verifyDir,f"epoch{epoch}","fromOK")
                    else:
                        save_path_str = os.path.join(verifyDir,f"epoch{epoch}","fromNG")


                    if os.path.exists(save_path_str) == False:
                        os.makedirs(save_path_str, exist_ok=True)
                    
                    save_image(imgTest.data, "%s/img_%d_.jpg"% (save_path_str, i)) # (save_path_str, i, labelStr))
                    save_image(segTest.data, "%s/img_%d_%.6f_seg.jpg"% (save_path_str, i,predictionTest.item()))#(save_path_str, i, labelStr))
                
                
                ROC = compute_roc(np.array(label_score), np.array(pred_socre))
                auc,tpr,fpr,roc_best_threshold,ROCcurve = ROC['AUC'],ROC['TPR'],ROC['FPR'],ROC['Thresholds'],ROC['Fig']
                AP = compute_ap(np.array(label_score), np.array(pred_socre))
                ap,pre,rec,pr_best_threshold,PRcurve = AP['AP'],AP['PRE'],AP['REC'],AP['Thresholds'],AP['Fig']

                writer.add_scalar('TestAUC',auc,global_step=(epoch))
                writer.add_figure('TestROC',ROCcurve,global_step=(epoch))
                writer.add_scalar('TestAP',ap,global_step=(epoch))
                writer.add_figure('TestPR',PRcurve,global_step=(epoch))
                writer.add_scalar('TestROC_Best_threshold',roc_best_threshold,global_step=(epoch))
                writer.add_scalar('TestPR_Best_threshold',pr_best_threshold,global_step=(epoch))

                # Early Stop
                if ap >= 1:
                    break

                decision_net.train()

    except FileExistsError as e:
        Error = True
        content = f"Decision Network Error have some problem: {e}"
        print(content)
    
    if not Error:
        content = f'''F{opt.Fold}_Dice{opt.Dice}_Bec{float(1-opt.Dice)}_Dilate{opt.Dilate}_Dec_seg{opt.dec_trainseg_epoch}_lr{opt.dec_lr}_Batch{opt.dec_batch_size}_Epoch{opt.dec_end_epoch}_{opt.remark} Trained!!!'''
        print(content)