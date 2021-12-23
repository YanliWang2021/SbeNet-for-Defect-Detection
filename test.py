
from models import SegmentNet,DecisionNet
from dataset import Dataset
import torch.nn as nn
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import PIL.Image as Image
import datetime as dt
from auxfunc import compute_roc,compute_ap
import numpy as np


def test(opt):
    
    dataSetRoot = f"Data/{opt.Dataset}/F{opt.Fold}"
    
    OutputPath = f"Model/{opt.Dataset}_F{opt.Fold}_lambda{opt.Lambda}_{opt.Remark}"
    
    now_time = dt.datetime.now().strftime('%b.%d %T')

    saveResultPath = os.path.join(OutputPath,"Testresult",now_time)
    
    logPath = os.path.join(OutputPath,f'Log/Test/{now_time}')
    

    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_ids
    gpu_num = len(opt.gpu_ids.split(','))
    
   # ***********************************************************************

    # Build nets
    segment_net = SegmentNet()
    decision_net = DecisionNet()
    
    if opt.cuda:
        segment_net = segment_net.cuda()
        decision_net = decision_net.cuda()

    if opt.Need_load_segment_model:
        segment_net.load_state_dict(torch.load(opt.Load_segment_model_dir))
    if opt.Need_load_decision_model:
        decision_net.load_state_dict(torch.load(opt.Load_decision_model_dir))

    if gpu_num >1 :
        decision_net = nn.DataParallel(decision_net,device_ids=list(range(gpu_num)))
        segment_net = nn.DataParallel(segment_net,device_ids=list(range(gpu_num)))

    writer = SummaryWriter(logPath)

    transforms_sample = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
    ])
    transforms_mask = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
    ])

    testloader = DataLoader(
        Dataset(dataSetRoot, opt.Dataset, transforms_sample=transforms_sample, transforms_mask= transforms_mask,  subFold="Test", isTrain=False, label="ALL"),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    lenNum = len(testloader)
    decision_net.eval()
    segment_net.eval()

    # test *****************************************************************
    testiter = testloader.__iter__()
    prob_socre=[]
    label_score = []
    for i in range(0, lenNum):
        print(i)
        testBatch = testiter.__next__()
        torch.cuda.synchronize()

        imgTest = testBatch["img"].cuda()
        label = testBatch["label"].cpu().numpy()

        with torch.no_grad():
            rstTest = segment_net(imgTest)

        segTest = rstTest["seg"]

        with torch.no_grad():
            cTest = decision_net(segTest)

        torch.cuda.synchronize()
        
        for x in range(len(label)):
            prob_socre.append(cTest[x].item())
            label_score.append(label[x])

            if not label[x]:
                save_path_str = os.path.join(saveResultPath,"fromOK")
            else:
                save_path_str = os.path.join(saveResultPath,"fromNG")

            if os.path.exists(save_path_str) == False:
                os.makedirs(save_path_str, exist_ok=True)
            
            save_image(imgTest[x].data, "%s/%d_%d.jpg"% (save_path_str, i,x))
            save_image(segTest[x].data, "%s/%d_%d_%.6f.jpg"% (save_path_str, i, x, cTest[x].item()))
            
    ROC = compute_roc(np.array(label_score), np.array(prob_socre))
    auc,tpr,fpr,roc_best_threshold,ROCcurve = ROC['AUC'],ROC['TPR'],ROC['FPR'],ROC['Thresholds'],ROC['Fig']
    AP = compute_ap(np.array(label_score), np.array(prob_socre))
    ap,pre,rec,pr_best_threshold,PRcurve = AP['AP'],AP['PRE'],AP['REC'],AP['Thresholds'],AP['Fig']

    writer.add_scalar('TestAUC',auc,global_step=(0))
    writer.add_figure('TestROC',ROCcurve,global_step=(0))
    writer.add_scalar('TestAP',ap,global_step=(0))
    writer.add_figure('TestPR',PRcurve,global_step=(0))
    writer.add_scalar('TestROC_Best_threshold',roc_best_threshold,global_step=(0))
    writer.add_scalar('TestPR_Best_threshold',pr_best_threshold,global_step=(0))

        
        
    print("Done")
    #writer.add_pr_curve('PR',np.array(label_score), np.array(prob_socre),global_step=(0))