import os
import argparse
from train_decision import train_Decision
from train_segment import train_Segment
from test import test
from auxfunc import setup_seed

def main(opt):
    try:
        setup_seed(opt.Randomseed)
        if opt.Mode not in ('Test','Train'):
            raise(ValueError('Undefine mode'))
        if opt.Network not in ('Segment','Decision'):
            raise(ValueError('Undefine network'))
        if opt.Dataset not in ('KSDD','DAGM','SSD'):
            raise(ValueError('Undefine dataset'))
        
        if opt.Mode == 'Test':
            if not (os.path.isfile(opt.Load_segment_model_dir) and os.path.isfile(opt.Load_decision_model_dir)):
                raise(FileNotFoundError("Can not find the model"))
            else:
                test(opt)
        elif opt.Mode == 'Train':
            if opt.Network == 'Segment':
                train_Segment(opt)
            elif opt.Network == 'Decision':
                train_Decision(opt)
            
    except FileExistsError as e:
        print(e)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #public argument
    parser.add_argument("--Mode", type=str, default='Test', help="Mode, e.g. Train, Test")
    parser.add_argument("--Network", type=str, default='Segment', help="Network, e.g. Segment, Decision")
    parser.add_argument("--Dataset", type=str, default='DAGM', help="Dataset: e.g. KSDD, DAGM,SSD")
    parser.add_argument("--Fold", type=int, default=5, help="The fold of KSDD or the classed of the DAGM: e.g. 0,1,2 for KSDD, 1-6 for DAGM and 0 for SSD")
    parser.add_argument("--Lambda", type=float, default=0.3, help="The precent of dice loss")
    parser.add_argument("--Randomseed", type=int, default=12120953, help="Random seed")

    parser.add_argument("--gpu_ids",default='0,1,2,3',type=str,help='The gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument("--cuda", type=bool, default=True, help="Enable gpu")
    parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")

    parser.add_argument("--img_height", type=int, default=512, help="The size of image height, e.g. KSDD:1408, DAGM:512, SSD:256 ")
    parser.add_argument("--img_width", type=int, default=512, help="The size of image width, e.g. KSDD:512, DAGM:512, SSD:1600")

    #segment argument
    parser.add_argument("--b1", type=float, default=0.9, help="Beta_1 of adam")
    parser.add_argument("--b2", type=float, default=0.999, help="Beta_2 of adam")
    parser.add_argument("--Batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--Lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--Milestones", type=str, default="80,140,180", help="Learning rate decay milestone")
    parser.add_argument("--Gamma", type=float, default=0.1, help="Learning rate decay rate")
    parser.add_argument("--Begin_epoch", type=int, default=0, help="Begin epoch")
    parser.add_argument("--End_epoch", type=int, default=201, help="End epoch")

    parser.add_argument("--Train_with_test", type=bool, default=True, help="SbeNet need to be tested")
    parser.add_argument("--Test_interval", type=int, default=5, help="SbeNet interval of test")
    parser.add_argument("--Train_with_save", type=bool, default=True, help="SbeNet need to be saved")
    parser.add_argument("--Save_interval", type=int, default=5, help="SbeNet interval of save weights")
    
    parser.add_argument("--Need_load_segment_model", type=bool, default=True, help="Need to load pretrain model")
    parser.add_argument("--Load_segment_model_dir", type=str, default='Pretrain_Model/DAGM/F5/Segment_Net.pth', help="Direction of pretrain model")

    parser.add_argument("--Need_load_decision_model", type=bool, default=True, help="Need to load pretrain model")
    parser.add_argument("--Load_decision_model_dir", type=str, default='Pretrain_Model/DAGM/F5/Decision_Net.pth', help="Direction of pretrain model")

    parser.add_argument("--Remark", type=str, default='', help="remark")


    opt = parser.parse_args()

    main(opt)