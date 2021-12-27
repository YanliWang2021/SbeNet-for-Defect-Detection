from configparser import Error
import os
import glob
import pickle
import argparse
from shutil import copy,copytree
import numpy as np
import cv2
import csv

def rle_to_mask(rle_string,height,width):

    rows, cols = height, width
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img

def make_label(filedir):
    file_dict = {}
    list_file=open(os.path.join(filedir,'train.csv'),'r') 
    list_content = list(csv.reader(list_file))
    
    for i in range(len(list_content)):
        row = list_content[i]
        if i == 0:
            continue
        imageid = row[0]
        encodepixel = row[2]

        if imageid not in file_dict.keys():
            file_dict[imageid] = encodepixel
    
    images = glob.glob(os.path.join(filedir,'train_images/*.jpg'))
        
    for image in images:
        imageid = image.split('/')[-1]

        sample_path = image
        sample = cv2.imread(sample_path)
        h,w,_ = sample.shape

        if imageid in file_dict.keys():
            encodepixel = file_dict[imageid]
            label = rle_to_mask(encodepixel,h,w)
        else:
            label = rle_to_mask(-1,h,w)
        label_savepath = image.replace('.jpg','_label.bmp')            
        cv2.imwrite(label_savepath,label)


def spiltdata(opt):
    data_add = opt.Dataset_dir
    spiltfile_add = opt.Splitfile_dir

    if opt.Dataset == 'KSDD':
        wildcard = "kos*"
    elif opt.Dataset == "DAGM":
        wildcard = f"Class{opt.Fold+1}/*/*.PNG"
    elif opt.Dataset == "SSD":
        make_label(data_add)
        wildcard = "train_images/*.jpg"

    files = glob.glob(os.path.join(data_add,wildcard))

    with open(spiltfile_add, 'rb') as f:
        split_dict =  pickle.load(f)

    for file in files:
        fileneame = file.split('/')[-1]
        if fileneame not in split_dict.keys():
            continue
        targetpath = split_dict[fileneame]
        if not os.path.exists(targetpath):
            os.makedirs(targetpath)
        if opt.Dataset == 'KSDD':
            copytree(file,os.path.join(targetpath,fileneame))
        elif opt.Dataset == 'DAGM':
            copy(file,os.path.join(targetpath,fileneame))
            label_path = file.replace(fileneame,"Label/" + fileneame.replace('.PNG','_label.PNG'))
            if os.path.isfile(label_path):
                mask = cv2.imread(label_path)
                cv2.imwrite(os.path.join(targetpath,fileneame.replace('.PNG','_label.bmp')),mask)
            else:
                img = cv2.imread(file)
                h,w,_ = img.shape
                cv2.imwrite(os.path.join(targetpath,fileneame.replace('.PNG','_label.bmp')),np.zeros((h,w)))
        elif opt.Dataset == 'SSD':
            copy(file,os.path.join(targetpath,fileneame))
            copy(file.replace('.jpg','_label.bmp'),os.path.join(targetpath,fileneame.replace('.jpg','_label.bmp')))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset", type=str, default='DAGM', help="Dataset: e.g. KSDD, DAGM,SSD")
    parser.add_argument("--Fold", type=int, default=5, help="The fold of KSDD or the classed of the DAGM: e.g. 0,1,2 for KSDD, 0-5 for DAGM, 0 for SSD and -1 for all folds/classes")
    parser.add_argument("--Dataset_dir", type=str, default='Data_Origin/DAGM', help="Dataset: e.g. KSDD, DAGM,SSD")
    parser.add_argument("--Splitfile_dir", type=str, default='Data_Origin/DAGM_F5_Split.pkl', help="Dataset split file")
    
    opt = parser.parse_args()
    spiltdata(opt)

    # add_split2={}
    # with open("Data_Origin/DAGM_F5_Split.pkl", 'rb') as f:
    #     split_dict =  pickle.load(f)

    # for i in split_dict.keys():
    #     add_split2[i+'.PNG'] = split_dict[i]
        
    # with open("Data_Origin/DAGM_F5_Split2.pkl", 'wb') as f:
    #     pickle.dump(add_split2, f, pickle.HIGHEST_PROTOCOL)

        


