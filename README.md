# SbeNet-for-Defect-Detection

## 1. Introduction
This repository is the offical implementation of Two-stage Deep Neural Network with Joint Loss and Multi-level Representations for Defect Detection. Up to now, we have uploaded the code and the pre-trained model.

## 2. Note
1. Due to the size of the dataset, we will provide the method of obtaining the dataset as soon as possible. To reproduce the results in our paper, please construct the Data folder indicated in `DataStructure.txt`.
2. For compatibility reasons, the code is recommended to be run under Linux environment.

## 3. Usage
1. Modify `init.ini` according to your device conditions.
2. We have provided quick-star scripts in folder `RunningScript` . Please run the script as follows:
```
sh RunningScript/Test_KSDD_F0.sh
```
3. We use tensorboard to record the results, please use the following command to view it:
```
tensorboard --logdir 'Model/XXXX/Log'
```
Where `XXXX` is the results storage folder, like `KSDD_F0_lambda0.7_Test`.