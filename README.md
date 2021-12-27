# SbeNet-for-Defect-Detection

## 1. Introduction
This repository is the offical implementation of Two-stage Deep Neural Network with Joint Loss and Multi-level Representations for Defect Detection. Up to now, we have uploaded the code and the pre-trained model.

## 2. Note
1. To reproduce the results in our paper, please construct the Data folder indicated in `DataStructure.txt`.
2. For compatibility reasons, the code is recommended to be run under Linux environment.

## 3. Usage
1. Modify `init.ini` according to your device conditions.
2. Download the dataset and construct the Data folder indicated in `DataStructure.txt`. For convenience we provide `QuickSplit.py` to build `Data` folder quickly (refer to Section 4).
3. We have provided quick-star scripts in folder `RunningScript` . Please run the script as follows:
```
sh RunningScript/Test_KSDD_F0.sh
```
3. We use tensorboard to record the results, please use the following command to view it:
```
tensorboard --logdir 'Model/XXXX/Log'
```
Where `XXXX` is the results storage folder, like `KSDD_F0_lambda0.7_Test`.

## 4. Build Data Folder
1. Down load and unpack the dataset. Please ensure the structure of dataset folder as follow:
   ```
   KSDD/
    ├── kos01
    ├── kos02
    ├── kos03
    ├──   .
    ├──   .
    ├──   .
    └── kos50
   ----------------------------
   Other/DAGM/
    ├── Class1
    │   ├── Test
    │   │   └── Label
    │   └── Train
    │       └── Label
    ├── Class2
    │   ├── Test
    │   │   └── Label
    │   └── Train
    │       └── Label
    │    .
    │    .
    │    .
    └── Class6
        ├── Test
        │   └── Label
        └── Train
            └── Label
    ----------------------------
    SSD/
    ├── sample_submission.csv
    ├── test_images
    ├── train.csv
    └── train_images
    
   ```
2. Run the `QuickSplit.py` as follos:
   ```
   python QuickSplit.py --Dataset 'KSDD' --Fold 0 --Dataset_dir 'KSDD' --Splitfile_dir 'Data_Split/KSDD_F0_Split.pkl'
   ```
   Where the split file for each datasets are provided in `Data_Split` .