import os
import pandas as pd
from glob import glob
import numpy as np
import cv2
from itertools import chain

def create_dataset(patient, start, end, dataset_type, dataset_path):
    train_files = []
    mask_files=[]
    c=0
    for i,p in enumerate(patient[start:end]):
        # print("i", i)
        # print("p", p)
        vals=[]
        mask_files.append(glob(dataset_path + '/lgg-mri-segmentation/kaggle_3m/'+p+'/*_mask*'))
        for m in mask_files[i]:
            vals.append(np.max(cv2.imread(m)))
        if max(vals)==0:
            print(f'patient { p } has no tumor')
            c+=1
    if c==0:
        print(f'Each patient in {dataset_type} dataset has brain tumor')
    mask_files=list(chain.from_iterable(mask_files))
    for m in mask_files:
        train_files.append(m.replace('_mask',''))
    df = pd.DataFrame(data={"filepath": train_files, 'mask' : mask_files})
    return df

#function to view pixel range of an image from the dataset. Note: The function selects a random image everytime it's called
def pixel_value_counts(col, df_train, end):
    p=np.random.randint(0, end)
    img=cv2.imread(df_train[col].loc[p])
    unique, counts = np.unique(img, return_counts=True)
    print(f'showing pixel value counts for image {p}')
    print(np.asarray((unique, counts)).T)


def data_preparation(dataset_path, mode):
    patient = []
    for d in os.listdir(dataset_path + 'lgg-mri-segmentation/kaggle_3m/'):
        if d != 'data.csv' and d != 'README.md':
            patient.append(d)

    #lenghts for training,validation and testing datasets
    a = int(0.9*len(patient))
    b = int(0.8*a)

    #Creating datasets and finding whether there's any patient with no tumor in the provided data
    if mode == 'train':
        df_train = create_dataset(patient, 0, b, 'training', dataset_path)
        df_val = create_dataset(patient, b, a, 'validation', dataset_path)

        # pixel_value_counts('filepath',len(df_train)) #can use test and validation sets too
        pixel_value_counts('mask', df_train, len(df_train))
        df_train.sample(5, random_state=42)
        return df_train, df_val

    elif mode == 'test':
        df_test = create_dataset(patient, a, len(patient), 'testing', dataset_path)

        #Marking masks in test dataset as 0 and 1. Will be useful while making final plots
        for i in range(0, len(df_test)):
            arr = np.where(cv2.imread(df_test['mask'].loc[i])==255, 1, 0)
            v = np.max(arr)
            if v == 1:
                df_test.loc[i, 'res'] = 1
            else:
                df_test.loc[i, 'res'] = 0

        return df_test
