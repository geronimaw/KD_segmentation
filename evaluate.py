import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_gen import img_dataset
from data_prep import data_preparation
from losses import *

def eval_model(model_wts, df_test, custom_objects):
    model = load_model(model_wts, custom_objects=custom_objects)
    test = img_dataset(df_test[['filepath', 'mask']], 'filepath', 'mask', dict(), 32)
    # print(test)
    model.evaluate(test, steps=len(df_test)/32)
    a = np.random.RandomState(seed=42)
    indexes = a.randint(1, len(df_test[df_test['res']==1]), 10)
    for i in indexes:
        img = cv2.imread(df_test[df_test['res']==1].reset_index().loc[i, 'filepath'])
        img = cv2.resize(img, (256, 256))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred = model.predict(img)

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(cv2.imread(df_test[df_test['res']==1].reset_index().loc[i, 'mask'])))
        plt.title('Original Mask')
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(pred) > .5)
        plt.title('Prediction')
        plt.show()


if __name__ == '__main__':
    state = 0
    parser = argparse.ArgumentParser(description='Specify the path to the dataset.')
    parser.add_argument('dataset_path', type=str, nargs='+', help='no quotations marks. add a slash at the end.')
    # parser.add_argument('model_weights', type=str)

    args = parser.parse_args()
    dataset_path = str(args.dataset_path)[2:-2]
    # model_weights = str(args.model_weights)#[2:-2]
    model_weights = '/home/alecacciatore/attentionUNet/src/attUnet_wts1.hdf5'

    if not os.path.isdir(dataset_path):
        print("the path does not exist")

    elif not (os.path.isdir(dataset_path + "lgg-mri-segmentation") and os.path.isdir(dataset_path + "kaggle_3m")):
        print("the path does not contain the required dataset")

    else:
        df_test = data_preparation(dataset_path, mode='test')
        eval_model(model_weights, df_test, {'dice_loss': dice_loss, 'iou': iou})
