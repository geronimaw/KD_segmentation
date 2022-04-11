import os
import matplotlib.pyplot as plt
import pandas as pd
# import cv2
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_prep import data_preparation
from losses import *
from teacher import attention_unet
from student import non_attention_unet
from data_gen import img_dataset
from distiller import Distiller


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    state = 0
    parser = argparse.ArgumentParser(description='Specify the path to the dataset.')
    parser.add_argument('dataset_path', type=str, nargs='+', help='no quotations marks. add a slash at the end.')

    args = parser.parse_args()
    dataset_path = str(args.dataset_path)[2:-2]

    if not os.path.isdir(dataset_path):
        print("the path does not exist")
    elif not (os.path.isdir(dataset_path + "lgg-mri-segmentation") and os.path.isdir(dataset_path + "kaggle_3m")):
        print("the path does not contain the required dataset")
    else:
        df_train, df_val = data_preparation(dataset_path, mode='train')
        # print(df_train)
        state = 1

    if state == 1:
        # f, ax = plt.subplots(3, 3, figsize=(14, 8))
        # ax = ax.flatten()
        # for j in range(0, 9):
        #     i = 1453 + j
        #     img = cv2.imread(df_train['filepath'].loc[i])
        #     msk = cv2.imread(df_train['mask'].loc[i])
        #     ax[j].imshow(msk)
        #     ax[j].imshow(img, alpha=0.7)
        # plt.show()

        #### Define model
        opt = Adam(learning_rate=1e-4, epsilon=None, amsgrad=False, beta_1=0.9, beta_2=0.99)
        teacher = attention_unet((256, 256, 3))
        teacher.load_weights("E:/sinc/segmentation/attentionUNet/1.attUNet/attUnet_wts1.hdf5")

        student = non_attention_unet((256, 256, 3))

        dist = Distiller()
        distiller = dist(student=student, teacher=teacher)

        distiller.compiles(
            optimizer=opt,
            student_loss_fn=dice_loss,
            distillation_loss_fn=dice_loss,
            metrics=[iou]
        )

        #### Data augmentation
        augmentation_args = dict(rotation_range=0.2,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=0.05,
                                 zoom_range=0.05,
                                 fill_mode='nearest')
        batch = 16

        callbacks = [ModelCheckpoint("attUnet_wts1.hdf5", verbose=1, save_best_only=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, min_lr=1e-6),
                     EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15)]
        train = img_dataset(df_train, 'filepath', 'mask', augmentation_args, batch)
        val = img_dataset(df_val, 'filepath', 'mask', dict(), batch)

        history = distiller.fit(train, validation_data=val,
                            steps_per_epoch=len(df_train) / batch,
                            validation_steps=len(df_val) / batch,
                            epochs=25,
                            callbacks=callbacks)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Training Loss Curve")
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("loss_curves")
        # plt.show()

        hist_df = pd.DataFrame(history.history)
        hist_json_file = os.path.abspath('') + '/history.json'
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
