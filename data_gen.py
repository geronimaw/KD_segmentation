#Function to create image datasets using keras flow_from_dataframe
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def img_dataset(df_inp, path_img, path_mask, aug_args, batch):
    img_gen = ImageDataGenerator(rescale=1./255., **aug_args)

    df_img = img_gen.flow_from_dataframe(dataframe=df_inp, x_col=path_img, class_mode=None, batch_size=batch,
                                         color_mode='rgb', seed=1, target_size=(256, 256))
    df_mask = img_gen.flow_from_dataframe(dataframe=df_inp, x_col=path_mask, class_mode=None, batch_size=batch,
                                          color_mode='grayscale', seed=1, target_size=(256, 256))
    data_gen = zip(df_img, df_mask)
    return data_gen
