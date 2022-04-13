from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
from data_gen import img_dataset

def conv_block(inp, filters, act=True):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    if act:
        x = Activation('relu')(x)
    return x


def encoder_block(inp, filters):
    x = conv_block(inp, filters)
    p = MaxPooling2D(pool_size=(2, 2))(x)
    return x, p


def attention_block(l_layer, h_layer):  # Attention Block
    phi = Conv2D(h_layer.shape[-1], (1, 1), padding='same')(l_layer)
    theta = Conv2D(h_layer.shape[-1], (1, 1), strides=(2, 2), padding='same')(h_layer)
    x = tf.keras.layers.add([phi, theta])
    x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.multiply([h_layer, x])
    x = BatchNormalization(axis=3)(x)
    return x


def decoder_block(inp, filters, concat_layer, act=True):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inp)
    concat_layer = attention_block(inp, concat_layer)
    x = concatenate([x, concat_layer])
    x = conv_block(x, filters, act)
    return x


def attention_unet(input_size):
    inputs = Input(input_size)
    d1, p1 = encoder_block(inputs, 64)
    d2, p2 = encoder_block(p1, 128)
    d3, p3 = encoder_block(p2, 256)
    d4, p4 = encoder_block(p3, 512)
    b0 = conv_block(p4, 1024, act=False)
    b1 = Activation('relu')(b0)
    e2 = decoder_block(b1, 512, d4)
    e3 = decoder_block(e2, 256, d3)
    e4 = decoder_block(e3, 128, d2)
    e5 = decoder_block(e4, 64, d1, act=False)
    e6 = Activation('relu')(e5)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(e6)

    return Model(inputs=[inputs], outputs=[outputs], name='AttentionUnet')
