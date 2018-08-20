"""
Networks for voxelwarp model
"""

# third party
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Conv3D, Activation, Input, UpSampling2D, concatenate, RepeatVector
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras
import keras.backend as K
import numpy as np
import cv2

# local
from new_2D_dense_transform import new_2D_dense_transform
import new_2D_losses
from plot import slices
from normalize import normalize

def unet(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (1024, 1024)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]
            TODO: make this flexible.
        full_size

    """

    # inputs
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])

    # down-sample path.

    x0 = myConv(x_in, enc_nf[0],2)  # 512x512
    x1 = myConv(x0, enc_nf[1],2)  # 256x256
    x2 = myConv(x1, enc_nf[2],2)  # 128x128
    x3 = myConv(x2, enc_nf[3],2)  # 64x64

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = UpSampling2D()(x)
    x = concatenate([x, x2])
    x = myConv(x, dec_nf[1])
    x = UpSampling2D()(x)
    x = concatenate([x, x1])
    x = myConv(x, dec_nf[2])
    x = UpSampling2D()(x)
    x = concatenate([x, x0])
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    if full_size:
        x = UpSampling2D()(x)
        x = concatenate([x, x_in])
        x = myConv(x, dec_nf[5])

        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])

    # transform the results into a flow.
    flow = Conv2D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)

    # warp the source with the flow
    #flow is normalized to [0,1] to be stronger
    print(flow.get_shape().as_list())
    '''
    flow0 = flow[0, :, :, 0]
    flow1 = flow[0, :, :, 1]

    tf.reshape(flow0, [1024,1024])
    tf.reshape(flow1, [1024,1024])
    
    norm_flow0 = tf.div(tf.subtract(flow0, tf.reduce_min(flow0)), tf.subtract(tf.reduce_max(flow0), tf.reduce_min(flow0)))
    norm_flow1 = tf.div(tf.subtract(flow1, tf.reduce_min(flow1)), tf.subtract(tf.reduce_max(flow1), tf.reduce_min(flow1)))

    new_flow = tf.stack([norm_flow0, norm_flow1])
    new_flow = tf.reshape(new_flow, [1024, 1024, 2])
    
   
    print(new_flow.shape)
    '''

    print(len(flow.shape))
    flow_output = normalize()(flow)


    y = new_2D_dense_transform()([src, flow_output])

    print(type(y))
    print(y.get_shape().as_list())
    # prepare model

    y = normalize()(y)

    model = Model(inputs=[src, tgt], outputs=[y, flow_output])

    #images_to_print = [y]
    #titles_input = ['result of transform']
    #cmaps_input = ['gray']
    #slices(images_to_print, titles_input, cmaps_input,  do_colorbars=True,show=True, grid=True)
    return model


def myConv(x_in, nf, strides = 1):
    """
    specific convolution module including convolution followed by leakyrelu
    """

    x_out = Conv2D(nf, kernel_size=3, padding = 'same', kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
