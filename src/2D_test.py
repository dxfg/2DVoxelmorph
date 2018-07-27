# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
import cv2 as cv2
from matplotlib import pyplot as pp

# project
import 2D_networks
import 2D_datagenerators


def test(model_name, iter_num, gpu_id, vol_size=(1024,1024), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16,2]):
  """
  test

  nf_enc and nf_dec
  #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  

  gpu = '/gpu:' + str(gpu_id)

  # Anatomical labels we want to evaluate
  labels = sio.loadmat('../data/labels.mat')['labels'][0]

  atlas = np.load(r'../data/atlas_2D.npz')
  atlas_vol = atlas['atlas2D']
  atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape+(1,))

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  set_session(tf.Session(config=config))

  # load weights of model
  with tf.device(gpu):
    net = 2D_networks.unet(vol_size, nf_enc, nf_dec)
    net.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

  xx = np.arange(vol_size[1])
  yy = np.arange(vol_size[0])
  grid = np.rollaxis(np.array(np.meshgrid(xx, yy)), 0, 3)

  X_vol = 2D_datagenerators.load_example_by_name(r'/Your image here/')

  with tf.device(gpu):
    pred = net.predict([X_vol[0], atlas_vol])

  # Warp segments with flow
  flow = pred[1][0, :, :, :]

  sample = flow+grid

  X_seg = np.zeros((1024,1024))
  X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))

  warp_seg = interpn((yy, xx), X_seg[0, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

  w = 10
  h = 10
  fig = pp.figure(figsize=(8, 8))
  columns = 3
  rows = 1
  a = np.load(r'/Your image here/')
  im_arr = a['image2D'] 
  fig.add_subplot(rows, columns, 1)
  pp.imshow(im_arr, cmap = 'gray')
  fig.add_subplot(rows, columns, 2)
  image1 = flow[:, :, 0]
  pp.imshow(image1, cmap = 'gray')
  fig.add_subplot(rows, columns, 3)
  image2 = flow[:, :, 1]
  pp.imshow(image2, cmap = 'gray')
  pp.show()
  
  
if __name__ == "__main__":
  test(sys.argv[1], sys.argv[2], sys.argv[3])
