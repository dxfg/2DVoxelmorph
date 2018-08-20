"""
train atlas-based alignment with voxelmorph
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model
import matplotlib.pyplot as plt

# project imports
import new_2D_datagenerators
import new_2D_networks
import new_2D_losses


vol_size = (1024, 1024)
base_data_dir = 'C:\\Users\\Sriram\\Documents\\Testing-VMorph\\src\\training_images\\*.npz'
train_vol_names = glob.glob(base_data_dir)
random.shuffle(train_vol_names)

#atlas = np.load(r'../data/atlas_2D.npz')
#atlas_vol = atlas['atlas2D']
atlas = np.load(r'../src/test_images/test_image_18.npz')
atlas_vol = atlas['image2D']
atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape+(1,))

def train(model,save_name, gpu_id, lr, n_iterations, reg_param, model_save_iter):

    model_dir = '../models/' + save_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))


    # UNET filters
    nf_enc = [16,32,32,32]
    if(model == 'vm1'):
        nf_dec = [32,32,32,32,8,8,2]
    else:
        nf_dec = [32,32,32,32,32,16,16,2]

    with tf.device(gpu):
        model = new_2D_networks.unet(vol_size, nf_enc, nf_dec)
        model.compile(optimizer=Adam(lr=lr), loss=[
                      new_2D_losses.cc2D(), new_2D_losses.gradientLoss2D()], loss_weights=[1.0, reg_param])
        # model.load_weights('../models/udrnet2/udrnet1_1/120000.h5')

    train_example_gen = new_2D_datagenerators.example_gen(train_vol_names)
    zero_flow = np.zeros((1, vol_size[0], vol_size[1], 2))

    cc_loss = []
    grad_loss = []

    for step in range(1, n_iterations):

        X = train_example_gen.__next__()[0]
        train_loss = model.train_on_batch(
            [X, atlas_vol], [atlas_vol, zero_flow])

        if not isinstance(train_loss, list):
            train_loss = [train_loss]

        printLoss(step, 1, train_loss)
        train_loss = [train_loss]
        cc_loss.append(train_loss[0][0])
        #print(train_loss[0][0])
        #print(train_loss[0][2])
        grad_loss.append(train_loss[0][2])

        if(step % model_save_iter == 0):
            model.save(model_dir + '/' + str(step) + '.h5')

    
    plt.figure(1)
    plt.plot(cc_loss)
    plt.xlabel('Step') 
    plt.ylabel('CC Loss')
    plt.plot(cc_loss, 'ro', color='r',linewidth=1.0, label="CC Loss")
    plt.grid()    
    plt.figure(2)
    plt.plot(grad_loss)
    plt.xlabel('Step') 
    plt.ylabel('Gradient Loss')
    plt.plot(grad_loss, 'ro', color='r',linewidth=1.0, label="Gradient Loss")
    plt.grid()   
    plt.show()    


def printLoss(step, training, train_loss):
    s = str(step) + "," + str(training)

    if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print(s)
    sys.stdout.flush()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model", 
                        choices=['vm1','vm2'],default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--save_name", type=str,
                        dest="save_name", default= 'xray_test_image_atlas_reg1.0')
    parser.add_argument("--gpu", type=int,default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float, 
                        dest="lr", default=1e-4,help="learning rate") 
    parser.add_argument("--iters", type=int, 
                        dest="n_iterations", default=1001,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float, 
                        dest="reg_param", default=.5,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=50, 
                        help="frequency of model saves")

    args = parser.parse_args()
    train(**vars(args))
