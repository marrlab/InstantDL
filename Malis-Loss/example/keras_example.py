### use keras with backend tensorflow as a test example
### tensorflow version 2.1.0

import keras
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger
import tensorflow as tf
import numpy as np
import h5py
import malis as m
from malis.malis_keras import malis_loss2d


##### Loading data 
file_path_training_data = '...'  #please enter file path to training data here
f=h5py.File(file_path_training_data,'r')  

data_ch = f['train']
seg_gt = f['groundtruth']
seg_gt = np.expand_dims(seg_gt,axis=-1)

train_data = data_ch[:2000]
train_gt = seg_gt[:2000]

val_data = data_ch[2000:]
val_gt = seg_gt[2000:]

'''
Unet from:
InstandDL: An easy and convenient deep learning pipeline for image segmentation and classification
https://github.com/marrlab/InstantDL
'''

from keras.models import *
from keras.layers import Input, Conv2D, Conv3D, LeakyReLU, BatchNormalization, Dropout, MaxPooling2D, concatenate, MaxPooling3D, UpSampling2D, UpSampling3D
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
from keras.losses import *

class UNetBuilder(object):
    '''
    Builds a 2D or 3D U-Net for image segmentation following the publicaiton from Ronneberger et al.: https://arxiv.org/abs/1505.04597
    The layers consist of a convolution followed by a LeackyReLu activation and a Batch Normalization
    Dropout is integraeted with a dropout rate of 20% during training to increase generalizability
    Padding has bee set to "same"
    '''
    def unet2D(input_size, Dropout_On, base_n_filters = 32):
        print("started UNet")
        inputs = Input(input_size)
        
        conv1 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        print("finished 1. convolution")
        conv1 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        drop1 = Dropout(0.2)(conv1, training = Dropout_On)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

        conv2 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        drop2 = Dropout(0.2)(conv2, training = Dropout_On)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

        conv3 = Conv2D(base_n_filters*4, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = Conv2D(base_n_filters*4, 3,  padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        drop3 = Dropout(0.2)(conv3, training = Dropout_On)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

        conv4 = Conv2D(base_n_filters*8, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        conv4 = Conv2D(base_n_filters*8, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        drop4 = Dropout(0.2)(conv4, training = Dropout_On)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(base_n_filters*16, 3, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=0.2)(conv5)
        conv5 = Conv2D(base_n_filters*16, 3, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU(alpha=0.2)(conv5)
        drop5 = Dropout(0.2)(conv5, training = Dropout_On)

        up6 = Conv2D(base_n_filters*8, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(base_n_filters*8, 3,activation = 'relu',  padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = LeakyReLU(alpha=0.1)(conv6)
        conv6 = Conv2D(base_n_filters*8, 3, activation = 'relu',  padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = LeakyReLU(alpha=0.1)(conv6)

        up7 = Conv2D(base_n_filters*4, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([drop3, up7], axis=3)
        conv7 = Conv2D(base_n_filters*4, 3, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = LeakyReLU(alpha=0.1)(conv7)
        conv7 = Conv2D(base_n_filters*4, 3, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = LeakyReLU(alpha=0.1)(conv7)

        up8 = Conv2D(base_n_filters*2, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([drop2, up8], axis=3)
        conv8 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = LeakyReLU(alpha=0.1)(conv8)
        conv8 = Conv2D(base_n_filters*2, 3, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = LeakyReLU(alpha=0.1)(conv8)

        up9 = Conv2D(base_n_filters, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([drop1, up9], axis=3)

        conv9 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = LeakyReLU(alpha=0.1)(conv9)
        conv9 = Conv2D(base_n_filters, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(base_n_filters, 3,padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = LeakyReLU(alpha=0.1)(conv9)

        conv10 = Conv2D(2, 1, activation='sigmoid')(conv9)  # Simple segmentation with only one label

        model2D = Model(inputs=inputs, outputs=conv10)
        print("shape input UNet:", np.shape(inputs))
        print("shape output UNet:", np.shape(conv10))
        opt = Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.99)
        model2D.compile(optimizer=opt, loss = malis_loss2d)

        return model2D
    
inputShape = (data_ch.shape[1], data_ch.shape[2], data_ch.shape[3])

Early_Stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=0)
model_checkpoint = ModelCheckpoint(filepath = './results/malis_model2-{epoch:02d}.h5', verbose=1, save_best_only=True) 
callbacks_list = [model_checkpoint, Early_Stopping]
model = UNetBuilder.unet2D(inputShape, Dropout_On = True)
model.summary()
model.fit(train_data,train_gt,epochs=50,verbose=1,batch_size=4,validation_data=[val_data,val_gt],callbacks = callbacks_list)

