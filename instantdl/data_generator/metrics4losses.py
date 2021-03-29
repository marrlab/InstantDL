# File for custom loss functions

from keras import backend as K
from keras.losses import categorical_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)# In this file loss function, which are not in keras are implemented


def dice_crossentropy_loss(y_true, y_pred):
    if K.sum(y_true) != 0:
        vol_loss = K.abs(K.sum(y_true)-K.sum(y_pred))/(K.sum(y_true)*2)
    else:
        vol_loss = 0
    return dice_loss(y_true, y_pred) + categorical_crossentropy(y_true, y_pred) + vol_loss