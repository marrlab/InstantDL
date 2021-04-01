from instantdl.data_generator.metrics4losses import *
import numpy as np

def test_dice_loss():
    x = tf.ones([5,5])
    y_1 = tf.ones([5,5])
    y_2 = tf.zeros([5,5])
    l_1 = dice_coef(x,y_1)
    l_2 = dice_coef(x,y_2)
    assert (l_1.eval(session=K.get_session()) != l_2.eval(session=K.get_session())).all()

def test_dice_crossentropy_loss():
    x = tf.ones([5,5])
    y_1 = tf.ones([5,5])
    y_2 = tf.zeros([5,5])
    l_1 = dice_crossentropy_loss(x,y_1)
    l_2 = dice_crossentropy_loss(x,y_2)
    assert (l_1.eval(session=K.get_session()) != l_2.eval(session=K.get_session())).all()

def test_tversky_loss():
    x = tf.ones([5,5])
    y_1 = tf.ones([5,5])
    y_2 = tf.zeros([5,5])
    l_1 = tversky_loss(x,y_1)
    l_2 = tversky_loss(x,y_2)
    assert (l_1.eval(session=K.get_session()) != l_2.eval(session=K.get_session())).all()

def test_binary_focal_loss_fixed():
    x = tf.ones([5,5])
    y_1 = tf.ones([5,5])
    y_2 = tf.zeros([5,5])
    l_1 = binary_focal_loss_fixed(x,y_1)
    l_2 = binary_focal_loss_fixed(x,y_2)
    assert (l_1.eval(session=K.get_session()) != l_2.eval(session=K.get_session())).all()