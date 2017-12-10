import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels
import CNNModels2

class alexnet_model:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits_class=alexnet(x, keep_dropout, train_phase)
        self.loss_class =loss_class(y,self.logits_class)

class vgg_simple_model:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits_class=VGG16_Simple(x, keep_dropout, train_phase)
        self.loss_class =loss_class(y,logits_class)

class vgg_model:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits_class = VGG(x, keep_dropout, train_phase, num_classes=100)
        self.loss_class =loss_class(y,self.logits_class)

class vgg_bn_seg2:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits_class=VGG(x, keep_dropout, train_phase, num_classes=100, batch_norm=True)
        self.loss_class =loss_class(y,self.logits_class)

class vgg_seg2:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits_class，logits_seg=VGG(x, keep_dropout, train_phase,num_classes=100, seg=True, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)

class vgg_bn_seg2:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits_class，logits_seg=VGG(x, keep_dropout, train_phase,num_classes=100, batch_norm=True, seg=True, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)

def loss_class(y,logits):
    return loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
