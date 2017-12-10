import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels
import CNNModels2

class alexnet_model:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits=alexnet(x, keep_dropout, train_phase)
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.logits))

class vgg_model:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits=VGG(x, keep_dropout, train_phase)
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.logits))

class vgg_simple_model:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits=VGG16_Simple(x, keep_dropout, train_phase)
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.logits))


class vgg_seg:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits=VGG_BN(x, keep_dropout, train_phase)
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.logits))

class vgg_bn_seg2:
    def __init__(self, x, y, keep_dropout, lam, train_phase):
        self.logits=VGG_BN(x, keep_dropout, train_phase)
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.logits))
         