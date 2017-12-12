import os, datetime
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataLoader import *
from DataLoaderOld import *
from architect_joe import *
from exp_joe import *
import sys
from save import *

joint_ratio_decay = 0.9995
show_mask = True

# Dataset Parameters
print 'Running command: ',sys.argv
Parameters=sys.argv[1]
experiment=exp_joe
if Parameters in experiment:
    settings = experiment[Parameters]
    print 'Parameters: ',experiment[Parameters]
else:
    raise ValueError(Parameters,' no dict of parameters found')

# Training Parameters
learning_rate_class=settings['learning_rate_class']
learning_rate_seg=settings['learning_rate_seg']
training_iters = settings['training_iters']
step_display = settings['step_display']
step_save = settings['step_save']
exp_name = settings['exp_name']
pretrainedStep = settings['pretrainedStep']
selectedmodel= settings['selectedmodel']
plot=settings['plot']

joint_ratio= settings['joint_ratio']
train = settings['train']
validation = settings['validation']
test = settings['test']
batch_size = settings['batch_size']


base_learning_rate_class=learning_rate_class
base_learning_rate_seg=learning_rate_seg
path_save = './save/'+exp_name+'/'
start_from=''


num_seg_class=176

if pretrainedStep > 0:
    start_from = path_save+'-'+str(pretrainedStep)

load_size = 256
fine_size = 224
seg_size = 7
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842], dtype=np.float32)
dropout = 0.5 # Dropout, probability to keep units
# Construct dataloader
opt_data_train_seg = {
    'images_root': './data/images/',   # MODIFY PATH ACCORDINGLY
    'seg_labels_root': './data/seg_labels/',   # MODIFY PATH ACCORDINGLY
    'data_list': './data/new_train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'seg_size': seg_size,
    'data_mean': data_mean,
    'randomize': True,
    'perm' : True,
    'test': False
    }

opt_data_train_class = {
    'data_root': './data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': './data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'perm' : True,
    }

opt_data_val_seg = {
    'images_root': './data/images/',   # MODIFY PATH ACCORDINGLY
    'seg_labels_root': './data/seg_labels/',   # MODIFY PATH ACCORDINGLY
    'data_list': './data/new_val.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'seg_size': seg_size,
    'data_mean': data_mean,
    'randomize': True,
    'perm' : True,
    'test': False
    }

opt_data_val_class = {
    'data_root': './data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': './data/small_val.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'perm' : False,
    }


loader_train_seg = DataLoaderDisk(**opt_data_train_seg)
loader_train_class = DataLoaderDiskOld(**opt_data_train_class)
loader_val_seg = DataLoaderDisk(**opt_data_val_seg)
loader_val_class = DataLoaderDiskOld(**opt_data_val_class)
#loader_test = DataLoaderDiskOld(**opt_data_test)

print ('finish loading data')
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
seg_labels = tf.placeholder(tf.float32, [None, seg_size, seg_size, num_seg_class])
y = tf.placeholder(tf.int64, None)
lrc = tf.placeholder(tf.float32, None)
lrs = tf.placeholder(tf.float32, None)

keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
if selectedmodel=='vgg_seg1':
    myModel = vgg_seg1(x, y, seg_labels, keep_dropout, train_phase)
elif selectedmodel=='vgg_seg1_mask':
    myModel = vgg_seg1_mask(x, y, seg_labels, keep_dropout, train_phase)
elif selectedmodel=='vgg_seg1_fix_mask':
    myModel = vgg_seg1_fix_mask(x, y, seg_labels, keep_dropout, train_phase)
else:
    raise ValueError(selectedmodel,' no such model, end of the program')

# Define loss and optimizer
prob = myModel.prob_class
loss_class = myModel.loss_class
loss_seg = myModel.loss_seg
loss = loss_seg + loss_class

class_optimizer = tf.train.AdamOptimizer(learning_rate=lrc).minimize(loss_class)
seg_optimizer = tf.train.AdamOptimizer(learning_rate=lrs).minimize(loss_seg)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# Launch the graph
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    step = 0

    if train:
        train_class_accs=[]
        train_class_losses = []
        train_seg_accs=[]
        train_seg_losses = []

        val_class_accs=[]
        val_class_losses = []
        val_seg_accs = []
        val_seg_losses = []

        seg_labels_batch_class = np.zeros([batch_size, seg_size, seg_size, num_seg_class])
        while step < training_iters:

            #TODO: decrease learning rate
            if step > 1000:
                joint_ratio = joint_ratio * joint_ratio_decay
            # Load a batch of training data
            
            images_batch_seg, seg_labels_batch_seg, _, labels_batch_seg = loader_train_seg.next_batch(batch_size)

            images_batch_class, labels_batch_class = loader_train_class.next_batch(batch_size)
            if step % step_display == 0:
                #TODO: show mask
                if show_mask:
                    weight_mask = tf.get_default_graph().get_tensor_by_name("weight_mask")
                    mask = tf.sess.run(weight_mask, feed_dict={})
                    print "MASK: ", mask
                print('[%s]:' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                # Calculate batch loss and accuracy on class training set
                l, lc, ls, acc1, acc5 = sess.run([loss,loss_class,loss_seg, accuracy1, accuracy5], 
                        feed_dict={lrs:learning_rate_seg, 
                            lrc:learning_rate_class, 
                            x: images_batch_class, 
                            y: labels_batch_class, 
                            seg_labels: seg_labels_batch_class, 
                            keep_dropout: 1., 
                            train_phase: False}
                        ) 
                print('-Iter ' + str(step) + 
                        ', Class Train Dataset' + 
                        ', Training Loss= ' + '{:.6f}'.format(l) +
                        ', Class Loss= ' + '{:.6f}'.format(lc) + 
                        ', Seg Loss= ' + '{:.6f}'.format(ls) + 
                        ', Accuracy Top1 = ' + '{:.4f}'.format(acc1) + 
                        ', Top5 = ' + '{:.4f}'.format(acc5))
                train_class_accs.append(acc5)
                train_class_losses.append(lc)

                 # Calculate batch loss and accuracy on seg training set
                l, lc, ls, acc1, acc5 = sess.run([loss,loss_class,loss_seg, accuracy1, accuracy5], 
                        feed_dict={lrs:learning_rate_seg,
                            lrc:learning_rate_class,
                            x: images_batch_seg, 
                            y: labels_batch_seg, 
                            seg_labels: seg_labels_batch_seg, 
                            keep_dropout: 1., 
                            train_phase: False}
                        ) 
                print('-Iter ' + str(step) + 
                        ', Seg Train Dataset' + 
                        ', Training Loss= ' + '{:.6f}'.format(l) +
                        ', Class Loss= ' + '{:.6f}'.format(lc) + 
                        ', Seg Loss= ' + '{:.6f}'.format(ls) + 
                        ', Accuracy Top1 = ' + '{:.4f}'.format(acc1) + 
                        ', Top5 = ' + '{:.4f}'.format(acc5))
                train_seg_accs.append(acc5)
                train_seg_losses.append(l)
                



                # Evaluate on the class validation set
                num_batch = loader_val_class.size()//batch_size+1
                acc1_total = 0.
                acc5_total = 0.
                loss_total = 0.
                num_total = 0.
                loader_val_class.reset()

                for i in range(num_batch):
                    images_batch_val, labels_batch_val = loader_val_class.next_batch(batch_size)   
                    seg_labels_batch_val = np.zeros([batch_size, seg_size, seg_size, num_seg_class])
                        
                    l, acc1, acc5 = sess.run([loss_class, accuracy1, accuracy5], 
                            feed_dict={lrs:learning_rate_seg,
                                lrc:learning_rate_class,
                                x: images_batch_val, 
                                y: labels_batch_val, 
                                seg_labels: seg_labels_batch_val, 
                                keep_dropout: 1., 
                                train_phase: False}
                            )
                    acc1_total += acc1 * len(labels_batch_val)
                    acc5_total += acc5 * len(labels_batch_val)
                    loss_total += l * len(labels_batch_val)
                    num_total += len(labels_batch_val)
                
                acc1 = acc1_total/num_total
                acc5 = acc5_total/num_total
                l = loss_total/num_total
                print('Validation Class: Accuracy Top1 = ' + '{:.4f}'.format(acc1) + ', Top5 = ' + '{:.4f}'.format(acc5) + ',Loss = ' + '{:.4f}'.format(l))
                val_class_accs.append(acc5)
                val_class_losses.append(l)


                # Evaluate on the seg validation set
                num_batch = loader_val_seg.size()//batch_size+1
                acc1_total = 0.
                acc5_total = 0.
                loss_total = 0.
                num_total = 0.
                loader_val_seg.reset()

                for i in range(num_batch):
                    images_batch_val, seg_labels_batch_val, _, labels_batch_val = loader_val_seg.next_batch(batch_size)   
                        
                    l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], 
                            feed_dict={lrs:learning_rate_seg,
                                lrc:learning_rate_class,
                                x: images_batch_val, 
                                y: labels_batch_val, 
                                seg_labels: seg_labels_batch_val, 
                                keep_dropout: 1., 
                                train_phase: False}
                            )
                    acc1_total += acc1 * len(labels_batch_val)
                    acc5_total += acc5 * len(labels_batch_val)
                    loss_total += l * len(labels_batch_val)
                    num_total += len(labels_batch_val)
                
                acc1 = acc1_total/num_total
                acc5 = acc5_total/num_total
                l = loss_total/num_total
                print('Validation Seg: Accuracy Top1 = ' + '{:.4f}'.format(acc1) + ', Top5 = ' + '{:.4f}'.format(acc5) + ',Loss = ' + '{:.4f}'.format(l))
                val_seg_accs.append(acc5)
                val_seg_losses.append(l)


                print "VALIDATION ACCURACIES: ", val_class_accs
                print "DATASET WITH ONLY TRAINING ACCURACIES: ", train_class_accs

                if plot:
                    a=np.arange(1,len(val_class_accs)+1,1)*step_display
                    
                    fig = plt.figure()
                    plt.plot(a,train_class_accs,'-',label='Training Dataset with only class')
                    plt.plot(a,train_seg_accs,'-',label='Training Dataset with seg')
                    plt.xlabel('Iteration')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    fig.savefig('./fig/pic_train_accuracy_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)

                    fig = plt.figure()
                    plt.plot(a,val_class_accs,'-',label='Validation Dataset with only class')
                    plt.plot(a,val_seg_accs,'-',label='Validation Dataset with seg')
                    plt.xlabel('Iteration')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    fig.savefig('./fig/pic_val_accuracy_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)

                    fig = plt.figure()
                    plt.plot(a,train_seg_losses,'-',label='Seg')
                    plt.plot(a,train_class_losses,'-',label='Class')
                    plt.xlabel('Iteration')
                    plt.ylabel('Train Loss')
                    plt.legend()
                    fig.savefig('./fig/pic_train_loss_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)

                    fig = plt.figure()
                    plt.plot(a,val_seg_losses,'-',label='Seg')
                    plt.plot(a,val_class_losses,'-',label='Class')
                    plt.xlabel('Iteration')
                    plt.ylabel('Validation Loss')
                    plt.legend()
                    fig.savefig('./fig/pic_val_loss_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)
                    print 'finish saving figure to view'
            
            # Run optimization op (backprop)

            flip = np.random.uniform(0, 1)
            if flip<=joint_ratio:
                images_batch, seg_labels_batch, labels_batch = images_batch_seg, seg_labels_batch_seg, labels_batch_seg
                sess.run(class_optimizer, 
                        feed_dict={lrs:learning_rate_seg,
                            lrc:learning_rate_class,
                            x: images_batch, 
                            y: labels_batch, 
                            seg_labels: seg_labels_batch, 
                            keep_dropout: dropout, 
                            train_phase: True}
                        )

                sess.run(seg_optimizer, 
                        feed_dict={lrs:learning_rate_seg,
                            lrc:learning_rate_class,
                            x: images_batch, 
                            y: labels_batch, 
                            seg_labels: seg_labels_batch, 
                            keep_dropout: dropout, 
                            train_phase: True}
                        )
            else:
                images_batch, seg_labels_batch, labels_batch = images_batch_class, seg_labels_batch_class, labels_batch_class
                sess.run(class_optimizer, 
                        feed_dict={lrs:learning_rate_seg,
                            lrc:learning_rate_class,
                            x: images_batch, 
                            y: labels_batch, 
                            seg_labels: seg_labels_batch, 
                            keep_dropout: dropout, 
                            train_phase: True}
                        )
            step += 1
            
            # Save model
            if step % step_save == 0 or step==1:
                saver.save(sess, path_save, global_step=step+pretrainedStep)
                print('Model saved at Iter %d !' %(step))
        print('Optimization Finished!')
