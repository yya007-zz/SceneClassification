import os, datetime
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataLoader import *
from architect import *
from exp import *
import sys
from save import *

print('[%s]:' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# Dataset Parameters
print 'Running command: ',sys.argv
Parameters=sys.argv[1]
experiment=exp
if Parameters in experiment:
    settings = experiment[Parameters]
    print 'Parameters: ',experiment[Parameters]
else:
    raise ValueError(Parameters,' no dict of parameters found')

debug = False
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
lr_decay=settings['lr_decay']

joint_ratio= settings['joint_ratio']
joint_ratio_decay = settings['joint_ratio_decay']
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

opt_data_train = {
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
    'randomize': False,
    'perm' : False,
    'test': False
    }

opt_data_test = {
    'data_root': './data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': './data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'perm' : False
    }

opt_data_val = {
    'data_root': './data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': './data/small_val.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'perm' : False,
    }

loader_train_seg = DataLoaderDiskSeg(**opt_data_train_seg)
loader_train = DataLoaderDiskClass(**opt_data_train)
loader_val = DataLoaderDiskClass(**opt_data_val)
loader_val_seg = DataLoaderDiskSeg(**opt_data_val_seg)
loader_test = DataLoaderDiskClass(**opt_data_test)

print ('finish loading data')
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
seg_labels = tf.placeholder(tf.float32, [None, seg_size, seg_size, num_seg_class])
obj_class = tf.placeholder(tf.float32, [None, num_seg_class])
y = tf.placeholder(tf.int64, None)
lrc = tf.placeholder(tf.float32, None)
lrs = tf.placeholder(tf.float32, None)

keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
if selectedmodel=='vgg':
    myModel = vgg_model(x, y, seg_labels, obj_class, keep_dropout, train_phase)
elif selectedmodel=='vgg_bn':
    myModel = vgg_bn_model(x, y, seg_labels, obj_class, keep_dropout, train_phase)
elif selectedmodel=='alexnet':
    myModel = alexnet_model(x, y, seg_labels, obj_class, keep_dropout, train_phase)
elif selectedmodel=='vgg_objnet':
    myModel = vgg_objnet(x, y, seg_labels, obj_class, keep_dropout, train_phase)    
elif selectedmodel=='vgg_segnet':
    myModel = vgg_segnet(x, y, seg_labels, keep_dropout, train_phase)
else:
    raise ValueError(selectedmodel,' no such model, end of the program')

# Define loss and optimizer
logits= myModel.logits_class
loss_seg = myModel.loss_seg
loss_class = myModel.loss_class
loss = loss_seg+loss_class

class_optimizer = tf.train.AdamOptimizer(learning_rate=lrc).minimize(loss_class)
seg_optimizer = tf.train.AdamOptimizer(learning_rate=lrs).minimize(loss_seg)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    def use_evaluation(loader, mode):
        t=time.time()
        # Evaluate on the whole validation set
        print('Evaluation on the whole validation set...')
        num_batch = loader.size()//batch_size+1
        acc1_total = 0.
        acc5_total = 0.
        loader.reset()
        
        seg_labels_batch_empty = np.zeros([batch_size, seg_size, seg_size, num_seg_class])
        obj_class_batch_empty = np.zeros([batch_size, num_seg_class])

        for i in range(num_batch):
            if mode=='Seg Val':
                images_batch, seg_labels_batch, obj_class_batch, labels_batch = loader.next_batch(batch_size)   
                if debug:
                    acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={lrs:learning_rate_seg,lrc:learning_rate_class,x: images_batch, y: labels_batch, seg_labels: seg_labels_batch_empty, obj_class: obj_class_batch_empty, keep_dropout: 1., train_phase: False})
                    print('Validation Accuracy with empty Top1 = ' + '{:.4f}'.format(acc1) + ', Top5 = ' + '{:.4f}'.format(acc5))
            else:
                images_batch, labels_batch = loader.next_batch(batch_size)
                seg_labels_batch = seg_labels_batch_empty
                obj_class_batch = obj_class_batch_empty
                
            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={lrs:learning_rate_seg,lrc:learning_rate_class,x: images_batch, y: labels_batch, seg_labels: seg_labels_batch, obj_class: obj_class_batch, keep_dropout: 1., train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
            if debug:
                print('Validation Accuracy Top1 = ' + '{:.4f}'.format(acc1) + ', Top5 = ' + '{:.4f}'.format(acc5))
        
        acc1_total /= num_batch
        acc5_total /= num_batch
        t=int(time.time()-t)
        if debug:
            print('used'+str(t)+'s to validate')
        print('Evaluation Finished! Accuracy Top1 = ' + '{:.4f}'.format(acc1_total) + ', Top5 = ' + '{:.4f}'.format(acc5_total))
        return acc1_total,acc5_total
    
    def use_validation():
        if not validation:
            return 0,0
        acc1_total, acc5_total = use_evaluation(loader_val,'val')
        return acc1_total,acc5_total

    def use_test():
        if not test:
            return 0,0
        acc1_total, acc5_total = use_evaluation(loader_test,'test')
        return acc1_total,acc5_total

    step = 0
    train_class=0
    train_seg=0

    if train:
        train_accs=[]
        train_seg_accs=[]
        val_accs=[]
        test_accs=[]
        seg_losses=[]
        class_losses=[]
        lr_s=[]

        seg_labels_batch_1 = np.zeros([batch_size, seg_size, seg_size, num_seg_class])
        obj_class_batch_1 = np.zeros([batch_size, num_seg_class])
        while step < training_iters:

            if joint_ratio_decay:
                if step > 2000:
                    joint_ratio = joint_ratio * 0.9995
            if lr_decay:
                if step < 3000:
                    learning_rate_class = base_learning_rate_class
                    learning_rate_seg = base_learning_rate_seg
                else:
                    learning_rate_class = 0.9995 * learning_rate_class
                    learning_rate_seg = 0.9995 * learning_rate_seg
            # Load a batch of training data
            
            images_batch_2, seg_labels_batch_2, obj_class_batch_2, labels_batch_2 = loader_train_seg.next_batch(batch_size)

            images_batch_1, labels_batch_1 = loader_train.next_batch(batch_size)
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                print('Train Class',train_class,'Train Seg',train_seg)
                # Calculate batch loss and accuracy on training set
                l, lc, ls, acc1, acc5 = sess.run([loss,loss_class,loss_seg, accuracy1, accuracy5], feed_dict={lrs:learning_rate_seg,lrc:learning_rate_class,x: images_batch_1, y: labels_batch_1, seg_labels: seg_labels_batch_1, obj_class: obj_class_batch_1, keep_dropout: 1., train_phase: False}) 
                print('-Iter ' + str(step) + ', Training Loss= ' + '{:.6f}'.format(l) +', Class Loss= ' + '{:.6f}'.format(lc) + ', Seg Loss= ' + '{:.6f}'.format(ls) + ', Accuracy Top1 = ' + '{:.4f}'.format(acc1) + ', Top5 = ' + '{:.4f}'.format(acc5))
                train_accs.append(acc5)

                class_losses.append(lc)

                 # Calculate batch loss and accuracy on training set
                l, lc, ls, acc1, acc5 = sess.run([loss,loss_class,loss_seg, accuracy1, accuracy5], feed_dict={lrs:learning_rate_seg,lrc:learning_rate_class,x: images_batch_2, y: labels_batch_2, seg_labels: seg_labels_batch_2, obj_class: obj_class_batch_2, keep_dropout: 1., train_phase: False}) 
                print('-Iter ' + str(step) + ', Training with seg Loss= ' + '{:.6f}'.format(l) +', Class Loss= ' + '{:.6f}'.format(lc) + ', Seg Loss= ' + '{:.6f}'.format(ls) + ', Accuracy Top1 = ' + '{:.4f}'.format(acc1) + ', Top5 = ' + '{:.4f}'.format(acc5))
                train_seg_accs.append(acc5)
                
                seg_losses.append(ls)
                


                acc1, acc5=use_validation()
                val_accs.append(acc5)
                acc1, acc5=use_test()
                test_accs.append(acc5)

                lr_s.append(learning_rate_class)

                print val_accs
                print train_accs

                if plot:
                    a=np.arange(1,len(val_accs)+1,1)*step_display
                    
                    fig = plt.figure()
                    plt.plot(a,train_accs,'-',label='Training')
                    if validation:
                        # plt.plot(a,train_seg_accs,'-',label='Training with segm')
                        plt.plot(a,val_accs,'-',label='Validation')
                    if test:
                        plt.plot(a,test_accs,'-',label='Test')
                    plt.xlabel('Iteration')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    fig.savefig('./fig/pic_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)

                    fig = plt.figure()
                    plt.plot(a,seg_losses,'-',label='Seg')
                    plt.plot(a,class_losses,'-',label='Class')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    fig.savefig('./fig/pic_loss_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)
                    print 'finish saving figure to view'

                    fig = plt.figure()
                    plt.plot(a,lr_s,'-',label='Seg')
                    plt.xlabel('Iteration')
                    plt.ylabel('Learning Rate')
                    plt.legend()
                    fig.savefig('./fig/pic_lr_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)
                    print 'finish saving figure to view'
            
            # Run optimization op (backprop)

            flip = np.random.random_sample()
            if flip<joint_ratio:
                images_batch, seg_labels_batch, obj_class_batch, labels_batch = images_batch_2, seg_labels_batch_2, obj_class_batch_2, labels_batch_2
                sess.run(seg_optimizer, feed_dict={lrs:learning_rate_seg,lrc:learning_rate_class,x: images_batch, y: labels_batch, seg_labels: seg_labels_batch, obj_class: obj_class_batch, keep_dropout: dropout, train_phase: True})                
                train_seg+=1
            
            images_batch, seg_labels_batch, obj_class_batch, labels_batch = images_batch_1, seg_labels_batch_1, obj_class_batch_1, labels_batch_1 
            sess.run(class_optimizer, feed_dict={lrs:learning_rate_seg,lrc:learning_rate_class,x: images_batch, y: labels_batch, seg_labels: seg_labels_batch, obj_class: obj_class_batch, keep_dropout: dropout, train_phase: True})
            step += 1
            train_class+=1

            # Save model
            if step % step_save == 0 or step==1:
                saver.save(sess, path_save, global_step=step+pretrainedStep)
                print('Model saved at Iter %d !' %(step))
        print('Optimization Finished!')

    
    use_validation()
    use_test()
print('[%s]:' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

