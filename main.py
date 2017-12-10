import os, datetime
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataLoader import *
from DataLoaderOld import *
from architect import *
from architect2 import *
from exp import *
from exp2 import *
import sys
from save import *
# Dataset Parameters
print 'Running command: ',sys.argv
ParametersDict=sys.argv[1]
Parameters=sys.argv[2]
if ParametersDict == 'exp2':
    experiment=exp2
else:
    experiment=exp
if Parameters in experiment:
    settings = experiment[Parameters]
    print 'Parameters: ',experiment[Parameters]
else:
    raise ValueError('no dict of parameters found')

# Training Parameters
learning_rate = settings['learning_rate']
set_lam = settings['lam']
training_iters = settings['training_iters']
step_display = settings['step_display']
step_save = settings['step_save']
exp_name = settings['exp_name']
pretrainedStep = settings['pretrainedStep']
selectedmodel= settings['selectedmodel']
plot=settings['plot']

train = settings['train']
validation = settings['validation']
test = settings['test']
batch_size = settings['batch_size']

path_save = './save/'+exp_name+'/'
start_from=''

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

opt_data_val = {
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

loader_train_seg = DataLoaderDisk(**opt_data_train_seg)
loader_train = DataLoaderDiskOld(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDiskOld(**opt_data_test)

print ('finish loading data')
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
seg_labels = tf.placeholder(tf.float32, [None, seg_size, seg_size, 176])
obj_class = tf.placeholder(tf.float32, [None, 176])
y = tf.placeholder(tf.int64, None)


keep_dropout = tf.placeholder(tf.float32)
lam = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
if selectedmodel=='VGG':
    myModel = vgg_model(x, y, seg_labels, obj_class, lam, keep_dropout, train_phase)
elif selectedmodel=='VGG_BN':
    myModel = vgg_bn_model(x, y, seg_labels, obj_class, lam, keep_dropout, train_phase)
elif selectedmodel=='alexnet':
    myModel = alexnet_model(x, y, seg_labels, obj_class, lam, keep_dropout, train_phase)
elif selectedmodel=='VGG_simple':
    myModel = vgg_simple_model(x, y, seg_labels, obj_class, lam, keep_dropout, train_phase)
elif selectedmodel=='vgg_seg2':
    myModel = vgg_seg2_model(x, y, seg_labels, obj_class, lam, keep_dropout, train_phase)   
elif selectedmodel=='vgg_bn_seg2':
    myModel = vgg_seg2_model(x, y, seg_labels, obj_class, lam, keep_dropout, train_phase)   
else:
    raise ValueError('no such model, end of the program')

# Define loss and optimizer
logits= myModel.logits_class
loss = myModel.loss
loss_seg = myModel.loss_seg
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

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
with tf.Session() as sess:
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
        loader_val.reset()
        for i in range(num_batch):
            if mode=='val':
                images_batch, seg_labels_batch, obj_class_batch, labels_batch = loader.next_batch(batch_size)    
            elif mode == 'test':
                images_batch, labels_batch = loader.next_batch(batch_size)
                seg_labels_batch = np.zeros([batch_size, seg_size, seg_size, c])
                obj_class_batch = np.zeros([batch_size, 176])

            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, seg_labels: seg_labels_batch, obj_class: obj_class_batch, lam:set_lam, keep_dropout: 1., train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
            print('Validation Accuracy Top1 = ' + '{:.4f}'.format(acc1) + ', Top5 = ' + '{:.4f}'.format(acc5))
        acc1_total /= num_batch
        acc5_total /= num_batch
        t=int(time.time()-t)
        print('used'+str(t)+'s to validate')
        print('Evaluation Finished! Accuracy Top1 = ' + '{:.4f}'.format(acc1_total) + ', Top5 = ' + '{:.4f}'.format(acc5_total))
        return acc1_total,acc5_total
    
    def use_validation():
        if not validation:
            return 0,0
        acc1_total, acc5_total = use_evaluation(loader_val,'val')

    def use_test():
        if not test:
            return 0,0
        acc1_total, acc5_total = use_evaluation(loader_test,'test')
        
    
    step = 0

    if train:
        train_accs=[]
        val_accs=[]
        while step < training_iters:
            # Load a batch of training data
            seg_labels_batch = np.zeros([batch_size, seg_size, seg_size, c])
            obj_class_batch = np.zeros([batch_size, 176])
            flip = np.random.random_integers(0, 1)
            mylam=set_lam
            if flip>0.5:
                images_batch, seg_labels_batch, obj_class_batch, labels_batch = loader_train_seg.next_batch(batch_size)
            else:
                mylam=0;
                images_batch, labels_batch = loader_train.next_batch(batch_size)
            
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                # Calculate batch loss and accuracy on training set
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, seg_labels: seg_labels_batch, obj_class: obj_class_batch, lam:mylam, keep_dropout: 1., train_phase: False}) 
                print('-Iter ' + str(step) + ', Training Loss= ' + \
                      '{:.6f}'.format(l) + ', Accuracy Top1 = ' + \
                      '{:.4f}'.format(acc1) + ', Top5 = ' + \
                      '{:.4f}'.format(acc5))
                train_accs.append(acc5)

                # # Calculate batch loss and accuracy on validation set
                # images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
                # l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
                # print('-Iter ' + str(step) + ', Validation Loss= ' + \
                #       '{:.6f}'.format(l) + ', Accuracy Top1 = ' + \
                #       '{:.4f}'.format(acc1) + ', Top5 = ' + \
                #       '{:.4f}'.format(acc5))
                acc1, acc5=use_validation()
                val_accs.append(acc5)
                print val_accs
                print train_accs

            if plot:
                fig = plt.figure()
                a=np.arange(1,len(val_accs)+1,1)
                plt.plot(a,train_accs,'-',label='Training')
                plt.plot(a,val_accs,'-',label='Validation')
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
                plt.legend()
                fig.savefig('./fig/pic_'+str(exp_name)+'.png')   # save the figure to file
                plt.close(fig)
                print 'finish saving figure to view'
            
            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, seg_labels: seg_labels_batch, obj_class: obj_class_batch,lam:mylam, keep_dropout: dropout, train_phase: True})
            
            step += 1
            
            # Save model
            if step % step_save == 0 or step==1:
                saver.save(sess, path_save, global_step=step+pretrainedStep)
                print('Model saved at Iter %d !' %(step))
        print('Optimization Finished!')

    
    use_validation()
    use_test()

