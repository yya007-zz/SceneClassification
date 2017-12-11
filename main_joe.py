import os, datetime
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataLoader import *
from architect_joe import *
from exp_joe import *
import sys
from save import *

# Dataset Parameters
print 'Running command: ',sys.argv
Parameters=sys.argv[1]
experiment=exp_joe
if Parameters in experiment:
    settings = experiment[Parameters]
    print 'Parameters: ',experiment[Parameters]
else:
    raise ValueError(Parameters,' no dict of parameters found')

debug = False
# Training Parameters
learning_rate = settings['learning_rate']
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

loader_train_seg = DataLoaderDisk(**opt_data_train_seg)
loader_val = DataLoaderDisk(**opt_data_val)

print ('finish loading data')
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
seg_labels = tf.placeholder(tf.float32, [None, seg_size, seg_size, num_seg_class])


keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
if selectedmodel=='vgg_seg1':
    myModel = vgg_seg1(x, seg_labels, keep_dropout, train_phase)
else:
    raise ValueError(selectedmodel,' no such model, end of the program')

# Define loss and optimizer
logits_seg= myModel.logits_seg
loss = myModel.loss

seg_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


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

    def use_validation():
        return acc1_total,acc5_total

    step = 0

    if train:
        val_losses = []
        losses=[]

        while step < training_iters:
            # Load a batch of training data
            
            images_batch, seg_labels_batch, obj_class_batch, labels_batch = loader_train_seg.next_batch(batch_size)
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
                # Calculate batch loss and accuracy on training set
                l, = sess.run([loss], 
                        feed_dict={x: images_batch, seg_labels: seg_labels_batch, keep_dropout: 1., train_phase: False}) 
                print('-Iter ' + str(step) + ', Training Loss= ' + '{:.10f}'.format(l))
                losses.append(l)

                # Evaluate on the whole validation set
                print('Evaluation on the whole validation set...')
                num_batch = loader_val.size()//batch_size+1
                val_loss = 0.
                loader_val.reset()
                
                for i in range(num_batch):
                    images_batch, seg_labels_batch, obj_class_batch, labels_batch = loader_val.next_batch(batch_size)    
                        
                    l, = sess.run([loss], 
                            feed_dict={x: images_batch, seg_labels: seg_labels_batch, keep_dropout: 1., train_phase: False})
                    val_loss += l
                print('Evaluation Finished! Validation Loss = ' + '{:.10f}'.format(val_loss))
                val_losses.append(val_loss)


                if plot:
                    fig = plt.figure()
                    a=np.arange(1,len(losses)+1,1)
                    plt.plot(a,losses,'-',label='Class')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    fig.savefig('./fig/pic_train_loss_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)


                    fig = plt.figure()
                    a=np.arange(1,len(val_losses)+1,1)
                    plt.plot(a,val_losses,'-',label='Class')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    fig.savefig('./fig/pic_val_loss_'+str(exp_name)+'.png')   # save the figure to file
                    plt.close(fig)
                    print 'finish saving figure to view'

            
            # Run optimization op (backprop)

            sess.run(seg_optimizer, feed_dict={x: images_batch, seg_labels: seg_labels_batch, keep_dropout: dropout, train_phase: True})
            step += 1
            
            # Save model
            if step % step_save == 0 or step==1:
                saver.save(sess, path_save, global_step=step+pretrainedStep)
                print('Model saved at Iter %d !' %(step))
        print('Optimization Finished!')

