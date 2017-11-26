import os
import os.path as osp
import numpy as np
import cv2
#np.random.seed(123)

# Loading data from disk
# NOTE:seg_label will show -1 if the class is background!

CLASS_NUMBER_WITH_BACKGROUND = 176 # with background

class DataLoaderDisk(object):
    """
    Loading data from disk. Users are able to use kwargs to define some changeable variables:

    images_root:        string; root of images, ex: "./data/images/"
    seg_labels_root:    string; root of seg labels, ex: "./data/seg_labels/"
    data_list:          string; the list of data, ex:"./data/new_train.txt"
    load_size:          int; for data augmentation, image will first reshape to this size
    fine_size:          int; for data augmentation, after reshape, image will be sliced to this size
    seg_size:           int; for data augmentation, the segmentation label will be reshape to this size
    randomize:          bool; for data augmentation, after reshape, image will slice randomly if this is true
    data_mean:          np1darray (3,); the mean of the data in percentage
    perm:               bool; the data will be randomly permuted to create batches
    test:               bool; test or not, if test, then will not return segmentation related values

    The only supported function of this class is next_batch, please read following for detailed description
    """
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.seg_size = int(kwargs['seg_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.perm = kwargs['perm']
        self.images_root = kwargs['images_root']
        self.seg_labels_root = kwargs['seg_labels_root']
        self.data_list = kwargs['data_list']
        self.test = kwargs['test']

        if not osp.exists(self.images_root):
            raise ValueError("directory not exists: {}".format(osp.join(os.getcwd(), self.images_root)))
        if not osp.exists(self.seg_labels_root):
            raise ValueError("directory not exists: {}".format(osp.join(os.getcwd(), self.seg_labels_root)))
        if not osp.exists(self.data_list):
            raise ValueError("file not exists: {}".format(osp.join(os.getcwd(), self.data_list)))

        # read data info from lists
        if self.test:
            self.list_image_path = []
            self.list_label = []
            with open(self.data_list, 'r') as f:
                for line in f:
                    image_subpath, label =line.rstrip().split(' ')
                    image_path = osp.join(self.images_root, image_subpath)
                    if not osp.exists(image_path):
                        raise ValueError("file not exists: {}".format(image_path))
                    self.list_image_path.append(image_path)
                    self.list_label.append(int(label))
            self.num = len(self.list_image_path)
            print('# Images found:', self.num)
        else:
            self.list_image_path = []
            self.list_seg_label_path = []
            self.list_label = []
            with open(self.data_list, 'r') as f:
                for line in f:
                    image_subpath, seg_label_subpath, label =line.rstrip().split(' ')
                    image_path = osp.join(self.images_root, image_subpath)
                    seg_label_path = osp.join(self.seg_labels_root, seg_label_subpath)
                    if not osp.exists(image_path):
                        raise ValueError("file not exists: {}".format(image_path))
                    if not osp.exists(seg_label_path):
                        raise ValueError("file not exists: {}".format(seg_label_path))
                    self.list_image_path.append(image_path)
                    self.list_seg_label_path.append(seg_label_path)
                    self.list_label.append(int(label))
            self.num = len(self.list_image_path)
            print('# Images found:', self.num)

        # create self.order which is the order to generate batch
        self.permutation()

        self._idx = 0
        
    def next_batch(self, batch_size):
        """
        Create the next batch
        Input:
            batch_size: int; the batch size to generate
        Return:
            if not test:
            images_batch: np4darray (batch_size, fine_size, fine_size, 3); the image batch
            seg_labels_batch: np4darray (batch_size, seg_size, seg_size, CLASS_NUMBER_WITH_BACKGROUND); the seg label batch
            obj_class_batch: np2darray (batch_size, CLASS_NUMBER_WITH_BACKGROUND); the object class label batch
            labels_batch: np1darray (batch_size,); the label batch
            if test:
            images_batch: np4darray (batch_size, fine_size, fine_size, 3); the image batch
            labels_batch: np1darray (batch_size,); the label batch
        """
        if self.test:
            # if it is test mode
            images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
            labels_batch = np.zeros(batch_size, dtype=np.int16)
            for i in range(batch_size):
                image = cv2.imread(self.list_image_path[self.order[self._idx]])
                print "image path: {}".format(self.list_image_path[self.order[self._idx]])
                image = cv2.resize(image, (self.load_size, self.load_size))
                image = image.astype(np.float32)/255.
                image = image - self.data_mean

                if self.randomize:
                    flip = np.random.random_integers(0, 1)
                    if flip>0:
                        image = image[:,::-1,:]
                    offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                    offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
                else:
                    offset_h = (self.load_size-self.fine_size)//2
                    offset_w = (self.load_size-self.fine_size)//2

                images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
                labels_batch[i, ...] = self.list_label[self.order[self._idx]]
                
                self._idx += 1
                if self._idx == self.num:
                    self._idx = 0
                    self.permutation()
            return images_batch, labels_batch
        else:
            images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
            seg_labels_batch = np.zeros((batch_size, self.seg_size, self.seg_size, CLASS_NUMBER_WITH_BACKGROUND))
            obj_class_batch = np.zeros((batch_size, CLASS_NUMBER_WITH_BACKGROUND))
            labels_batch = np.zeros(batch_size, dtype=np.int16)
            for i in range(batch_size):
                image = cv2.imread(self.list_image_path[self.order[self._idx]])
                #print "image path: {}".format(self.list_image_path[self.order[self._idx]])
                image = cv2.resize(image, (self.load_size, self.load_size))
                image = image.astype(np.float32)/255.
                image = image - self.data_mean

                seg_label_org = np.load(self.list_seg_label_path[self.order[self._idx]])
                seg_label_org = np.unpackbits(seg_label_org, axis = -1)
                seg_label_org = seg_label_org.astype(np.float32)
                seg_label_list = []
                for j in range(CLASS_NUMBER_WITH_BACKGROUND):
                    if np.sum(seg_label_org[j]) != 0:
                        obj_class_batch[i][j] = 1.
                        tmp = cv2.resize(seg_label_org[j], (self.load_size, self.load_size),interpolation = cv2.INTER_NEAREST)
                        seg_label_list.append(tmp)
                    else:
                        tmp = np.zeros((self.load_size, self.load_size), dtype=np.uint8)
                        seg_label_list.append(tmp)

                if self.randomize:
                    flip = np.random.random_integers(0, 1)
                    if flip>0:
                        image = image[:,::-1,:]
                        for j in range(len(seg_label_list)):
                            if obj_class_batch[i][j] == 1.:
                                seg_label_list[j] = seg_label_list[j][:,::-1]
                    offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                    offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
                else:
                    offset_h = (self.load_size-self.fine_size)//2
                    offset_w = (self.load_size-self.fine_size)//2

                images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
                tmp_list = []
                for j in range(len(seg_label_list)):
                    if obj_class_batch[i][j] == 1.:
                        tmp = seg_label_list[j][offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size]
                        tmp = cv2.resize(tmp, (self.seg_size, self.seg_size),interpolation = cv2.INTER_NEAREST)
                        tmp_list.append(np.expand_dims(tmp,2))
                    else:
                        tmp = np.zeros((self.seg_size, self.seg_size), dtype=np.uint8)
                        tmp_list.append(np.expand_dims(tmp,2))
                seg_labels_batch[i, ...] = np.concatenate(tmp_list, 2)
                labels_batch[i, ...] = self.list_label[self.order[self._idx]]
                
                self._idx += 1
                if self._idx == self.num:
                    self._idx = 0
                    self.permutation()
            
            return images_batch, seg_labels_batch, obj_class_batch, labels_batch

    def permutation(self):
        # permutation
        if self.perm:
            self.order = np.random.permutation(self.num) 
        else:
            self.order = np.arange(self.num)
        return
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

if __name__ == '__main__':
    opt_data_train = {
        'images_root': './data/images/',   # MODIFY PATH ACCORDINGLY
        'seg_labels_root': './data/seg_labels/',   # MODIFY PATH ACCORDINGLY
        'data_list': './data/new_val.txt', # MODIFY PATH ACCORDINGLY
        'load_size': 256,
        'fine_size': 214,
        'seg_size': 7,
        'data_mean': np.array([0.45834960097,0.44674252445,0.41352266842], dtype=np.float32),
        'randomize': True,
        'perm' : True,
        'test' : False
        }
    loader = DataLoaderDisk(**opt_data_train)
    data_mean = np.array([0.45834960097,0.44674252445,0.41352266842], dtype=np.float32)
    for i in range(5):
        batch_size = 53
        images_batch, seg_labels_batch, obj_class_batch, labels_batch = loader.next_batch(batch_size)
        image = cv2.convertScaleAbs((images_batch[0] + data_mean) * 255.)
        bar = np.array([[[255,0,0]]], dtype=np.uint8)
        bar = np.tile(bar, [214,10,1])
        print np.where(obj_class_batch[0] == 1.)

        for j in range(CLASS_NUMBER_WITH_BACKGROUND):
            label = seg_labels_batch[0,:,:,j]
            if np.sum(label) != 0:
                label = label.astype(np.float32)*255
                label = cv2.convertScaleAbs(label)
                label = cv2.resize(label, (214,214), interpolation = cv2.INTER_NEAREST)
                label = np.tile(np.expand_dims(label,2), (1,1,3))
                image = np.concatenate([image, bar, label], axis=1)
                print j
        print "image class: {}".format(labels_batch[0])
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print "\n"


    opt_data_train = {
        'images_root': './data/images/',   # MODIFY PATH ACCORDINGLY
        'seg_labels_root': './data/seg_labels/',   # MODIFY PATH ACCORDINGLY
        'data_list': './data/test.txt', # MODIFY PATH ACCORDINGLY
        'load_size': 256,
        'fine_size': 214,
        'seg_size': 7,
        'data_mean': np.array([0.45834960097,0.44674252445,0.41352266842], dtype=np.float32),
        'randomize': False,
        'perm' : False,
        'test' : True
        }
    loader = DataLoaderDisk(**opt_data_train)
    data_mean = np.array([0.45834960097,0.44674252445,0.41352266842], dtype=np.float32)
    for i in range(5):
        batch_size = 53
        images_batch, labels_batch = loader.next_batch(batch_size)
        image = cv2.convertScaleAbs((images_batch[0] + data_mean) * 255.)
        print "image class: {}".format(labels_batch[0])
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print "\n"
