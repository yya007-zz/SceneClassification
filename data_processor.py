import cv2
import numpy as np
import xml.etree.ElementTree as ET
import itertools
import os.path as osp
import os

CLASS_NUMBER = 175 # without background

def xml_extract(xml_path, seg_label_path):
    """
    extract information in xml file
    Input:
        xml_path: string; the path of the xml file
        seg_label_path: string; the path of the npy file to be stored
    Return:
        image_label: int; the class of this image
    Note:
        in npy file, it is stored as np3darray (w, h, class_number + 1) in np.bool type
        the last layer of class is background
    """
    with open(xml_path) as f:
        it = itertools.chain('<root>', f, '</root>')
        root = ET.fromstringlist(it)
    seg_label = np.zeros([CLASS_NUMBER, 128, 128], dtype=np.uint8)

    for obj in root:
        if obj.tag == 'class':
            image_label = int(obj.text)
        if obj.tag == 'objects':
            polygon_list = []
            label = obj.find('class')
            label = int(label.text)
            polygon = obj.find('polygon')
            for point in polygon:
                x = point.find('x')
                x = int(x.text) - 1
                y = point.find('y')
                y = int(y.text) - 1
                pt = np.array([[[x,y]]], dtype=np.int32)
                polygon_list.append(pt)
            polygon = np.concatenate(polygon_list, axis=1)
            cv2.fillPoly(seg_label[label], polygon, 255)
    seg_label = seg_label.astype(bool)
    background = np.ones([128,128], dtype=bool) - np.sum(seg_label, axis=0).astype(bool)
    seg_label = np.concatenate([seg_label, np.expand_dims(background, axis=0)], axis=0)
    seg_label = np.packbits(seg_label, axis=-1)

    """
    for i in range(seg_label.shape[0]):
        if np.sum(seg_label[i]) != 0:
            cv2.imshow('image', seg_label[i].astype(np.uint8) * 255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    """

    np.save(seg_label_path, seg_label)
    return image_label

def run(tv_subdir):
    cwd = os.getcwd()
    if not osp.exists(osp.join(cwd, 'data')):
        raise ValueError('Not in the root directory running this file')

    if tv_subdir == 'train':
        save_path = osp.join(cwd, 'data', 'new_train.txt')
    elif tv_subdir == 'val':
        save_path = osp.join(cwd, 'data', 'new_val.txt')
    else:
        raise ValueError('No {}, should type train or val'.format(tv_subdir))
    

    txt_list = []

    data_dir = osp.join(cwd, 'data')
    seg_labels_subdir = 'seg_labels'
    images_subdir = 'images'
    objects_subdir = 'objects'
    images_dir = osp.join(data_dir, images_subdir, tv_subdir)
    seg_labels_dir = osp.join(data_dir, seg_labels_subdir, tv_subdir)
    objects_dir = osp.join(data_dir, objects_subdir, tv_subdir)

    path_queue = [] # 
    dir_queue = [objects_dir]
    while len(dir_queue) != 0:
        _dir = dir_queue.pop(0)
        for _subdir in os.listdir(_dir):
            _new_dir = osp.join(_dir, _subdir)
            if osp.isdir(_new_dir):
                dir_queue.append(_new_dir)
            else:
                if osp.splitext(_new_dir)[1] == ".xml":
                    path_queue.append(_new_dir)
    count = 0
    for objects_path in path_queue:
        objects_final_dir, filename = osp.split(objects_path)
        filename_prefix = osp.splitext(filename)[0]
        images_final_dir = objects_final_dir.replace("objects", "images")
        seg_labels_final_dir = objects_final_dir.replace("objects", "seg_labels")
        images_path = osp.join(images_final_dir, filename_prefix + '.jpg')
        seg_labels_path = osp.join(seg_labels_final_dir, filename_prefix + '.npy')
        if not osp.exists(objects_path):
            raise ValueError("no {} directory".format(objects_path))
        if not osp.exists(images_path):
            raise ValueError("no {} directory".format(images_path))
        if not osp.exists(seg_labels_final_dir):
            os.makedirs(seg_labels_final_dir)
        images_label = str(xml_extract(objects_path, seg_labels_path))
        images_relpath = osp.relpath(images_path, osp.join(data_dir, images_subdir))
        seg_labels_relpath = osp.relpath(seg_labels_path, osp.join(data_dir, seg_labels_subdir))
        txt_list.append('{} {} {}\n'.format(images_relpath, seg_labels_relpath, images_label))
        print "processing image {}".format(count)
        count += 1
    with open(save_path, 'w') as f:
        for line in txt_list:
            f.write(line)
    print "successfully saved the txt file in {} !".format(save_path)

if __name__ == '__main__':
    #tv_subdir = 'val'
    # run('train')
    run('val')