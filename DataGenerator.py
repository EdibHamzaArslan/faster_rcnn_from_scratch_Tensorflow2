import os
import numpy as np
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class VOC2012(tf.keras.utils.Sequence):
    ''' Generates data for keras
    Sources:
    https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''

    def __init__(self, input_data_path, annotations_data_path, training=True, num_class=21, batch_size=2, dim=(224, 224), shuffle=True):

        self.input_data_path = input_data_path
        self.annotations_data_path = annotations_data_path

        self.n_classes = num_class
        self.training = training

        self.input_files = sorted([file for file in glob.glob(os.path.join(self.input_data_path, '*.jpg'))])
        self.annotations = sorted([file for file in glob.glob(os.path.join(self.annotations_data_path, '*.xml'))])

        self.batch_size = batch_size
        self.dim = dim

        self.shuffle = shuffle

        self.n = 0
        self.max = self.__len__()
        self.on_epoch_end()

        self.category_map = {'background': 0,
                             'aeroplane': 1,
                             'bicycle': 2,
                             'bird': 3,
                             'boat': 4,
                             'bottle': 5,
                             'bus': 6,
                             'car': 7,
                             'cat': 8,
                             'chair': 9,
                             'cow': 10,
                             'diningtable': 11,
                             'dog': 12,
                             'horse': 13,
                             'motorbike': 14,
                             'person': 15,
                             'pottedplant': 16,
                             'sheep': 17,
                             'sofa': 18,
                             'train': 19,
                             'tvmonitor': 20}


    def __len__(self):
        if self.batch_size > len(self.input_files):
            print("Batch size is greater than data size!!")
            return -1
        return int(np.ceil(len(self.input_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        input_images = [self.input_files[k] for k in indexes]
        annotations = [self.annotations[k] for k in indexes]

        input_image, bbox, cls = self.__data_generation(input_images, annotations)

        return input_image, bbox, cls

    def normalize(self, input_image):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image

    def read_annotations(self, annotations):
        batch_annotation_boxes = []
        batch_annotation_names = []
        for index, annotation_file in enumerate(annotations):
            root = ET.parse(annotation_file)
            n_object = len(root.findall('object'))

            for child in zip(root.findall('size/width'), root.findall('size/height')):
                width, height = int(child[0].text), int(child[1].text)

            x_scale = self.dim[0] / width
            y_scale = self.dim[1] / height

            bboxes = np.empty((n_object, 4), dtype=np.int32)
            for i, child in enumerate(root.findall('object/bndbox')):
                for j, c in enumerate(child):
                    # TODO the coordinates should order of y1, x1, y2, x2
                    # x1, y1, x2, y2 => xmin, ymin, xmax, ymax
                    if j % 2 == 0: # x1, x2
                        bboxes[i, j+1] = int(np.round(int(c.text) * x_scale))
                    else: # y1, y2
                        bboxes[i, j-1] = int(np.round(int(c.text) * y_scale))

            name_index = []
            for i, child in enumerate(root.findall('object/name')):
                name_index.append(self.category_map[child.text])

            batch_annotation_boxes.append(bboxes)
            one_hot_names = tf.one_hot(name_index, self.n_classes, dtype=tf.int8)
            batch_annotation_names.append(one_hot_names)
        return np.array(batch_annotation_boxes), np.array(batch_annotation_names)

    def __data_generation(self, input_images, annotations):
        batched_input_images = np.empty((len(input_images), *self.dim, 3), dtype=np.float32)
        batched_boxes, batched_cls = self.read_annotations(annotations)

        for index, input_path in enumerate(input_images):

            input_img = tf.io.read_file(input_path)
            input_img = tf.image.decode_jpeg(input_img)
            input_img = tf.image.resize(input_img,
                                        [self.dim[0], self.dim[1]],
                                        tf.image.ResizeMethod.BICUBIC)
            input_img = self.normalize(input_img)
            batched_input_images[index] = input_img

        return batched_input_images, batched_boxes, batched_cls

    def one_hot2label(self, one_hot):
        max_value = np.argmax(one_hot, axis=0)
        for name, category in self.category_map.items():
            if category == max_value:
                return name

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.input_files))
        if self.shuffle == True:
            # np.random.seed(2)
            np.random.shuffle(self.indexes)

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

if __name__ == '__main__':
    input_imgs_path = '../data/imgs'
    annotations_path = '../data/annotations'
    data_gen = VOC2012(input_data_path=input_imgs_path, annotations_data_path=annotations_path)

    count = 0
    for imgs, boxes, cls in data_gen:
        img = imgs[0]
        # print(boxes.shape)
        # print(boxes[0].shape)
        # print(boxes[1].shape)
        # break
        for box, label in zip(boxes[0], cls[0]):
            # TODO Order are changed, the order is ymin, xmin, ymax, xmax
            ymin, xmin, ymax, xmax = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, data_gen.one_hot2label(label), (xmin + 30, ymin + 10), 0, 0.3, (0, 255, 0))
        plt.imshow(img)
        plt.show()
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        if count == 2:
            break
        count += 1

