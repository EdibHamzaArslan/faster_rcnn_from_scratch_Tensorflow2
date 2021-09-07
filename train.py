import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import backbone
from RPN import rpn
from ROI import head

from Anchor_gen import AnchorGenerator
import Proposal_gen as proposal_gen
from DataGenerator import VOC2012


# 14 14 512 feature map

inputs = tf.keras.Input((224, 224, 3))
vgg16_model = backbone.get_backbone(inputs)
vgg16_model.trainable = False
# vgg16_model.summary() # 14 14 512

rpn_model = rpn(vgg16_model.outputs[0].shape[1:], 9)
# rpn_model.summary()

batch_size = 2
epochs = 2

input_imgs_path = '../data/imgs' # sys.argv[1]
annotations_path = '../data/annotations' # sys.argv[2]
data_gen = VOC2012(input_data_path=input_imgs_path,
                   annotations_data_path=annotations_path,
                   batch_size=batch_size)


for epoch in range(epochs):
    for imgs, boxes, cls in data_gen:
        anchor_label_list = []
        anchor_location_list = []
        all_anchors_list = []
        for batch_index in range(batch_size):
            anchor_gen = AnchorGenerator(boxes[batch_index])
            all_anchors_list.append(anchor_gen.all_anchors[np.newaxis, :])
            anchor_label_list.append(anchor_gen.anchor_labels[np.newaxis, :])
            anchor_location_list.append(anchor_gen.anchor_locations[np.newaxis, :])

        all_anchors = np.concatenate(all_anchors_list, axis=0)
        anchor_labels = np.concatenate(anchor_label_list, axis=0)
        anchor_locations = np.concatenate(anchor_location_list, axis=0)

        # print(f"valid anchors shape {all_anchors.shape}") # 2 1764 4
        # print(anchor_locations.shape) # 2 1764 4
        # print(anchor_labels.shape) # 2 1764 1
        with tf.GradientTape() as rpn_tape:
            feature_maps = vgg16_model(imgs, training=False)
            pred_anchor_locs, pred_cls_scores = rpn_model(feature_maps)
            rpn_loss_cls = tf.keras.losses.binary_crossentropy(anchor_labels, pred_cls_scores)
            rpn_loss_box = tf.keras.losses.mse(anchor_locations, pred_anchor_locs)
            rpn_loss = tf.reduce_sum(rpn_loss_box, axis=-1) + tf.reduce_sum(rpn_loss_cls, axis=-1)
        print(f"[*]rpn loss ==> {rpn_loss}")
        rpn_grads = rpn_tape.gradient(rpn_loss, rpn_model.trainable_weights)
        tf.keras.optimizers.Adam(learning_rate=0.01).apply_gradients(zip(rpn_grads, rpn_model.trainable_weights))

        roi_list = proposal_gen.get_roi(all_anchors,
                                        pred_anchor_locs,
                                        pred_cls_scores)

        # colors = np.array([[0.0, 0.0, 1.0]])
        # te = imgs[0][tf.newaxis, ...]
        # roi_list[0] = roi_list[0][np.newaxis, ...]
        # print(te.shape)
        # print(roi_list[0].shape)
        # te = tf.image.draw_bounding_boxes(te, roi_list[0] / 224, colors)
        # plt.imshow(te[0])
        # plt.show()
        # sys.exit(1)

        for batch_index in range(len(roi_list)):
            roi = roi_list[batch_index]
            total_gt_box_n = boxes[batch_index].shape[0]
            gt_roi_locs, gt_roi_labels, indices_and_rois = proposal_gen.get_roi_target(roi,
                                                                                       boxes[batch_index],
                                                                                       cls[batch_index],
                                                                                       total_gt_box_n)

            # print(gt_roi_locs.shape, gt_roi_labels.shape, indices_and_rois.shape) # 128, 4 & 128, 21 & 128, 5

            roi_model, pool_output = head(indices_and_rois, feature_maps[batch_index], indices_and_rois.shape[0])

            with tf.GradientTape() as roi_tape:
                pred_roi_box, pred_roi_cls = roi_model(pool_output)
                # print(f"pred roi box {pred_roi_box.shape}, pred_roi_cls {pred_roi_cls.shape}")  # 128 84 & 128 21
                te_roi_loc = np.array(pred_roi_box)
                n_sample = te_roi_loc.shape[0]
                roi_loc = te_roi_loc.reshape((n_sample, -1, 4)) # 128, 21, 4
                roi_loc = roi_loc[np.arange(0, n_sample), gt_roi_labels.argmax(axis=-1)]
                pred_roi_box = tf.convert_to_tensor(roi_loc)
                loss_roi_box = tf.keras.losses.mse(gt_roi_locs, pred_roi_box)
                loss_roi_cls = tf.keras.losses.categorical_crossentropy(gt_roi_labels, pred_roi_cls)
                roi_loss = tf.reduce_sum(loss_roi_box) + tf.reduce_sum(loss_roi_cls)
            print(f"[*]roi loss ==> {roi_loss}")
            roi_grads = roi_tape.gradient(roi_loss, roi_model.trainable_weights)
            tf.keras.optimizers.Adam(learning_rate=0.01).apply_gradients(zip(roi_grads, roi_model.trainable_weights))