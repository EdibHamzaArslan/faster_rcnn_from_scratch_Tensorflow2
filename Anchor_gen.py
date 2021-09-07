import sys

import numpy as np
from utils import iou, box2loc
import Config as C


class AnchorGenerator:
    def __init__(self, gt_boxes):
        self.gt_boxes = gt_boxes

        self.feature_size = C.input_width // C.sub_sample  # 14

        self.all_anchors, self.valid_anchors, self.valid_anchor_indexes = self.get_anchors()

        self.ious = self.get_anchor_ious(self.valid_anchors, self.gt_boxes)
        self.max_ious, self.max_iou_boxes = self.get_max_ious(self.ious, self.gt_boxes, self.valid_anchor_indexes)
        self.gt_argmax_ious = self.get_gt_argmax_ious(self.ious)

        # FINAL ANCHORS
        self.anchor_labels = self.get_anchor_labels(self.valid_anchor_indexes,
                                                    self.max_ious,
                                                    self.gt_argmax_ious,
                                                    self.all_anchors)
        self.anchor_locations = self.get_final_anchors(self.all_anchors,
                                                       self.valid_anchors,
                                                       self.max_iou_boxes,
                                                       self.valid_anchor_indexes)



    def calculate_centers(self):
        ctr_x = np.arange(C.sub_sample,
                          (self.feature_size + 1) * C.sub_sample,
                          C.sub_sample)
        ctr_y = np.arange(C.sub_sample,
                          (self.feature_size + 1) * C.sub_sample,
                          C.sub_sample)
        index = 0
        ctr_s = np.zeros((self.feature_size * self.feature_size, 2))
        for x in range(len(ctr_x)):
            for y in range(len(ctr_y)):
                ctr_s[index, 1] = ctr_x[x] - 8
                ctr_s[index, 0] = ctr_y[y] - 8
                index += 1
        return ctr_s

    def get_anchors(self):
        anchors = np.zeros((self.feature_size * self.feature_size * C.n_anchors, 4))
        anchor_centers = self.calculate_centers()

        index = 0
        for ctr in anchor_centers:
            ctr_y, ctr_x = ctr
            for i in range(len(C.ratios)):
                for j in range(len(C.anchor_scales)):
                    h = C.sub_sample * C.anchor_scales[j] * np.sqrt(C.ratios[i])
                    w = C.sub_sample * C.anchor_scales[j] * np.sqrt(1. / C.ratios[i])

                    anchors[index, 0] = ctr_y - h / 2.
                    anchors[index, 1] = ctr_x - w / 2.
                    anchors[index, 2] = ctr_y + h / 2.
                    anchors[index, 3] = ctr_x + w / 2.
                    index += 1

        valid_anchor_indexes = np.where((anchors[:, 0] >= 0) &
                                        (anchors[:, 1] >= 0) &
                                        (anchors[:, 2] <= C.input_width) &
                                        (anchors[:, 3] <= C.input_height))[0]

        return anchors, anchors[valid_anchor_indexes], valid_anchor_indexes

    def get_anchor_labels(self,
                          valid_anchor_indexes,
                          max_ious,
                          gt_argmax_ious,
                          all_anchors):
        # assert self.valid_anchor_indexes == [], "Valid anchors are empty"
        anchor_labels = np.zeros((len(valid_anchor_indexes),), dtype=np.int32)
        anchor_labels.fill(-1)

        anchor_labels[max_ious < C.anchor_neg_iou_threshold] = 0
        anchor_labels[gt_argmax_ious] = 1
        anchor_labels[max_ious >= C.anchor_pos_iou_threshold] = 1

        # Cascading
        # Total positive samples
        n_pos = C.anchor_pos_ratio * C.anchor_n_sample  # 128
        pos_index = np.where(anchor_labels == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=int(len(pos_index) - n_pos), replace=False)
            anchor_labels[disable_index] = -1

        # Total negative samples
        n_neg = C.anchor_n_sample - np.sum(anchor_labels == 1)  # Note: you can remove np.sum with n_pos, try later!!
        neg_index = np.where(anchor_labels == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=int(len(neg_index) - n_neg), replace=False)
            anchor_labels[disable_index] = -1

        final_anchor_labels = np.empty((len(all_anchors)), dtype=anchor_labels.dtype)
        final_anchor_labels.fill(-1)
        final_anchor_labels[valid_anchor_indexes] = anchor_labels
        final_anchor_labels = final_anchor_labels[..., np.newaxis]

        return final_anchor_labels

    def get_anchor_ious(self, valid_anchors, gt_boxes):
        ious = np.empty((len(valid_anchors), len(gt_boxes)), dtype=np.float32)
        ious.fill(0)
        for anchor_index, anchor in enumerate(valid_anchors):
            for gt_box_index, gt_box in enumerate(gt_boxes):
                ious[anchor_index, gt_box_index] = iou(anchor, gt_box)
        return ious

    def get_max_ious(self, ious, gt_boxes, valid_anchor_indexes):
        argmax_ious = ious.argmax(axis=1)
        max_iou_boxes = gt_boxes[argmax_ious]
        max_ious = ious[np.arange(len(valid_anchor_indexes)), argmax_ious]
        return max_ious, max_iou_boxes

    def get_gt_argmax_ious(self, ious):
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        return gt_argmax_ious

    def get_final_anchors(self, all_anchors, valid_anchors, max_iou_bbox, valid_anchor_indexes):
        anchor_locs = box2loc(valid_anchors, max_iou_bbox)
        anchor_locations = np.empty(all_anchors.shape, dtype=anchor_locs.dtype)
        anchor_locations.fill(0)
        anchor_locations[valid_anchor_indexes, :] = anchor_locs
        return anchor_locations



# if __name__ == '__main__':
#     gt_boxes = np.array([[20, 30, 400, 500], [300, 400, 500, 600]])
#     anchor_gen = AnchorGenerator(gt_boxes)
#     print(anchor_gen.anchor_labels.shape)
#     # print(anchor_gen.anchor_locations.shape) # 22500, 4


