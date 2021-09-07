from utils import loc2box, nms, box2loc, iou
import numpy as np
import Config as C
import sys

def get_roi(all_anchors, pred_anchor_locs, pred_cls_scores):

    roi = loc2box(all_anchors, pred_anchor_locs)

    # print(roi.shape) # 2, 1764, 4
    roi[:, :, slice(0, 4, 2)] = np.clip(roi[:, :, slice(0, 4, 2)], 0, C.input_width)
    roi[:, :, slice(1, 4, 2)] = np.clip(roi[:, :, slice(1, 4, 2)], 0, C.input_height)

    hs = roi[:, :, 2] - roi[:, :, 0]
    ws = roi[:, :, 3] - roi[:, :, 1]
    roi_list = []

    for index, (h, w) in enumerate(zip(hs, ws)):
        keep = np.where((h >= C.min_size) & (w >= C.min_size))[0]
        score = np.squeeze(pred_cls_scores[index])[keep]
        order = score.argsort()[::-1]
        order = order[:C.n_train_pre_nms]
        kept_roi = roi[index][keep, :]

        kept_roi = kept_roi[order, :]
        score = score[order]

        nms_keep = nms(kept_roi, score, C.nms_threshold)
        if len(nms_keep) < C.n_train_post_nms:
            random_nms = np.random.choice(nms_keep, size=int(C.n_train_post_nms - len(nms_keep)))
            for item in random_nms:
                nms_keep.append(item)
        nms_keep = nms_keep[:C.n_train_post_nms] # change the post nms value like 200
        kept_roi = kept_roi[nms_keep, :]
        # print(f"kept roi shape {kept_roi.shape}") # 2000, 4 for every region of interest
        roi_list.append(kept_roi)
    return roi_list

def get_roi_cascade_index(max_iou):
    # Positive index
    pos_index = np.where(max_iou >= C.pt_pos_iou_threshold)[0]
    pos_roi_per_image = np.round(C.pt_n_sample * C.pt_pos_ratio)
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

    # Negative index
    neg_index = np.where((max_iou < C.pt_neg_iou_threshold_max) &
                         (max_iou >= C.pt_neg_iou_threshold_min))[0]
    neg_roi_per_this_image = C.pt_n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

    # Add pos_index and neg_index
    keep_index = np.append(pos_index, neg_index)
    return keep_index, pos_roi_per_this_image

def get_roi_target(roi, gt_boxes, class_labels, total_gt_box_n):
    ious = np.empty((len(roi), total_gt_box_n), dtype=np.float32)
    ious.fill(0)
    for roi_index, roi_item in enumerate(roi):
        for gt_box_index, gt_box in enumerate(gt_boxes):
            ious[roi_index, gt_box_index] = iou(roi_item, gt_box)

    # ious shape 46, 3  The three is equal the number of ground truth object
    gt_assignment = ious.argmax(axis=-1)  # 46, max iou index
    max_iou = ious.max(axis=-1)  # 46, max iou value

    gt_roi_label = np.array(class_labels)[gt_assignment, :]  # Due to tensorflow one hot array np.array used

    keep_index, pos_roi_per_this_image = get_roi_cascade_index(max_iou)
    gt_roi_labels = gt_roi_label[keep_index]
    # gt_roi_labels are gone use for ROI loss
    gt_roi_labels[pos_roi_per_this_image:] = 0 # negative labels 0
    sample_roi = roi[keep_index]
    bbox_for_sampled_roi = gt_boxes[gt_assignment[keep_index]]

    gt_roi_locs = box2loc(sample_roi, bbox_for_sampled_roi)
    rois = sample_roi
    roi_indices = np.zeros((len(rois),), dtype=np.float64)
    indices_and_rois = np.concatenate([roi_indices[:, None], rois], axis=1)
    indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    # print(indices_and_rois.shape) 20, 5
    return gt_roi_locs, gt_roi_labels, indices_and_rois
