# import tensorflow as tf
import numpy as np


def iou(boxA, boxB):
    """
    https://keras.io/examples/vision/retinanet/#computing-pairwise-intersection-over-union-iou
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """

    ya1, xa1, ya2, xa2 = boxA
    yb1, xb1, yb2, xb2 = boxB

    x_min = np.max([xb1, xa1])
    y_min = np.max([yb1, ya1])
    x_max = np.min([xb2, xa2])
    y_max = np.min([yb2, ya2])

    inter_area = np.max((x_max - x_min), 0) * np.max((y_max - y_min), 0)

    boxA_area = (ya2 - ya1) * (xa2 - xa1)
    boxB_area = (yb2 - yb1) * (xb2 - xb1)

    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return np.clip(iou, 0.0, 1.0)

def box2loc(src_box, dst_box):
    """
    https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py
    """
    height = src_box[:, 2] - src_box[:, 0]
    width = src_box[:, 3] - src_box[:, 1]
    ctr_y = src_box[:, 0] + 0.5 * height
    ctr_x = src_box[:, 1] + 0.5 * width

    base_height = dst_box[:, 2] - dst_box[:, 0]
    base_width = dst_box[:, 3] - dst_box[:, 1]
    base_ctr_y = dst_box[:, 0] + 0.5 * base_height
    base_ctr_x = dst_box[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def loc2box(src_box, loc):
    """
    https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py
    """
    if src_box.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_box.astype(src_box.dtype, copy=False)

    # print(f"src box shape {src_bbox.shape}")
    # print(f"loc box shape {loc.shape}")
    src_height = src_bbox[:, :, 2] - src_bbox[:, :, 0]
    src_width = src_bbox[:, :, 3] - src_bbox[:, :, 1]
    src_ctr_y = src_bbox[:, :, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, :, 1] + 0.5 * src_width

    dy = loc[:, :, 0::4]
    dx = loc[:, :, 1::4]
    dh = loc[:, :, 2::4]
    dw = loc[:, :, 3::4]

    ctr_y = dy * src_height[:, :, np.newaxis] + src_ctr_y[:, :, np.newaxis]
    # print(f"***ctr_y shape {ctr_y.shape}")
    ctr_x = dx * src_width[:, :, np.newaxis] + src_ctr_x[:, :, np.newaxis]
    # print(f"***ctr_x shape {ctr_x.shape}")
    h = np.exp(dh) * src_height[:, :, np.newaxis]
    w = np.exp(dw) * src_width[:, :, np.newaxis]
    # print(f"[*]h shape {h.shape} [*]w shape {w.shape}")
    dst_bbox = np.zeros(loc.shape, dtype=np.float32)
    # print(f"dst box shape is {dst_bbox.shape}")
    dst_bbox[:, :, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, :, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, :, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, :, 3::4] = ctr_x + 0.5 * w
    return dst_bbox

def nms(boxes, scores, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick