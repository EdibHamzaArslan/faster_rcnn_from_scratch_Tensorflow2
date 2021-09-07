
input_width, input_height = (224, 224)
ratios = [0.5, 1, 2]
anchor_scales = [4, 8, 16] # the original ones 8 16 32

n_anchors = len(ratios) * len(anchor_scales)
sub_sample = 16

anchor_pos_iou_threshold = 0.7
anchor_neg_iou_threshold = 0.3

anchor_pos_ratio = 0.5
anchor_n_sample = 256

# Proposal Params

nms_threshold = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

# Proposal Targets params

pt_n_sample = 128
pt_pos_ratio = 0.25
pt_pos_iou_threshold = 0.5
pt_neg_iou_threshold_max = 0.5
pt_neg_iou_threshold_min = 0.0