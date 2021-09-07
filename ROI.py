import tensorflow as tf
from ROI_Pooling import ROI_Pooling
import sys

def head(rois, feature_maps, n_rois):
    output = ROI_Pooling(7, 7, n_rois)([rois, feature_maps])
    # print(f"Before reshape {output.shape}")  # 128 7 7 512
    output = output.reshape(output.shape[0], -1)
    # print(f"After reshape {output.shape}")  # 128 25088

    inputs = tf.keras.Input(shape=output.shape, name='input')
    fc1 = tf.keras.layers.Dense(1024)(inputs)
    fc2 = tf.keras.layers.Dense(1024)(fc1)
    roi_class_box = tf.keras.layers.Dense(21 * 4, activation='linear')(fc2)
    roi_class_label = tf.keras.layers.Dense(21, activation='softmax')(fc2)
    return tf.keras.Model(inputs=[inputs],
                          outputs=[roi_class_box, roi_class_label], name="roi"), output