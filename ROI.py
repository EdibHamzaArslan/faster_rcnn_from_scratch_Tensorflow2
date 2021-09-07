import tensorflow as tf
from ROI_Pooling import ROI_Pooling


def head(rois, feature_maps, n_rois):
    output = ROI_Pooling(7, 7, n_rois)([rois, feature_maps])
    # print(output.shape)
    output = output.reshape(output.shape[0], -1)
    # print(output.shape)
    # model = tf.keras.Sequential(layers=[tf.keras.layers.Dense(1024),
    #                                     tf.keras.layers.Dense(1024)])
    inputs = tf.keras.Input(shape=output.shape)
    fc1 = tf.keras.layers.Dense(1024)(inputs)
    fc2 = tf.keras.layers.Dense(1024)(fc1)
    roi_class_box = tf.keras.layers.Dense(21 * 4)(fc2)
    roi_class_label = tf.keras.layers.Dense(21, activation='softmax')(fc2)
    return tf.keras.Model(inputs=[inputs],
                          outputs=[roi_class_box, roi_class_label], name="roi"), output