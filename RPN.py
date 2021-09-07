import tensorflow as tf
import Config as C

def rpn(rpn_inputs_shape, num_anchors=C.n_anchors):

    inputs = tf.keras.Input(shape=rpn_inputs_shape)
    x = tf.keras.layers.Conv2D(512, 3,
                               padding='same',
                               activation='relu',
                               bias_initializer='zeros')(inputs)

    # 14 14 18
    pred_cls_scores = tf.keras.layers.Conv2D(num_anchors * 1, 1,
                                             padding='same',
                                             activation='softmax',
                                             bias_initializer='zeros')(x)
    # 14 14 36
    pred_anchor_locs = tf.keras.layers.Conv2D(num_anchors * 4, 1,
                                              padding='same',
                                              activation='linear',
                                              bias_initializer='zeros')(x)

    pred_anchor_locs = tf.reshape(pred_anchor_locs, (-1, pred_anchor_locs.shape[1] * pred_anchor_locs.shape[2] * 9, 4), name="reg_layer")
    pred_cls_scores = tf.reshape(pred_cls_scores, (-1, pred_cls_scores.shape[1] * pred_cls_scores.shape[2] * 9, 1), name="cls_layer")

    return tf.keras.Model(inputs=[inputs], outputs=[pred_anchor_locs, pred_cls_scores], name='rpn')


