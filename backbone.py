import tensorflow as tf
from tensorflow.keras.applications import VGG16


def get_backbone(input_tensor):
    backbone = VGG16(include_top=False, input_tensor=input_tensor)
    final_layer = backbone.get_layer('block5_conv3').output
    return tf.keras.Model(inputs=[backbone.inputs], outputs=[final_layer], name='backbone')