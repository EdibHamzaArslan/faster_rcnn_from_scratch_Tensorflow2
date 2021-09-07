import tensorflow as tf
# import tensorflow_addons as tfa
import Config as C
import numpy as np
import sys
tf.get_logger().setLevel('ERROR')

class ROI_Pooling(tf.keras.layers.Layer):
    def __init__(self, pool_size_width, pool_size_height, num_rois):
        super(ROI_Pooling, self).__init__()
        self.pool_size_width = pool_size_width
        self.pool_size_height = pool_size_height
        self.num_rois = num_rois
        # self.pooling_layer = tfa.layers.AdaptiveMaxPooling2D((self.pool_size_width,
        #                                                      self.pool_size_height))

    def _roi_pool(self, features):
        """
        https://github.com/Parsa33033/RoiPooling/blob/master/RoiPooling.py
        """
        h, w, num_channels = features.shape
        h_stride = h / self.pool_size_height
        w_stride = w / self.pool_size_width
        pool = np.zeros((self.pool_size_width, self.pool_size_height, num_channels))
        for i in range(self.pool_size_width):
            for j in range(self.pool_size_height):
                w_start = int(np.floor(j * w_stride))
                w_end = int(np.ceil((j + 1) * w_stride))
                h_start = int(np.floor(i * h_stride))
                h_end = int(np.ceil((i + 1) * h_stride))

                # limiting start and end based on feature limits
                w_start = min(max(w_start, 0), w)
                w_end = min(max(w_end, 0), w)
                h_start = min(max(h_start, 0), h)
                h_end = min(max(h_end, 0), h)
                patch = features[h_start:h_end, w_start:w_end, :]
                # print(f"patch.shape {patch.shape}")
                max_val = np.max(patch.reshape(-1, num_channels)) # TODO Can be bag
                pool[i, j, :] = max_val
        return pool

    def call(self, inputs, **kwargs):
        rois = inputs[0]
        feature_maps = inputs[1]
        rois = np.array(rois)
        feature_maps = np.array(feature_maps)
        assert rois.shape[1] == 5, "The rois should have 5 index!"
        assert len(feature_maps.shape) == 3, "Remove batch index from feature maps!"
        rois[:, 1:] = np.multiply(rois[:, 1:], 1/C.sub_sample) # C.sub_sample => 16
        rois = rois.astype(np.int)

        # self.num_rois = rois.size(0)
        output = np.zeros((self.num_rois,
                           self.pool_size_width,
                           self.pool_size_height,
                           feature_maps.shape[-1]))
        for i in range(self.num_rois):
            roi = rois[i]
            extracted_feature = feature_maps[roi[2]:(roi[4]+1), roi[1]:(roi[3]+1), ...]
            pool = self._roi_pool(extracted_feature)
            output[i] = pool

        return output

if __name__ == '__main__':
    inputs = tf.keras.Input(())