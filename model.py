import tensorflow as tf

class model():
    def __init__(self, args):
        self.args = args

        self.input_img = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        self.label_img = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])

    def generate(self, x):
        enc_conv_1 = tf.layers.batch_normalization(tf.layers.conv2d(x, 64, [3,3], (1,1), 'SAME', name='enc_conv_1'))
        enc_conv_2 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_1), 64, [4,4], (2,2), 'SAME', name='enc_conv_1'))
        enc_conv_3 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_2), 128, [3,3], (1,1), "SAME", name='enc_conv_3'))
        enc_conv_4 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_3), 128, [4,4], (2,2), 'SAME', name='enc_conv_4'))
        enc_conv_5 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_4, 256, [3,3]), (1,1), 'SAME', name='enc_conv_5'))
        enc_conv_6 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_5), 256, [4,4], (2,2), 'SAME', name='enc_conv_6'))
        enc_conv_7 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_6), 512, [3,3], (1,1), 'SAME', name='enc_conv_7'))
        enc_conv_8 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_7), 512, [4,4], (2,2), 'SAME', name='enc_conv_8'))

        dec_deconv_1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(enc_conv_8), 512, [4,4], (2,2), 'SAME', name='dec_conv_1'))
        dec_conv_2 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(tf.concat([dec_deconv_1, enc_conv_7], -1)), 512, [3,3], (1,1), 'SAME', name='dec_conv_2'))
        dec_conv_3 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(dec_conv_2), 256, [3,3], (1,1), 'SAME', name='dec_conv_3'))
        dec_deconv_4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(dec_conv_3), 256, [4,4], (2,2), 'SAME', name='dec_deconv_4'))
        dec_conv_5 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(tf.concat([dec_deconv_4, enc_conv_5], -1)), 256, [3,3], (1,1), 'SAME', name='dec_conv_5'))
        dec_conv_6 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(dec_conv_5), 128, [3,3], (1,1), 'SAME', name='dec_conv_6'))
        dec_deconv_7 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(dec_conv_6), 128, [4,4], (2,2), 'SAME', name='dec_deconv_7'))
        dec_conv_8 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(tf.concat([dec_deconv_7, enc_conv_3], -1)), 128, [3,3], (1,1), 'SAME', name='dec_conv_8'))
        dec_conv_9 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(dec_conv_8), 64, [3,3], (1,1), 'SAME', name='dec_conv_9'))
		dec_deconv_10 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(dec_conv_9), 64, [4,4], (2,2), 'SAME', name='dec_deconv_10'))
		dec_conv_11 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(tf.concat([dec_deconv_10, enc_conv_1], -1)), 64, [3,3], (1,1), 'SAME', name='dec_conv_11'))
		dec_conv_12 = tf.layers.conv2d(tf.nn.relu(dec_conv_11), 1, [3,3], (1,1), 'SAME', name='dec_conv_12')
		return tf.nn.tanh(dec_conv_12)
