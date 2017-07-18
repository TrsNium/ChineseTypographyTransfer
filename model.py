import tensorflow as tf

class model():
    def __init__(self, args):
        self.args = args

        self.input_img = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        self.label_img = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        
        fake_ = self.generate(self.input_img, 'Generator')
        dis_fake = self.discriminator(fake_, 'Discriminator', False)
        dis_real = self.discriminator(self.label_img, 'Discriminator', True)

        V1_loss = -self.label_img * args.lambda * (tf.log(tf.nn.sigmoid(fake_)) - (1 - self.label_img)*tf.log(1 - tf.nn.sigmoid(fake_)))
        self.gen_loss = tf.reduce_mean(tf.square(dis_fake - tf.ones_like(dis_fake))) + V1_loss
        self.dis_loss = tf.reduce_mean(tf.square(dis_fake - tf.zeros_like(dis_fake))) + tf.reduce_mean(tf.square(dis_real - tf.ones_like(dis_real)))
        
        trainable_bar = tf.trainable_variables()
        self.dis_var = [var for var in trainable_bar if 'Discriminator' in var.name]
        self.gen_var = [var for var in trainable_bar if 'Generator' in var.name]


    def discriminator(self, x, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = tf.layers.conv2d(x, filters=64, kernel_size=[4,4], strides=(2,2), padding='SAME',name='d_h0_conv')
            h1 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(h0), filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h1_conv'), name="d_bn_h1")
            h2 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(h1), filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h2_conv'), name="d_bn_h2")
            h3 = tf.layers.batch_normalization(tf.layers.conv2d(self.lrelu(h2), filters=512, kernel_size=[4,4], strides=(1,1), padding='SAME', name='d_h3_conv'), name="d_bn_h3")
            out = tf.layers.conv2d(self.lrelu(h3), filters = 1, kernel_size=[4,4], strides=(1,1), name='d_out_conv')
            return out

    def generate(self, x, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

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

    def train(self):
        opt_g = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.gen_loss, var_list=self.gen_var)
        opt_d = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.dis_loss, var_list=self.dis_var)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter("./logs", sess.graph)

            for itr in range(self.args.itrs):

                if self.args.visualize and itr%100==0:
                    pass
                
                if l%1000==0:
                    saver.save(sess, 'save/model.ckpt')

                print(itr,':    d_loss:',d_loss,'    g_loss:',g_loss)

