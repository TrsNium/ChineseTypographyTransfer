import tensorflow as tf
import numpy as np
import argparse
import os 
from util  import *

class model():
    def __init__(self, args):
        self.args = args

        self.input_img = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        self.label_img = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        
        self.fake_ = self.generate(self.input_img, 'Generator')
        dis_fake = self.discriminator(self.fake_, 'Discriminator', False)
        dis_real = self.discriminator(self.label_img, 'Discriminator', True)

        V1_loss = -self.label_img * args.lambda_ * (tf.log(tf.nn.sigmoid(self.fake_)) - (1 - self.label_img)*tf.log(1 - tf.nn.sigmoid(self.fake_)))
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
            h1 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(h0), filters=128, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h1_conv'), name="d_bn_h1")
            h2 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(h1), filters=256, kernel_size=[4,4], strides=(2,2), padding='SAME', name='d_h2_conv'), name="d_bn_h2")
            h3 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(h2), filters=512, kernel_size=[4,4], strides=(1,1), padding='SAME', name='d_h3_conv'), name="d_bn_h3")
            out = tf.layers.conv2d(tf.nn.relu(h3), filters = 1, kernel_size=[4,4], strides=(1,1), name='d_out_conv')
            return out

    def generate(self, x, name):
        with tf.variable_scope(name) as scope:
            enc_conv_1 = tf.layers.batch_normalization(tf.layers.conv2d(x, 64, [3,3], (1,1), 'SAME', name='enc_conv_1'))
            enc_conv_2 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_1), 64, [4,4], (2,2), 'SAME', name='enc_conv_2'))
            enc_conv_3 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_2), 128, [3,3], (1,1), "SAME", name='enc_conv_3'))
            enc_conv_4 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_3), 128, [4,4], (2,2), 'SAME', name='enc_conv_4'))
            enc_conv_5 = tf.layers.batch_normalization(tf.layers.conv2d(tf.nn.relu(enc_conv_4), 256, [3,3], (1,1), 'SAME', name='enc_conv_5'))
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
            dec_conv_12 = tf.layers.conv2d(tf.nn.relu(dec_conv_11), 1, [1,1], (1,1), 'SAME', name='dec_conv_12')
            return tf.nn.tanh(dec_conv_12)

    def train(self):
        opt_g = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.gen_loss, var_list=self.gen_var)
        opt_d = tf.train.AdamOptimizer(self.args.lr, beta1=self.args.beta1).minimize(self.dis_loss, var_list=self.dis_var)

        sample_fun = sample_files_function(self.args.batch_size)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter("./logs", sess.graph)

            for itr in range(self.args.itrs):

                sampled_file_name = sample_fun()
                content_imgs = sample(64, 1, self.args.content_dir, sampled_file_name)
                style_imgs = sample(64, 1, self.args.style_dir, sampled_file_name)

                g_loss, _ = sess.run([self.gen_loss, opt_g], feed_dict={self.input_img:content_imgs})
                d_loss, _ = sess.run([self.dis_loss, opt_d], feed_dict={self.input_img:content_imgs, self.label_img:style_imgs})

                if self.args.visualize and itr%100==0:
                    fake_ = sess.run(self.fake_, feed_dict={self.input_img:content_imgs})
                    visualize(content_imgs, fake_, style_imgs, self.args.batch_size, itr)

                if itr%100 == 0:
                    print('itr:', itr, '    d_loss:', d_loss, '    g_loss:', g_loss)

                if itr%1000==0:
                    print('---------------------saved model------------------------------')
                    saver.save(sess, 'save/model.ckpt')

if __name__ == '__main__':
    argp = argparse.ArgumentParser(description="")
    argp.add_argument("--lr", dest="lr", type=float, default= 0.003)
    argp.add_argument("--lambda", dest="lambda_", type=float, default= 0.0002)
    argp.add_argument("--itrs", dest="itrs", type=int, default=3000000)
    argp.add_argument("--batch_size", dest="batch_size", type=int, default=3)
    argp.add_argument("--visualize", dest="visualize", type=bool, default=True)
    argp.add_argument("--beta1", dest="beta1", type=float, default=0.5)
    argp.add_argument("--content_dir", dest="content_dir", type=str, default="./togoshi_mono/")
    argp.add_argument("--style_dir", dest="style_dir", type=str, default="./nicokaku/")
    argp.add_argument("--content_font_dir", dest="content_font_dir", type=str, default="./togoshi-mono-20080629/")
    argp.add_argument("--style_font_dir", dest="style_font_dir", type=str, default="./nicokaku-plus/")
    argp.add_argument("--train", dest="train", type=bool, default=True)
    args = argp.parse_args()

    if not os.path.exists('./saved/'):
        os.mkdir('./saved/')

    if not os.path.exists('./visualized/'):
        os.mkdir('./visualized/')

    if not os.path.exists(args.content_dir):
        mk_font_imgs(args.content_font_dir)
    
    if not os.path.exists(args.style_dir):
        mk_font_imgs(args.style_font_dir)

    model_ = model(args)
    if args.train:
        model_.train()
    else:
        pass