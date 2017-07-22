from PIL import Image, ImageFont, ImageDraw
import scipy.misc
import numpy as np
import random 
import os 

def mk_img_from_font(char, font_path, save_path, font_size = 15):
    image = Image.new('RGB', (64, 64), '#FFF')
    draw = ImageDraw.Draw(image)

    # use a truetype font
    font = ImageFont.truetype(font_path, font_size)
    tw, th = font.getsize(char)
    draw.text(((64-tw)/2, (64-th)/2), char, font=font, fill=(0,0,0))
    image.save(save_path+char+'.jpg')

def sample(size, channel, fontimg_path, file_names):
    imgs = np.empty((0,size,size,channel), dtype=np.float32)
    encode = lambda x: x/127.5 -1
    #encode = lambda x: x/1

    for file_name in file_names:
        img = np.expand_dims(encode(scipy.misc.imread(fontimg_path+file_name+'.jpg', mode='L').astype(np.float32)), -1)
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((-1,size,size,channel))
    return imgs

def visualize(x, fakeY, Y, batch_size, itr):
    decode = lambda x: (x+1.)/2
    #decode = lambda x: x/1
    imgs = np.empty((0, 64, 64*3, 1), dtype=np.float32)
    for n in range(batch_size):
        img = np.hstack((x[n], fakeY[n], Y[n]))
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((64*batch_size, 64*3))
    scipy.misc.imsave('./visualized/itr{}.jpg'.format(itr), decode(imgs))

def mk_font_imgs(font_path, save_path, font_size = 42, text_lists = 'moji_lists_.txt'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with open(text_lists) as fs:
        chars = fs.readlines()
    
    for char in chars:
        char = char.split('\n')[0]
        mk_img_from_font(char, font_path, save_path, 42)

def sample_files_function(batch_size, text_lists = 'moji_lists_.txt'):
    with open(text_lists) as fs:
        chars = fs.readlines()
    
    char_lists = []
    for char in chars:
        char = char.split('\n')[0]
        char_lists.append(char)
    
    def r_random():
        return random.sample(char_lists, batch_size)

    return r_random
