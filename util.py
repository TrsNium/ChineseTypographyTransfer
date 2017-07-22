from PIL import Image, ImageFont, ImageDraw
import scipy
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

def sample(size, channel, path, batch_size):
    choice_file_names = random.sample(os.listdir(path), batch_size)
    imgs = np.empty((0,size,size,channel), dtype=np.float32)
    encode = lambda x: x/127.5 -1

    for file_name in choice_file_names:
        img = encode(scipy.misc.imread(path+file_name, mode='L').astype(np.float32))
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((-1,size,size,channel))
    return imgs

def visualize(x, fakeY, Y, batch_size, itr):
    decode = lambda x: (x+1.)/2
    imgs = np.empty((64*3, 64*3, 1), dtype=np.float32)
    for n in range(batch_size):
        img = np.hstack(x[n], fakeY[n], Y[n])
        imgs = np.append(imgs, np.array([img]), axis=0)
    imgs = imgs.reshape((64*batch_size, 64*3, 1))
    scipy.misc.imsave('./visualized/itr{}.jpg'.format(itr), decode(imgs))

def mk_font_imgs(font_path, save_path, font_size = 42, text_lists = 'moji_lists_.txt'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with open(text_lists) as fs:
        chars = fs.readlines()
    
    for char in chars:
        char = char.split('\n')[0]
        mk_img_from_font(char, font_path, save_path, 42)

mk_font_imgs('nicokaku-plus/nicokaku_v1.ttf', './nicokaku/')
mk_font_imgs('togoshi-mono-20080629/togoshi-mono.ttf', './togoshi_mono/')
