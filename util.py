from PIL import Image, ImageFont, ImageDraw

def mk_img_from_font(char, font_path, save_path, font_size = 15):
    image = Image.new('RGB', (64, 64), '#FFF')
    draw = ImageDraw.Draw(image)

    # use a truetype font
    font = ImageFont.truetype(font_path, font_size)
    tw, th = font.getsize(char)
    draw.text(((64-tw)/2, (64-th)/2), char, font=font, fill=(0,0,0))
    image.save(save_path+char+'.jpg')

mk_img_from_font('あ', 'nicokaku-plus/nicokaku_v1.ttf', './nicokaku_', 42)
mk_img_from_font('あ', './togoshi-mono-20080629/togoshi-mono.ttf', './togoshi_mono', 42)
