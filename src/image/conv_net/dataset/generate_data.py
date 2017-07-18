import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def make_PIL_img_with_text(text_str, text_size, text_pos, text_intensity,
                           img_size, bg_intensity, font_path):
    font = ImageFont.truetype(font_path, text_size)
    img = Image.new('L', img_size, bg_intensity)
    draw = ImageDraw.Draw(img)
    draw.text(text_pos, text_str, text_intensity, font=font)
    return img


def main():
    text_str = 'x'
    text_size = 50
    text_pos = (0, 0)
    text_intensity = 0
    img_size = (60, 60)
    bg_intensity = 255
    img_fname = 'test.png'
    font_fname = 'OpenSans-BoldItalic.ttf'
    font_folder = 'fonts'  # Relative to this file
    data_folder = 'data'  # Relative to this file

    rel_dir = os.path.dirname(__file__)
    font_path = os.path.join(rel_dir, font_folder, font_fname)

    img = make_PIL_img_with_text(text_str, text_size, text_pos,
                                 text_intensity, img_size, bg_intensity, font_path)

    img_path = os.path.join(rel_dir, data_folder, img_fname)
    img.save(img_path)


if __name__ == '__main__':
    main()
