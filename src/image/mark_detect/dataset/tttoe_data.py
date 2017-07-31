import os
import glob
from random import choice

import numpy as np
from numpy.random import randint
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import torch.utils.data as data


class Tttoe_dataset(data.Dataset):
    fonts_folder = 'fonts'

    def __init__(self, num_elements, side_length, text_size, rand_offset_limit, noise_alpha=None, img_transform=None, target_transform=None):
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.num_elements = num_elements
        self.size = (side_length, side_length)
        self.rel_dir = os.path.dirname(__file__)
        if noise_alpha is not None:
            self.should_add_noise = True
            self.noise_alpha = noise_alpha
        else:
            self.should_add_noise = False

        chars = 'xo XO '
        fonts = glob.glob(os.path.join(
            self.rel_dir, self.fonts_folder, '*.ttf'))
        self.data = [None for i in range(self.num_elements)]
        for i in range(self.num_elements):
            char = choice(chars)
            rand_offset = randint(-rand_offset_limit, rand_offset_limit, 2)
            if char == 'x' or char == 'X':
                target = 0
            elif char == 'o' or char == 'O':
                target = 1
            else:
                target = 2
            image = self.make_data(char, text_size, rand_offset, 0,
                                   self.size, 255, choice(fonts))
            self.data[i] = [image, target]

    def __getitem__(self, index):
        img, target = self.data[index]
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.num_elements

    def make_data(self, text_str, text_size, text_offset, text_intensity,
                  img_size, bg_intensity, font_path):
        font = ImageFont.truetype(font_path, text_size)
        img = Image.new('L', img_size, bg_intensity)
        draw = ImageDraw.Draw(img)
        text_w, text_h = draw.textsize(text_str, font=font)

        x_pos = (img_size[0] - text_w) / 2 + text_offset[0]
        y_pos = (img_size[1] - text_h) / 2 + text_offset[1]

        draw.text((x_pos, y_pos), text_str, text_intensity, font=font)
        if self.should_add_noise:
            img = self.add_noise(img)
        return img

    def add_noise(self, img):
        noise = randint(0, 255, self.size, dtype=np.uint8)
        noise = Image.fromarray(noise, 'L')

        return Image.blend(img, noise, self.noise_alpha)


if __name__ == '__main__':
    data = Tttoe_dataset(10, 64, 30, 10, 0.2)
    img, label = data[0]
    print(label)
    img.show()