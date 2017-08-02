''' Dataset for tic tac toe '''

import os
import glob
from random import choice

import numpy as np
from numpy.random import randint
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import torch.utils.data as data


class RandomRotate():
    ''' Rotates the given PIL image from [-max_angle, max_angle] degrees '''

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, img):
        rand_angle = randint(-self.max_angle, self.max_angle)

        return img.rotate(rand_angle, expand=False)


class TttoeDataset(data.Dataset):
    """ Dataset class generating images with either blank field, cross or circle, as well as applying
    data transformations to the images. Cross and circle is drawn from fonts in the fonts folder
    """

    fonts_folder = 'fonts'

    def __init__(self, num_elements, side_length, text_size, rand_offset_limit,
                 noise_alpha=None, rect_pos=None, img_transform=None, target_transform=None):
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

        if rect_pos is not None:
            self.should_add_rect = True
            self.rect_pos = rect_pos
        else:
            self.should_add_rect = False

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
            img = self.make_data(char, text_size, rand_offset, 0,
                                 self.size, 255, choice(fonts))

            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.data[i] = [img, target]

    def __getitem__(self, index):
        img, target = self.data[index]

        return img, target

    def __len__(self):
        return self.num_elements

    def make_data(self, text_str, text_size, text_offset, text_intensity,
                  img_size, bg_intensity, font_path):
        font = ImageFont.truetype(font_path, text_size)
        img = Image.new('L', img_size, bg_intensity)

        if self.should_add_noise:
            img = self.add_noise(img)

        draw = ImageDraw.Draw(img)
        text_w, text_h = draw.textsize(text_str, font=font)
        x_pos = (img_size[0] - text_w) / 2 + text_offset[0]
        y_pos = (img_size[1] - text_h) / 2 + text_offset[1]
        draw.text((x_pos, y_pos), text_str, text_intensity, font=font)

        if self.should_add_rect:
            img = self.add_rectangle(img)

        return img

    def add_noise(self, img):
        noise = randint(0, 255, self.size, dtype=np.uint8)
        noise = Image.fromarray(noise, 'L')

        return Image.blend(img, noise, self.noise_alpha)

    def add_rectangle(self, img):
        # make more scalable?
        draw = ImageDraw.Draw(img)
        rect_offs = randint(-10, 10, 4)
        x1, y1, x2, y2 = self.rect_pos + rect_offs
        rect_width = randint(3, 5)
        draw.line([x1, y1, x1, y2], 0, width=rect_width)
        draw.line([x1, y2, x2, y2], 0, width=rect_width)
        draw.line([x2, y2, x2, y1], 0, width=rect_width)
        draw.line([x2, y1, x1, y1], 0, width=rect_width)

        return img


def main():
    side_length = 76
    text_size = 60
    num_elements = 20
    rand_offs = 10
    import torchvision.transforms as tf
    dataset = TttoeDataset(num_elements, side_length, text_size, rand_offs, 0.2, [
                           0, 0, side_length, side_length], tf.Compose([RandomRotate(20), tf.CenterCrop(64)]))
    img, label = dataset[0]
    print(label)
    img.show()


if __name__ == '__main__':
    main()
