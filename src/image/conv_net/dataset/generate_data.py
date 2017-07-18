import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

# Input args
img_width = 64
img_height = 64
font_size = 50
font_intensity = 0
background_color = 255
draw_string = 'Test'
draw_xy = (0, 0)
img_fname = 'test.png'
font_fname = 'OpenSans-Regular.ttf'
font_folder = 'fonts'  # Relative to this file
data_folder = 'data'  # Relative to this file
# End input args

dir = os.path.dirname(__file__)
font_path = os.path.join(dir, font_folder, font_fname)
img_path = os.path.join(dir, data_folder, img_fname)

font = ImageFont.truetype(font_path, font_size)
img = Image.new('L', (img_width, img_height), background_color)
draw = ImageDraw.Draw(img)
draw.text(draw_xy, draw_string, font_intensity, font=font)
draw = ImageDraw.Draw(img)
draw = ImageDraw.Draw(img)
img.save(img_path)
