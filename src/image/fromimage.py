import os

import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

import mark_detect.models.custom_model as custom_model


def board_from_image(board_img):
    board_img = board_img.convert('L')
    tiles = split_board_into_tiles(board_img)
    return [[detect_mark(tile) for tile in row] for row in tiles]


def split_board_into_tiles(board_img):
    width, height = board_img.size
    tiles = [[None for i in range(3)] for j in range(3)]
    for i in range(3):
        for j in range(3):
            box = (j * width // 3, i * height // 3,
                   (j + 1) * width // 3, (i + 1) * height // 3)
            tiles[i][j] = board_img.crop(box)
    return tiles


def detect_mark(tile):
    tile.show()
    tile = tile.resize((64, 64))  # TODO: Fix size here
    rel_dir = os.path.dirname(__file__)
    model_path = os.path.join(
        rel_dir, 'mark_detect', 'checkpoints', 'model_best.pth.tar')
    model = custom_model.Net()
    if not os.path.isfile(model_path):
        raise FileNotFoundError('Could not find trained model')

    checkpoint = torch.load(model_path)  # TODO load outside of func
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    transform = transforms.ToTensor()
    data = transform(tile)
    data = data.view(1, 1, 64, 64)
    #data = Variable(data)
    output = model(data)
    pred = output.max(1)[1]
    #pred = output.data.max(1)[1]
    pred = pred[0].numpy()[0]
    if pred == 0:
        return 'x'
    elif pred == 1:
        return 'o'
    elif pred == 2:
        return ' '
    else:
        raise ValueError("Model gives wrong output")


def main():
    rel_dir = os.path.dirname(__file__)
    print(rel_dir)
    filename = os.path.join(rel_dir, 'board', 'testboards', 'pen.jpg')
    board = Image.open(filename)
    ret = board_from_image(board)
    print(ret)


if __name__ == '__main__':
    main()
