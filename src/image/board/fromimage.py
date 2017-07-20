import os
import math
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable


import image.mark_detect as md
from image.mark_detect.models.custom_model import Net
crosspath = './shapes/cross.png'


def from_image(filename):
    ret = [[' ', ' ', ' '], ['', ' ', ' '], [' ', ' ', ' ']]
    board_img = Image.open(filename).convert('L')

    # Divide board into 9 pics and search each individual pic
    # How to find the squares:
    # - search for horizontal and vertical lines
    #   that span a big portion of the image, and find their crossings
    # - divide image in 3x3 images
    tiles = split_board_into_tiles(board_img)
    ret = [[detect_mark(tile) for tile in row] for row in tiles]
    return ret


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
    tile = tile.resize((64, 64))  # TODO: Fix size here
    rel_dir = os.path.dirname(__file__)
    model_path = os.path.join(
        rel_dir, '..', 'mark_detect', 'checkpoints', 'model_best.pth.tar')
    model = Net()
    if not os.path.isfile(model_path):
        raise FileNotFoundError('Could not find trained model')

    checkpoint = torch.load(model_path)  # TODO load outside of func
    model.load_state_dict(checkpoint['state_dict'])
    transform = transforms.ToTensor()
    data = transform(tile)
    data = data.view(1, 1, 64, 64)
    data = Variable(data)
    output = model(data)
    pred = output.data.max(1)[1]
    pred = pred[0].numpy()[0]
    if pred == 0:
        return 'x'
    elif pred == 1:
        return 'o'
    elif pred == 2:
        return ' '
    else:
        raise ValueError("Model gives wrong output")


'''
def detectSquare(im_square):

    height, width = im_square.shape[:2]
    gray = cv2.cvtColor(im_square, cv2.COLOR_BGR2GRAY)

    # Check for circle using hough transform
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
    if circles != None:
        for circ in circles:
            # Check that the radius is of a certain size
            radius = circ[0][2]
            if radius > 0.2 * (height + width) / 2:
                return 'o'

    # Check for cross
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html#fast
    # http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    # http://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
    im_cross = cv2.imread(crosspath)
    # Scale cross to smaller size than square
    im_cross = cv2.resize(im_cross, (int(0.9 * width), int(0.9 * height)))
    im_cross = cv2.cvtColor(im_cross, cv2.COLOR_BGR2GRAY)
    edges_cross = cv2.Canny(im_cross, 100, 200)
    edges_square = cv2.Canny(im_square, 100, 200)
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(edges_cross, edges_square, method)
    minval, maxval, minpos, maxpos = cv2.minMaxLoc(res)
    print(minval, maxval, minpos, maxpos)
    if maxval > 0.5:
        return 'x'
    return ' '


def edgeHistogram(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    theta = np.arctan2(sobely, sobelx)
    print np.min(theta)
    cv2.imshow('b', theta[5:10, 0:2])
    cv2.waitKey()

# Take an image and turn it into rho,theta space for lines
# Inputs:
#-Opencv numpy image
# Outputs:
#-Opencv matrix where rows corresponds to rho, cols corresponds to theta axis


def houghLineTransform(image):
    start = time.time()

    # Resolution of hough transform along theta axis
    resTh = 1

    # Find edges
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(bw, 150, 200)
    cv2.imshow('testimg', edges)
    height, width = image.shape[:2]
    lineTransform = np.zeros(
        (int(180 / resTh), int(np.sqrt(2) * np.max((width, height)))))
    for row in range(height):
        for col in range(width):
            if edges[row, col] > 200:
                for theta in range(0, 180 / resTh, resTh):
                    rho = col * np.cos(np.pi * theta / 180) + \
                        (height - row) * np.sin(np.pi * theta / 180)
                    lineTransform[theta, int(abs(rho))] += 1
    print(time.time() - start)
    return lineTransform / lineTransform.max()


def testHLine():
    img = cv2.imread('../testboards/board4.jpg')
    lineTransform = houghLineTransform(img)
    cv2.imshow('lineT', lineTransform)
    cv2.waitKey()
    maxes = [0 for i in range(9)]
    rows = [0 for i in range(9)]
    cols = [0 for i in range(9)]
    height, width = lineTransform.shape[:2]

    # Find lines corresponding to grid
    for row in range(height):
        for col in range(width):
            index = 0
            while lineTransform[row, col] > maxes[index] and index < 8:
                index += 1
            maxes.insert(index, lineTransform[row, col])
            maxes.pop(0)
            rows.insert(index, row)
            rows.pop(0)
            cols.insert(index, col)
            cols.pop(0)
    maxes.pop()
    cols.pop()
    rows.pop()
    maxes = [x for (y, x) in sorted(zip(rows, maxes))]
    cols = [x for (y, x) in sorted(zip(rows, cols))]
    rows = sorted(rows)
    # print maxes
    # print rows
    # print cols

    # Draw lines
    pt1 = [0, 0]
    pt2 = [10, 10]
    height, width = img.shape[:2]
    for i in range(8):
        rho = cols[i]
        theta = rows[i]
        if -2 < theta < 2:
            pt1 = [rho, height]
            pt2 = [rho, 0]
        else:
            pt1 = [0, height - rho / np.sin(theta * np.pi / 180) + 1]
            pt2 = [width, height - np.cos(theta * np.pi / 180) / np.sin(
                theta * np.pi / 180) * width - rho / np.sin(theta * np.pi / 180) + 1]

        cv2.line(img, tuple(map(int, pt1)), tuple(
            map(int, pt2)), (0, 255, 0), 2)
    cv2.imshow('imline', img)
    cv2.waitKey(0)

'''
if __name__ == '__main__':
    ret = from_image('test.png')
    print(ret)
