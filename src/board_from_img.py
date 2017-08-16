import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import tile_detect.detect as dt


from sklearn.cluster import DBSCAN


def board_from_image(board_img):
    board_img = board_img.convert('L')
    tiles = split_board_into_tiles(board_img)
    return [[dt.detect(tile) for tile in row] for row in tiles]


def split_board_into_tiles(board_img):
    img = np.array(board_img)
    width, height = img.shape

    # Find board lines
    edges = cv2.Canny(img, 300, 600, L2gradient=False)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Cluster board lines
    def line_distance(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        dist_cw = (theta1 - theta2) % (2 * np.pi)
        dist_ccw = (theta2 - theta1) % (2 * np.pi)
        dist_theta = min(dist_cw, dist_ccw)

        dist_rho = abs(rho2 - rho1)
        return dist_theta / np.pi + dist_rho / (height + width)
    # Wrap coordinates to positive rho for easier dist calculations
    X = np.apply_along_axis(wrap_line, 1, lines[:, 0, :])
    db = DBSCAN(eps=0.1, min_samples=1, metric=line_distance).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Cluster lines
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    clustered_lines = np.empty([n_clusters_, 2])
    for i, label in enumerate(unique_labels):
        class_member_mask = (labels == label)
        xy = X[class_member_mask & core_samples_mask]
        clustered_lines[i] = xy.mean(0)

    # Find crossing points of lines
    # Filter out lines with angle greater than max angle from horizontal or vertical
    max_angle = 15 * np.pi / 180
    vertical_lines = np.zeros((4, 2))
    horizontal_lines = np.zeros((4, 2))
    vertical_lines[1:3] = clustered_lines[(abs(clustered_lines[:, 1]) <= max_angle) | (
        abs(abs(clustered_lines[:, 1]) - np.pi) <= max_angle)]
    horizontal_lines[1:3] = clustered_lines[(abs((abs(clustered_lines[:, 1]) - np.pi / 2)) <= max_angle) | (
        abs(abs(clustered_lines[:, 1]) - 3 / 2 * np.pi) <= max_angle)]
    horizontal_lines[0] = (0, np.pi / 2)
    horizontal_lines[3] = (height, np.pi / 2)
    vertical_lines[0] = (0, 0)
    vertical_lines[3] = (width, 0)

    tile_corners = calc_corners(
        vertical_lines, horizontal_lines)
    print(tile_corners)
    plt.imshow(img)
    plt.show()
    # Transform input image into tiles
    tiles = [[None for i in range(3)] for j in range(3)]
    for i in range(3):
        for j in range(3):
            data = np.concatenate(
                (tile_corners[i][j], tile_corners[i + 1][j],
                 tile_corners[i + 1][j + 1], tile_corners[i][j + 1]))
            tiles[i][j] = board_img.transform((64, 64), PIL.Image.QUAD, data)
            # tiles[i][j].show()
    return tiles


def wrap_line(line):
    rho, theta = line
    if rho < 0:
        rho = -rho
        theta = theta - np.pi
    return rho, theta


def calc_corners(vertical_lines, horizontal_lines):
    corners = [[None for i in range(4)] for j in range(4)]

    vertical_lines = np.sort(vertical_lines, 0)
    horizontal_lines = np.sort(horizontal_lines, 0)

    for i in range(4):
        for j in range(4):
            corners[i][j] = lines_crossing_point(
                horizontal_lines[i], vertical_lines[j])

    return corners


def lines_crossing_point(line1, line2):
    line1 = polar_line_to_euclidian(line1)
    line2 = polar_line_to_euclidian(line2)

    crossing_point = np.cross(line1, line2)
    return crossing_point[0:2] / crossing_point[2]


def polar_line_to_euclidian(line):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    c = - rho
    return np.array([a, b, c])


def main():
    rel_dir = os.path.dirname(__file__)
    print(rel_dir)
    filename = os.path.join(rel_dir, 'board', 'testboards', 'pen.jpg')
    board = Image.open(filename)
    ret = board_from_image(board)
    print(ret)


if __name__ == '__main__':
    main()
