import os


import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import mark_detect.models.custom_model as custom_model


from sklearn.cluster import DBSCAN


def board_from_image(board_img):
    board_img = board_img.convert('L')
    tiles = split_board_into_tiles(board_img)
    return [[detect_mark(tile) for tile in row] for row in tiles]


def wrap_line(line):
    rho, theta = line
    if rho < 0:
        rho = -rho
        theta = theta - np.pi
    return rho, theta


def split_board_into_tiles(board_img):
    img = np.array(board_img)
    w, h = img.shape
    img_col = np.empty((w, h, 3), dtype=np.uint8)
    img_col[:, :, 2] = img_col[:, :, 1] = img_col[:, :, 0] = img
    edges = cv2.Canny(img, 300, 600, L2gradient=False)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    def line_distance(X1, X2):
        rho1, theta1 = X1
        rho2, theta2 = X2

        dist_cw = (theta1 - theta2) % (2 * np.pi)
        dist_ccw = (theta2 - theta1) % (2 * np.pi)
        dist_theta = min(dist_cw, dist_ccw)

        dist_rho = abs(rho2 - rho1)
        return dist_theta / np.pi + dist_rho / (h + w)

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

    # Plot image clustered lines
    for rho, theta in clustered_lines:
        # ignore nonvertical/nonhorizontal lines
        modpi_2 = theta % (np.pi / 2)
        if np.pi / 12 < modpi_2 and modpi_2 < np.pi * 5 / 12:
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_col, (x1, y1), (x2, y2), (255, 0, 0), 5)
    plt.imshow(img_col, cmap='Greys')
    plt.show()

    width, height = board_img.size
    tiles = [[None for i in range(3)] for j in range(3)]
    for i in range(3):
        for j in range(3):
            box = (j * width // 3, i * height // 3,
                   (j + 1) * width // 3, (i + 1) * height // 3)
            tiles[i][j] = board_img.crop(box)
    return tiles


def detect_mark(tile):
    # tile.show()
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


def main():
    rel_dir = os.path.dirname(__file__)
    print(rel_dir)
    filename = os.path.join(rel_dir, 'board', 'testboards', 'pen.jpg')
    board = Image.open(filename)
    ret = board_from_image(board)
    print(ret)


if __name__ == '__main__':
    main()
