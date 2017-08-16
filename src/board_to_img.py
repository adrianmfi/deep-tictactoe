import os
from PIL import Image
WIDTH = 512
HEIGHT = 512


def to_image(board_arr, img_name):
    img = Image.new("RGB", (WIDTH, HEIGHT))
    shapes_path = 'shapes'
    rel_dir = os.path.dirname(__file__)

    circ = Image.open(os.path.join(rel_dir, shapes_path, 'circle.png')).resize((WIDTH //
                                                                                4, HEIGHT // 4), Image.ANTIALIAS)
    cross = Image.open(os.path.join(rel_dir, shapes_path, 'cross.png')).resize((WIDTH //
                                                                                4, HEIGHT // 4), Image.ANTIALIAS)

    pix = img.load()
    for x_pos in range(WIDTH):
        for y_pos in range(HEIGHT):
            pix[x_pos, y_pos] = (255, 255, 255)

    for x_pos in range(WIDTH):
        for y_pos in range(int(HEIGHT // 3) - int(HEIGHT // 64), int(HEIGHT // 3) + int(HEIGHT // 64)):
            pix[x_pos, y_pos] = (0, 0, 0)

    for x_pos in range(WIDTH):
        for y_pos in range(int(2 * HEIGHT // 3) - int(HEIGHT // 64), int(2 * HEIGHT // 3) + int(HEIGHT // 64)):
            pix[x_pos, y_pos] = (0, 0, 0)

    for y_pos in range(HEIGHT):
        for x_pos in range(int(HEIGHT // 3) - int(HEIGHT // 64), int(HEIGHT // 3) + int(HEIGHT // 64)):
            pix[x_pos, y_pos] = (0, 0, 0)

    for y_pos in range(HEIGHT):
        for x_pos in range(int(2 * HEIGHT // 3) - int(HEIGHT // 64), int(2 * HEIGHT // 3) + int(HEIGHT // 64)):
            pix[x_pos, y_pos] = (0, 0, 0)
    for i in range(3):
        for j in range(3):
            if board_arr[i][j] == 1:
                img.paste(cross, (WIDTH * j // 3 + HEIGHT //
                                  25, WIDTH // 25 + WIDTH * i // 3))
            elif board_arr[i][j] == 2:
                img.paste(circ, (WIDTH * j // 3 + HEIGHT //
                                 25, WIDTH // 25 + WIDTH * i // 3))
            elif not board_arr[i][j] == 0:
                raise ValueError(
                    'Board [{}][{}] = {}'.format(i, j, board_arr[i][j]))
    img.save(img_name + ".jpg")


if __name__ == '__main__':
    to_image([[2, 0, 1], [0, 0, 1], [2, 1, 0]], "test")
