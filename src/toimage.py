from PIL import Image, ImageOps, ImageDraw

WIDTH = 512
HEIGHT = 512


def toImage(boardArr, imgname):
    img = Image.new("RGB", (WIDTH, HEIGHT))
    circ = Image.open('./shapes/circle.png').resize((WIDTH /
                                                     4, HEIGHT / 4), Image.ANTIALIAS)
    cross = Image.open('./shapes/cross.png').resize((WIDTH / 4, HEIGHT / 4))

    pix = img.load()
    for x_pos in range(WIDTH):
        for y_pos in range(HEIGHT):
            pix[x_pos, y_pos] = (255, 255, 255)

    for x_pos in range(WIDTH):
        for y_pos in range(int(HEIGHT / 3) - int(HEIGHT / 64), int(HEIGHT / 3) + int(HEIGHT / 64)):
            pix[x_pos, y_pos] = (0, 0, 0)

    for x_pos in range(WIDTH):
        for y_pos in range(int(2 * HEIGHT / 3) - int(HEIGHT / 64), int(2 * HEIGHT / 3) + int(HEIGHT / 64)):
            pix[x_pos, y_pos] = (0, 0, 0)

    for y_pos in range(HEIGHT):
        for x_pos in range(int(HEIGHT / 3) - int(HEIGHT / 64), int(HEIGHT / 3) + int(HEIGHT / 64)):
            pix[x_pos, y_pos] = (0, 0, 0)

    for y_pos in range(HEIGHT):
        for x_pos in range(int(2 * HEIGHT / 3) - int(HEIGHT / 64), int(2 * HEIGHT / 3) + int(HEIGHT / 64)):
            pix[x_pos, y_pos] = (0, 0, 0)
    for i in range(3):
        for j in range(3):
            if boardArr[i][j] == 'x_pos':
                img.paste(cross, (WIDTH * j / 3 + HEIGHT /
                                  25, WIDTH / 25 + WIDTH * i / 3))
            elif boardArr[i][j] == 'o':
                img.paste(circ, (WIDTH * j / 3 + HEIGHT /
                                 25, WIDTH / 25 + WIDTH * i / 3))
            elif not boardArr[i][j] == ' ':
                print('Error, boardArr[i][j] = ' + boardArr[i][j])
    img.save(imgname + ".png", "PNG")


if __name__ == '__main__':
    toImage([[' ', ' ', 'x'], [' ', ' ', 'x'], ['o', ' ', ' ']], "test")
