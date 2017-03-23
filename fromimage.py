import cv2
import numpy as np
import math

crosspath = './shapes/cross.png'

def fromImage(filename):
	ret = [[' ',' ',' '],['',' ',' '],[' ',' ',' ']]
	img = cv2.imread(filename)
	assert img is not None, "Problem reading image"
	height,width = img.shape[:2]
	#Divide board into 9 pics and search each individual pic
	# How to find the squares:
	# - search for horizontal and vertical lines
	#   that span a big portion of the image, and find their crossings
	# - divide image in 3x3 images
	squares = [[None for i in range(3)] for j in range(3)]
	for i in range(3):
		for j in range(3):
			squares[i][j] = img[i*height/3:(i+1)*height/3,j*height/3:(j+1)*height/3]
			#squares[i][j] = img[i*height/6:i*height/6 + 2*height/3,j*width/6:j*width/6+2*width/3]

	ret = [map(detectSquare,square) for square in squares]
	return ret

def detectSquare(im_square):
	height,width = im_square.shape[:2]
	gray = cv2.cvtColor(im_square, cv2.COLOR_BGR2GRAY)


	#Check for circle using hough transform
	blur = cv2.GaussianBlur( gray, (9, 9), 2);
	circles = cv2.HoughCircles(blur, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
	if circles != None:
		for circ in circles:
			#Check that the radius is of a certain size
			radius = circ[0][2]
			if radius > 0.2*(height+width)/2:
				return 'o'

	#Check for cross
	#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html#fast
	#http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	#http://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
	im_cross = cv2.imread(crosspath)
	#Scale cross to smaller size than square
	im_cross = cv2.resize(im_cross,(int(0.9*width),int(0.9*height)))
	im_cross = cv2.cvtColor(im_cross,cv2.COLOR_BGR2GRAY)
	edges_cross = cv2.Canny(im_cross,100,200)
	edges_square = cv2.Canny(im_square,100,200)
	method = cv2.TM_CCOEFF_NORMED
	res = cv2.matchTemplate(gray,im_cross,method)
	minval,maxval,minpos,maxpos = cv2.minMaxLoc(res)
	if maxval > 0.5:
		return 'x'
	return ' '

if __name__ == '__main__':
	board = fromImage("board.png")
	print board

