# Nomaan Khan
# Program 4
import cv2
import numpy as np
import sys
import math

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

# w1 = 0.2#float(sys.argv[1])
# h1 = 0.1#float(sys.argv[2])
# w2 = 0.8#float(sys.argv[3])
# h2 = 0.5#float(sys.argv[4])
# name_input = 'fruits.jpg'#sys.argv[5]
# name_output = 'f2.jpg'#sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))


img_luv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LUV)
img_hsclipped = img_luv.copy()
img_luv[:,:,0] = cv2.equalizeHist(img_luv[:,:,0])
for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        img_hsclipped[i,j] = img_luv[i,j]
        
img_output = cv2.cvtColor(img_hsclipped, cv2.COLOR_LUV2BGR)

cv2.imshow('Color input image', inputImage)
cv2.imshow('Histogram equalized', img_output)
cv2.imwrite(name_output, img_output)

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()