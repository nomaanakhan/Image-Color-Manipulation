# Nomaan Khan
# Program 3
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

# cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

def inverseGammaCorrection(D):
    if (D < 0.03928):
        return D/12.92
    else:
        return ((D + 0.055)/1.055)**2.4

def gammaCorrection(D):
    if (D < 0.00304):
        return 12.92*D
    else:
        return 1.055*((D**(1/2.4))) - 0.055

def round(D):
    if(D < 0):
        return 0
    elif(D > 1):
        return 1
    else:
        return D
    
# To LUV conversions
LuvMatrix = np.zeros([rows, cols, bands], dtype=np.float16)
uw = 4*0.95/(0.95 + 15 + 3*1.09)
vw = 9/(0.95 + 15 + 3*1.09)
for i in range(H1, H2+1):
    for j in range(W1, W2+1):
        b, g, r = inputImage[i, j]

        nonLinB = b/255.0
        nonLinG = g/255.0
        nonLinR = r/255.0

        linB = inverseGammaCorrection(nonLinB)
        linG = inverseGammaCorrection(nonLinG)
        linR = inverseGammaCorrection(nonLinR)
        
        X = 0.412453*linR + 0.357580*linG + 0.180423*linB
        Y = 0.212671*linR + 0.715160*linG + 0.072169*linB
        Z = 0.019334*linR + 0.119193*linG + 0.950227*linB

        if Y > 0.008856:
            L = 116*(Y**(1/3.0)) - 16.0 
        else:
            L = 903.3*Y

        d = X + 15*Y + 3*Z
        if(d <= 0):
            d = 0.1
 
        uprime = (4*X)/d
        u = 13*L*(uprime - uw)
        
        vprime = (9*Y)/d
        v = 13*L*(vprime - vw)

        LuvMatrix[i,j] = [L, u, v]

# Finding min and max L
maxL = 0
minL = 100
for i in range(H1, H2+1):
    for j in range(W1, W2+1):
        L, u, v = LuvMatrix[i, j]
        if(L >= maxL):
            maxL = L
        if(L <= minL):
            minL = L   

# h(i)
h = np.zeros(101)
for i in range(H1, H2+1):
    for j in range(W1, W2+1):
        L, u, v = LuvMatrix[i, j]
        if(L > maxL):
            L = 100
        elif (L < minL):
            L = 0
        else:
            L = math.floor(L)
        LuvMatrix[i,j] = [L, u, v]   
        h[L] += + 1
 
fs = np.cumsum(h) # Frequency sum
fi1 = np.zeros(101) # (f(i) + f(i-1))/2
for i in range(0, 101):
    if (i == 0):
        fi1[i] = math.floor(fs[i]*101 / (2*fs[100]))
    else:
        fi1[i] = math.floor((fs[i - 1] + fs[i])*101 / (2*fs[100]))


HSOutput = inputImage.copy()
# Histogram equalization and conversions
for i in range(H1, H2+1):
    for j in range(W1, W2+1):
        # Histogram equalization in Luv Domain
        L, u, v = LuvMatrix[i, j]
        L = fi1[int(L)]

        if(L == 0):
            uprime = 0
            vprime = 0
        else:
            uprime = (u + 13*uw*L)/(13*L)
            vprime = (v + 13*vw*L)/(13*L)

        if(L > 7.9996):
            Y = np.power((L + 16)/116, 3)
        else:
            Y = L/903.3
        
        if(vprime == 0):
            X = 0
            Z = 0
        else:
            X = Y*2.25*(uprime/vprime)
            Z = (Y*(3 - 0.75*uprime - 5*vprime))/vprime

        R = (3.240479 * X) - (1.53715 * Y) - (0.498535 * Z)
        G = (-0.969256 * X) + (1.875991 * Y) + (0.041556* Z)
        B = (0.055648 * X) - (0.204043* Y) + (1.057311 * Z)
 
        nonLinR = round(gammaCorrection(R))
        nonLinG = round(gammaCorrection(G))
        nonLinB = round(gammaCorrection(B))

        r = np.rint(nonLinR*255)
        g = np.rint(nonLinG*255)
        b = np.rint(nonLinB*255)
      
        HSOutput[i,j] = [b, g, r]

cv2.imshow('Input Image', inputImage)
cv2.imshow('Histogram equalization Output', HSOutput)
cv2.imwrite(name_output, HSOutput)    

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()