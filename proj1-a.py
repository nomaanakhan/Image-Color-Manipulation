# Nomaan Khan
# Program 1
import cv2
import numpy as np
import sys

def gammaCorrection(D):
    if D < 0.00304:
        return 12.92*D
    else:
        return 1.055*(D**(1/2.4)) - 0.055

def round(D):
    if(D < 0):
        return 0
    elif(D > 1):
        return 1
    else:
        return D

def convertLuv2BGR(L,u,v):
    #### Replace the color conversion code here ######
    # place holder code below should be deleted and replace with correct code

    uw = (4*0.95) / (0.95 + 15.0 + (3*1.09))
    vw = 9/(0.95 + 15.0 + (3*1.09))
    if(L == 0):
        uprime = 0
        vprime = 0
    else:
        uprime = (u + 13*uw*L)/(13*L)
        vprime = (v + 13*vw*L)/(13*L)

    if(L > 7.9996):
        Y = ((L + 16)/116)**3
    else:
        Y = L/903.3
    
    if(vprime == 0):
        X = 0
        Z = 0
    else:
        X = Y*2.25*(uprime/vprime)
        Z = (Y*(3 - 0.75*uprime - 5*vprime))/vprime

    linR = (3.240479 * X) - (1.53715 * Y) - (0.498535 * Z)
    linG = (-0.969256 * X) + (1.875991 * Y) + (0.041556* Z)
    linB = (0.055648 * X) - (0.204043* Y) + (1.057311 * Z)

    nonLinR = round(gammaCorrection(linR))
    nonLinG = round(gammaCorrection(linG))
    nonLinB = round(gammaCorrection(linB))

    r = nonLinR*255
    g = nonLinG*255
    b = nonLinB*255
    ##################################################
    return b,g,r

    
if(len(sys.argv) != 3) :
    print(sys.argv[0], ": takes 2 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: width height.")
    print("Example:", sys.argv[0], " 200 300")
    sys.exit()

# cols = 200#int(sys.argv[1])
# rows = 300#int(sys.argv[2])

cols = int(sys.argv[1])
rows = int(sys.argv[2])

image = np.zeros([rows, cols, 3], dtype='uint8') # Initialize the image with all 0
for i in range(0, rows):
    for j in range (0,cols):
        # 0≤L≤100 , −134≤u≤220, −140≤v≤122 
        L = 90      
        u = (354 * j / cols) - 134
        v = (262 * i / rows) - 140
        b,g,r = convertLuv2BGR(L,u,v)
        image[i,j]=np.array([b,g,r],dtype='uint8')

#image = image*255

cv2.imshow("Luv:", image)
cv2.imwrite('luvImg.jpg', image)
# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()