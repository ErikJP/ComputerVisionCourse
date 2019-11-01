import matplotlib.pyplot as plt, numpy as np, cv2, io
from picamera import PiCamera
from time import sleep
from scipy import signal
import time
from picamera.array import PiRGBArray

ref = cv2.imread('ref.png') # Reference image
transformed = cv2.imread('tranformed.png') # Translated/Rotated image
chess_ref = np.copy(ref)
chess_transformed = np.copy(transformed)

# Getting and drawing interior corners
ret1,corners1 = cv2.findChessboardCorners(chess_ref,(7,7),None)
cv2.drawChessboardCorners(chess_ref,(7,7),corners1,ret1)
ret2,corners2 = cv2.findChessboardCorners(chess_transformed,(7,7),None)
cv2.drawChessboardCorners(chess_transformed,(7,7),corners2,ret2)

# Initializing arrays so I can re-format the corners arrays
corners1_adj = np.zeros([49,2])
corners2_adj = np.zeros([49,2])

# Looping over the corners arrays and normalizing them so that
#   they both start and end in the same corners
for i in range(0,49):
    corners2_adj[i,0] = corners2[i][0][0]
    corners2_adj[i,1] = corners2[i][0][1]

for i in range(0,7):
    for j in range(0,7):
        corners1_adj[7*i+j,0] = corners1[7*(j+1)-(i+1)][0][0]
        corners1_adj[7*i+j,1] = corners1[7*(j+1)-(i+1)][0][1]

# Initialize A and b "matrices"
A = np.zeros([98,6])
b = np.zeros(98)

# Populate first 49 rows with values corresponding to x_prime
for i in range(0,49):
    A[i,0] = corners1_adj[i,0] # x in 0th column
    A[i,1] = corners1_adj[i,1] # y in 1st column
    A[i,2] = 1 # 1 in 2nd column
    b[i] = corners2_adj[i,0] # x_prime

# Populate last 49 rows with values corresponding to y_prime
for i in range(49,98):
    A[i,3] = corners1_adj[i-49,0] # x in 3rd column
    A[i,4] = corners1_adj[i-49,1] # y in 4th column
    A[i,5] = 1 # 1 in 5th column
    b[i] = corners2_adj[i-49,1] # y_prime

#Solve for a11...a23
a11,a12,a13,a21,a22,a23 = np.linalg.pinv(A).dot(b)

# Print to view results
print a11
print a12
print a13
print a21
print a22
print a23        

# Show and save images
cv2.imshow('Ref',chess_ref)
cv2.imshow('Transformed',chess_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
