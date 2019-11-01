import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# initialize obj points
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Initialize arrays to hold obj points and img points
objpoints = []
imgpoints = [] 

# Read in 10 image names
images = glob.glob('*.jpg')
count = 0 # Used to see which (if any) images were not working
for fname in images:
    img = cv2.imread(fname)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    count += 1
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        # refine corners
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        imgpoints.append(corners)
        # print count and fname to see that an image worked
        print count
        print fname
        # Draw/display corners
        cv2.drawChessboardCorners(img, (7,7), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Use cv2.calibrateCamera to get properties
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# read in a new image to project coordinates onto
img = cv2.imread('img11.png')

# reset obj points
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Defining the coordinates
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

def draw_coordinates(img_src, corners, imgpts):
    img = np.copy(img_src)
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners_3D = cv2.findChessboardCorners(gray,(7,7),None)

# Go back through and refine corners and obtain img points
if ret==True:
    cv2.cornerSubPix(gray,corners_3D,(11,11),(-1,-1),criteria)

    rvecs1, tvecs1, inliers = cv2.solvePnPRansac(objp, corners_3D, mtx, dist)
    
    img_pts,jac = cv2.projectPoints(axis, rvecs1, tvecs1, mtx, dist)
    # Draw coordinates with image points specific to source image
    img_3D = draw_coordinates(img,corners_3D,img_pts)

# Show and save images
cv2.imshow('orig',img)
cv2.imshow('3D',img_3D)
cv2.imwrite('coordinates.png',img_3D)
cv2.waitKey(0)
cv2.destroyAllWindows()






