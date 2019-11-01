import numpy as np; import cv2

# Import image

img_grey = cv2.imread('Church.jpg',0)

# Blur image

# First with 3x3 chunks

size = 9
g = 1./size
rows,cols = np.shape(img_grey)
new_img = np.copy(img_grey)
new_pixel_sum = 0

for i in range(1,rows-1):
    for j in range (1,cols-1):
        new_pixel_sum = 0
        for k in range(1,4):
            for l in range(1,4):
                new_pixel_sum += g*img_grey[i-k+2,j-l+2]
        new_img[i,j] = new_pixel_sum

# Get blurrier image with 5x5 chunks

size2 = 25
g2 = 1./size2
new_img2 = np.copy(new_img)
for i in range(2,rows-2):
    for j in range (2,cols-2):
        new_pixel_sum = 0
        for k in range(1,6):
            for l in range(1,6):
                new_pixel_sum += g2*img_grey[i-k+3,j-l+3]
        new_img2[i,j] = new_pixel_sum

# Show/save images

cv2.imshow('Original Image',img_grey)
cv2.imshow('Blurred Image',new_img)
cv2.imshow('Blurred Image 2',new_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('OriginalImg.png',img_grey)
cv2.imwrite('BlurredImg.png',new_img)
cv2.imwrite('BlurredImg2.png',new_img2)
