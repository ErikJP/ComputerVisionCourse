import numpy as np; import cv2
# IMport image and get rows and columns count
img = cv2.imread('TrinityRegentHouse.jpg',0)
r,c = img.shape
# Initializing place holders
img1bit = np.copy(img)
img2bit = np.copy(img)
img4bit = np.copy(img)
bit0img = np.uint8(np.zeros([r,c]))
bit1img = np.uint8(np.zeros([r,c]))
bit2img = np.uint8(np.zeros([r,c]))
bit3img = np.uint8(np.zeros([r,c]))
bit4img = np.uint8(np.zeros([r,c]))
bit5img = np.uint8(np.zeros([r,c]))
bit6img = np.uint8(np.zeros([r,c]))
bit7img = np.uint8(np.zeros([r,c]))
# Set the 1 bit array
img1bit[np.where(img1bit>=128)] = 255
img1bit[np.where(img1bit<128)] = 0
# Set the 4 bit array
for i in range(0,16):
    img4bit[np.where(img>(i*16))] = (i*16)
# Set the 2 bit array
for i in range(0,4):
    img2bit[np.where(img>(i*64))] = (i*64)
# Extract bit information using bit manipulation
for i in range(0,r):
    for j in range(0,c):
        if (img[i,j]&1):
            bit0img[i,j] = 255
        if ((img[i,j]>>1)&1):
            bit1img[i,j] = 255
        if ((img[i,j]>>2)&1):
            bit2img[i,j] = 255
        if ((img[i,j]>>3)&1):
            bit3img[i,j] = 255
        if ((img[i,j]>>4)&1):
            bit4img[i,j] = 255
        if ((img[i,j]>>5)&1):
            bit5img[i,j] = 255
        if ((img[i,j]>>6)&1):
            bit6img[i,j] = 255
        if ((img[i,j]>>7)&1):
            bit7img[i,j] = 255
# Show and save images
cv2.imshow('original image',img)
cv2.imshow('1 bit image',img1bit)
cv2.imshow('2 bit image',img2bit)
cv2.imshow('4 bit image',img4bit)
cv2.imshow('Zeroeth bit on',bit0img)
cv2.imshow('First bit on',bit1img)
cv2.imshow('Second bit on',bit2img)
cv2.imshow('Third bit on',bit3img)
cv2.imshow('Fourth bit on',bit4img)
cv2.imshow('Fifth bit on',bit5img)
cv2.imshow('Sixth bit on',bit6img)
cv2.imshow('Seventh bit on',bit7img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
cv2.imwrite('original_image.png',img)
cv2.imwrite('1_bit_image.png',img1bit)
cv2.imwrite('2_bit_image.png',img2bit)
cv2.imwrite('4_bit_image.png',img4bit)
cv2.imwrite('Zeroeth_bit_on.png',bit0img)
cv2.imwrite('First_bit_on.png',bit1img)
cv2.imwrite('Second_bit_on.png',bit2img)
cv2.imwrite('Third_bit_on.png',bit3img)
cv2.imwrite('Fourth_bit_on.png',bit4img)
cv2.imwrite('Fifth_bit_on.png',bit5img)
cv2.imwrite('Sixth_bit_on.png',bit6img)
cv2.imwrite('Seventh_bit_on.png',bit7img)
