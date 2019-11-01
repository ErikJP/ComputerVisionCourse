import numpy as np; import cv2

img = cv2.imread('TrinityRegentHouse.jpg',0) 
r,c = img.shape
new_img = np.uint8(np.zeros([r/4,c/4]))
sample_img = np.uint8(np.zeros([r/4,c/4]))
avg4_img = np.uint8(np.zeros([r/4,c/4]))
zoom_naive = np.uint8(np.zeros([r,c]))

new_pixel = 0


for i in range(0,r,4):
    for j in range(0,c,4):
        new_pixel = 0
        sample_img[i/4,j/4] = img[i+1,j+1]
        for k in range(0,4):
            for l in range(0,4):
                new_pixel += img[i+k,j+l]
        new_img[i/4,j/4] = new_pixel/16

for ii in range(0,r,4):
    for jj in range(0,c,4):
        new_pixel = 0
        for kk in range(0,2):
            for ll in range(0,2):
                new_pixel += img[ii+kk+1,jj+ll+1]
        avg4_img[ii/4,jj/4] = new_pixel/4

resize = cv2.resize(img,(c/4,r/4))
for a in range(0,r,4):
    for b in range(0,c,4):
        new_pixel = resize[(a+1)/4,(b+1)/4]
        for aa in range(a,a+4):
            for bb in range(b,b+4):
                zoom_naive[aa,bb] = new_pixel

cv2.imshow('image',img)
cv2.imshow('averaged (16) image',new_img)
cv2.imshow('averaged (4) image',avg4_img)
cv2.imshow('sampled image',sample_img)
cv2.imshow('resize image',resize)
cv2.imshow('zoomed image',zoom_naive)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('orig.png',img)
cv2.imwrite('16avg.png',new_img)
cv2.imwrite('4avg.png',avg4_img)
cv2.imwrite('sample.png',sample_img)
cv2.imwrite('resize.png',resize)
cv2.imwrite('zoomed.png',zoom_naive)
