import numpy as np; import cv2, random

orig = cv2.imread('ckt_board_saltpep_prob_pt05.tif',0)
r,c = np.shape(orig)
new_img = np.uint8(np.zeros([r,c]))
church = cv2.imread('Church.jpg',0)
church_sp = np.copy(church)
church_gauss = np.copy(church)
r_church,c_church = np.shape(church)
new_pixel = 0
# My implementation of median blur
for i in range(2,r-2): # NOTE: I ignored edge case
    for j in range (2,c-2):
        new_img[i,j] = np.median(orig[i-2:i+3,j-2:j+3]) # Grab median value

built_in_med = cv2.medianBlur(orig,5) # Using built-in function
# Creating Salt and Pepper Noise
sp_prob = .10 # Probabilty of salt or pepper
sp_array = np.random.rand(r_church,c_church) # Salt and pepper array
s_array = np.copy(sp_array) # Salt array
p_array = np.copy(sp_array) # Pepper array
s_array[np.where(sp_array<=sp_prob/2)]=255 # Modify salt array
s_array[np.where(sp_array>=sp_prob/2)]=0
p_array[np.where(sp_array>=(1-sp_prob/2))]=0 # Modify pepper array
p_array[np.where(sp_array<=(1-sp_prob/2))]=255
s_array = np.uint8(s_array) # convert to correct type
p_array = np.uint8(p_array)
# Bitwise operators to mask the image
church_sp = np.bitwise_or(church_sp,s_array) 
church_sp = np.bitwise_and(church_sp,p_array)
# Creating Gauss Noise
gauss = (np.random.randn(r_church,c_church)+5) # Gauss centered about 5
church_gauss = np.float64(church_gauss) 
church_gauss = np.multiply(church_gauss,gauss)/5 # adjusting image
church_gauss[np.where(church_gauss>255)]=255 # Dealing with overflow
church_gauss = np.uint8(church_gauss) # Correcting type
# Show/Save images
cv2.imshow('Orig',orig)
cv2.imshow('Median',new_img)
cv2.imshow('Built-in median',built_in_med)
cv2.imshow('Church',church)
cv2.imshow('Church Salt Pep',church_sp)
cv2.imshow('Church Guass',church_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('orig.png',orig)
cv2.imwrite('medBlur.png',new_img)
cv2.imwrite('medBI.png',built_in_med)
cv2.imwrite('church_bw.png',church)
cv2.imwrite('church_sp.png',church_sp)
cv2.imwrite('churc_guass.png',church_gauss)
