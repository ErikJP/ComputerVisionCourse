import matplotlib.pyplot as plt, numpy as np, cv2, io
from picamera import PiCamera
from time import sleep

# Function I found online that converts from rgb to grayscale
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# Read images
std50 = cv2.imread('gaussian_noise_mean_0_std_50_added.tif',0)
wire = cv2.imread('wirebond-mask.tif',0)
shade = cv2.imread('septagon_noisy_shaded.tif',0)
stationery = cv2.imread('Stationery.jpg',0)

# Set up pi camera to take image
camera = PiCamera()
get_img = io.BytesIO()

# Taking the image
camera.start_preview()
sleep(3)
camera.capture(get_img,format='jpeg')
camera.stop_preview()

# Converting the image so that it can be used with cv2
data = np.fromstring(get_img.getvalue(),dtype=np.uint8)
image_cap = cv2.imdecode(data,1)

blur = cv2.GaussianBlur(std50,(7,7),0) # blurring std50 image

# obtaining the histograms of the blurred image
blur_hist = cv2.calcHist([blur],[0],None,[256],[0,256])
prob_hist = blur_hist/np.sum(blur_hist)

# initializing variables for my threshold method
mu1 = 0
mu2 = 0
tol = 10
t0 = 128.
t1 = t0

# Looping until some tolerance to get a threshold
while tol >.0001:
    mu1 = np.sum(prob_hist[0:t0])
    mu2 = np.sum(prob_hist[t0:256])
    t1 = t0
    t0 = 255*(mu1+mu2)/2
    tol = abs(t1-t0)

t0 = np.uint8(t0)

# Setting the new thresholded image
new_std50 = np.copy(blur)
new_std50[np.where(blur>t0)] = 255
new_std50[np.where(blur<=t0)] = 0

# using built in thresholds
ret,binary_std50 = cv2.threshold(std50,127,255,cv2.THRESH_BINARY)
ret,binary_blur = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
ret,OTSU_std50 = cv2.threshold(std50,0,255,cv2.THRESH_OTSU)
ret,OTSU_blur = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)

# Otsu and adaptive for shaded
ret,shade_OTSU = cv2.threshold(shade,0,255,cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(shade,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,435,2)

# Erosion and dilation kernel
kernel = np.ones((5,5),np.uint8)
# Using built ins to erode and dilate
wire_erosion = cv2.erode(wire,kernel,1)
wire_dilation = cv2.dilate(wire,kernel,1)

# The following is a method for sharpening an image that I looked up online,
# it subtracts the average of a neighborhood from twice a pixel value
k = np.zeros((9,9),np.float32)
k[4,4] = 2.
f = np.ones((9,9),np.float32)/81.
k=k-f
sharp = cv2.filter2D(stationery,-1,k)

# Converting captured img to grayscale and uint8
image_cap = rgb2gray(image_cap)
image_cap = np.uint8(image_cap)
# Thresholding captured image using binary, Otsu, and adaptive
ret,cap_bin = cv2.threshold(image_cap,127,255,cv2.THRESH_BINARY)
ret,cap_OTSU = cv2.threshold(image_cap,0,255,cv2.THRESH_OTSU)
cap_adapt = cv2.adaptiveThreshold(image_cap,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,99,2)

# Saving images (omitted from code writeup to save space)
cv2.imwrite('std50.png',std50)
cv2.imwrite('shade.png',shade)
cv2.imwrite('blur.png',blur)
cv2.imwrite('image_cap.png',image_cap)
cv2.imwrite('prob_hist_post_blur.png',prob_hist)
cv2.imwrite('new_std50.png',new_std50)
cv2.imwrite('binary_std50.png',binary_std50)
cv2.imwrite('binary_blur.png',binary_blur)
cv2.imwrite('OTSU_std50.png',OTSU_std50)
cv2.imwrite('OTSU_blur.png',OTSU_blur)
cv2.imwrite('Shade_OTSU.png',shade_OTSU)
cv2.imwrite('adaptive.png',adaptive)
cv2.imwrite('wire.png',wire)
cv2.imwrite('wire_erosion.png',wire_erosion)
cv2.imwrite('wire_dilation.png',wire_dilation)
cv2.imwrite('stationery.png',stationery)
cv2.imwrite('sharp.png',sharp)
cv2.imwrite('cap_bin.png',cap_bin)
cv2.imwrite('cap_OTSU.png',cap_OTSU)
cv2.imwrite('img_cap.png',image_cap)
cv2.imwrite('cap_adapt.png',cap_adapt)
