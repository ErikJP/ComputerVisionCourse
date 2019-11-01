import matplotlib.pyplot as plt, numpy as np, cv2, io
from picamera import PiCamera
from time import sleep
from scipy import signal
import time
from picamera.array import PiRGBArray

camera = PiCamera()
get_img = io.BytesIO()

camera.start_preview()
sleep(3)
camera.capture(get_img,format='jpeg')
camera.stop_preview()

data = np.fromstring(get_img.getvalue(),dtype=np.uint8)
image_cap = cv2.imdecode(data,1)

image_cap = cv2.cvtColor(image_cap,cv2.COLOR_BGR2GRAY)

r,c = np.shape(image_cap)

# METHOD 1
#   - Subtract blurred image from original and medianBlur twice to reduce salt/pepper
start = time.time()
blur = cv2.blur(image_cap,(5,5))
derivative = image_cap - blur
derivative = cv2.medianBlur(derivative,7)
derivative = cv2.medianBlur(derivative,7)
derivative = 255-derivative
print time.time() - start # takes one second and has a lot of noise

# METHOD 2
#   - Creat threshold and then subtract blur from the original
start0 = time.time()
thresh = np.copy(image_cap)
t = 80 # Set a threshold to accentuate the edges
thresh[np.where(image_cap>t)] = 255
thresh[np.where(image_cap<=t)] = 0

blur_t = cv2.blur(thresh,(5,5))
thresh_new = thresh-blur_t
thresh_new = 255-thresh_new
print time.time() - start0 # takes one second but is slightly inaccurate

# METHOD 3
#   - Use sobel operation and convolve2d
def sobel_convolve2d(img):
    s = np.array([[1-1j,0-2j,-1-1j],[2-0j,0-0j,-2-0j],[1+1j,0+2j,-1+1j]])
    grad = signal.convolve2d(img,s,boundary='symm',mode='same')
    # Adjust type
    grad = np.float32(abs(grad))
    grad = grad*255/(np.amax(grad))
    grad = np.uint8(grad)
    return grad

start1 = time.time()
grad1 = 255 - sobel_convolve2d(image_cap)
print time.time() - start1 # Slowest at about 2 seconds but the most accurate in all orientations

# Method 4
#   - Sobel with filter2D
def sobel_filter2D(img):
    s = np.array([[2,1,0],[1,0,-1],[0,-1,-2]]) # Using a diagonal sobel kernel
    grad = cv2.filter2D(img,cv2.cv.CV_32F,s)
    # Adjust type
    grad = abs(grad)
    grad = grad*255/(np.amax(grad))
    grad = np.uint8(grad)
    return grad

start2 = time.time()
grad2 = 255 - sobel_filter2D(image_cap)
print time.time() - start2 # Fastest at about .23 seconds but innacurate at certain angles

# Setting up camera for video
camera.resolution = (640,480)
camera.framerate = 30
rawCapture = PiRGBArray(camera,size=(640,480))

time.sleep(0.1)

# Take video implementing the sobel_filter2D() function
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',sobel_filter2D(gray)) # This was working at about 9 frames per second
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
	break

# Save images (omitted from writeup)
cv2.imshow('image cap',image_cap)
cv2.imshow('deriv',derivative)
cv2.imshow('thresh',thresh)
cv2.imshow('thresh new',thresh_new)
cv2.imshow('grad1',grad1)
cv2.imshow('grad2',grad2)

cv2.imwrite('image_cap.png',image_cap)
cv2.imwrite('deriv.png',derivative)
cv2.imwrite('thresh.png',thresh)
cv2.imwrite('thresh_new.png',thresh_new)
cv2.imwrite('grad1.png',grad1)
cv2.imwrite('grad2.png',grad2)

cv2.waitKey(0)
cv2.destroyAllWindows
