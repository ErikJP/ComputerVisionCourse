import matplotlib.pyplot as plt, numpy as np, cv2, io
from picamera import PiCamera
from time import sleep
from scipy import signal
import time
from picamera.array import PiRGBArray

camera = PiCamera()
get_img = io.BytesIO()
camera.resolution = (640,480)
camera.start_preview()
sleep(5)
camera.capture(get_img,format='jpeg')
camera.stop_preview()

data = np.fromstring(get_img.getvalue(),dtype=np.uint8)
image_cap = cv2.imdecode(data,1)

image_cap = cv2.cvtColor(image_cap,cv2.COLOR_BGR2GRAY)

r,c = np.shape(image_cap)

# METHOD 1
#   - Laplace with convolve2d
#   - Same as convolving for Sobel but with different kernel
def laplace_convolve2d(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    s = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    grad = signal.convolve2d(img,s,boundary='symm',mode='same')
    # Adjust type
    grad = np.float32(abs(grad))
    grad = grad*255/(np.amax(grad))
    grad = np.uint8(grad)
    return grad

start1 = time.time()
grad1 = laplace_convolve2d(image_cap)
print time.time() - start1 

# Method 2
#   - Laplace with filter2D
#   - Same as filter2D for Sobel but with different kernel
def laplace_filter2D(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    s = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    grad = cv2.filter2D(img,cv2.cv.CV_32F,s)
    # Adjust type
    grad = abs(grad)
    grad = grad*255/(np.amax(grad))
    grad = np.uint8(grad)
    return grad

start2 = time.time()
grad2 = laplace_filter2D(image_cap)
print time.time() - start2

# Method 3
#   - My hough transform

# Hough Fxn
def Hough(img):
    r,c = np.shape(img)
    canny = cv2.Canny(img,100,200) # Get canny
    x,y = np.where(canny == 255) # New array with only white indices
 
    max_len = np.ceil(np.sqrt(r*r+c*c)) # Diagonal length or the max length
    r = np.linspace(-max_len,max_len,max_len*2.) # r bins
    theta = np.deg2rad(np.linspace(-90.,90.,181)) # theta bins

    bins = np.zeros((2*max_len,181),dtype=np.uint32)
    # Set sin and cos for speed
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    length = len(x)
    for i in range(0,length):
        # set x and y for quicker access
        xi = x[i]
        yi = y[i]
        for j in range(0,181):
            # find r for given theta
            rho = round(xi*cos_theta[j] + yi*sin_theta[j]) + max_len
            bins[rho,j] += 1

    return bins,r,theta

bins,r,theta = Hough(image_cap)
sides = 8 # number of sides on my image focus
# initialize arrays
my_max = np.zeros(sides)
rho = np.copy(my_max)
t = np.copy(rho)
temp_bins = np.copy(bins)

for i in range(0,sides):
    my_max[i] = np.argmax(temp_bins)
    print my_max[i]
    print np.shape(temp_bins)
    rho[i] = r[my_max[i]/bins.shape[1]]
    t[i] = theta[my_max[i]%bins.shape[1]]
    # after finding the maximum I tried setting it to zero to find
    # the next maximum. This didn't necessarily capture all of the
    # lines that I wanted though.
    temp_bins[np.argmax(temp_bins,axis=0)] = 0

# Plotting a histogram
plt.plot(bins)
plt.show()

# Initialize arrays
m = np.zeros(sides)
b = np.zeros(sides)

# Solve for m and b with various r and theta
for i in range(0,sides):
    m[i] = -np.cos(t[i])/np.sin(t[i])
    b[i] = rho[i]/np.sin(t[i])

hough = np.copy(image_cap)
# Use m and b to put lines on the image
for i in range(0,sides):
    cv2.line(hough,(0,np.int32((0-b[i])/m[i])),(640,np.int32((640-b[i])/m[i])),0,2)

# Method 4
#   - Using built-ins

built_in = np.copy(image_cap)
canny = cv2.Canny(image_cap,100,200)
lines = cv2.HoughLines(canny,1,np.pi/180,200)
# cv2.HoughLines would only give me one line for some reason and sometimes didn't
# detect any lines at all and I couldn't figure out why
for rho1,theta1, in lines[0]:
    aa = np.cos(theta1)
    bb = np.sin(theta1)
    x0 = aa*rho1
    y0 = bb*rho1
    x1 = int(x0 + 1000*(-bb))
    y1 = int(y0 + 1000*(aa))
    x2 = int(x0 - 1000*(-bb))
    y2 = int(y0 - 1000*(aa))
    cv2.line(built_in,(x1,y1),(x2,y2),0,2)

# Setting up camera for video

camera.framerate = 30
rawCapture = PiRGBArray(camera,size=(640,480))

time.sleep(0.1)

# Take video so I can look at the performance of Canny
# It is the fastest and most accurate function I've used
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',cv2.Canny(gray,100,200))
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
	break

# Save images (omitted from writeup)
cv2.imshow('grad1',grad1)
cv2.imshow('grad2',grad2)
cv2.imshow('Canny',canny)
cv2.imshow('line',hough)
cv2.imshow('built in',built_in)

cv2.imwrite('grad_1.png',grad1)
cv2.imwrite('grad_2.png',grad2)
cv2.imwrite('Canny.png',canny)
cv2.imwrite('line.png',hough)
cv2.imwrite('built_in.png',built_in)

cv2.waitKey(0)
cv2.destroyAllWindows
