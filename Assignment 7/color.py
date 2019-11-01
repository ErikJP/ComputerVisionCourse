import matplotlib.pyplot as plt, numpy as np, cv2, colorsys

people2 = cv2.imread('People2.jpg')
r,c,d = np.shape(people2)
new_people = np.uint8(np.zeros([r,c]))
skin = cv2.imread('SkinSamples.JPG')

R = np.float64(skin[:,:,0])
G = np.float64(skin[:,:,1])
B = np.float64(skin[:,:,2])

hsv = cv2.cvtColor(skin,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

hist = np.uint8(hist*255/np.amax(hist))

hsv_people2 = cv2.cvtColor(people2,cv2.COLOR_BGR2HSV)
h_people = np.uint8(hsv_people2[:,:,0])
s_people = np.uint8(hsv_people2[:,:,1])
for i in range(0,r):
    for j in range(0,c):
        new_people[i,j] = hist[h_people[i,j],s_people[i,j]]

backProj = cv2.calcBackProject([hsv_people2],[0,1],np.float32(hist), [0, 180, 0, 256],1.0)

cv2.namedWindow('hsv hist', cv2.WND_PROP_FULLSCREEN)
cv2.imshow('hsv hist',hist)
cv2.imshow('New Pic',new_people)
cv2.imshow('backProj',backProj)
cv2.imshow('People',people2)
cv2.imshow('Skin',skin)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('skin_hsv_hist.png',hist)
cv2.imwrite('people_2.png',people2)
cv2.imwrite('my_people_backproj.png',new_people)
cv2.imwrite('BI_people_backproj.png',backProj)
