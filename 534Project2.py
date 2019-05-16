import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2
# author Yichen Wang
MIN_MATCH_COUNT = 100
sift=cv2.xfeatures2d.SIFT_create()

"""
This part is import the sample image,get its SIFT
"""
img1 = cv2.imread("/Users/saber/Desktop/01.jpg",0)
kp1, des1 = sift.detectAndCompute(img1,None)


"""
We need to go through all images in the folder once. 
I set DATADIR to be the path to the folder "paris"
And then convert the RGB images into GRAY as the GRAY has smaller size
"""
DATADIR="/Users/saber/Desktop/534A2/paris"
CATEGORIES=["eiffel"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for imgindex in os.listdir(path):
        img2=cv2.imread(os.path.join(path,imgindex),0)

        """
        I decide to use local features, SIFT
        """
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,flags=2)

        #plt.imshow(img3, 'gray'), plt.show()
        cv2.imshow("sift_image", img3)
        cv2.waitKey(600)

        #break
    break

cv2.destroyAllWindows()


