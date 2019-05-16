import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2
# author Yichen Wang

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
        img=cv2.imread(os.path.join(path,imgindex))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        """
        I decide to use local features, SIFT
        """
        kp2, des2 = sift.detectAndCompute(gray, None)
        bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
        #Match descriptors
        matches = bf.match(des1,des2)
        #Sort them in order of their distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 100 matches.
        img3 = cv2.drawMatches(sgray, kp1, gray, kp2, matches[:100],None, flags=2)



        #kp=sift.detect(gray,None)
        #img=cv2.drawKeypoints(gray,kp,img)
        cv2.imshow("sift_image", img3)
        cv2.waitKey(600)

        #break
    break

cv2.destroyAllWindows()


