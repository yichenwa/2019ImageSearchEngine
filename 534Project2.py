import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2
# author Yichen Wang
sift=cv2.xfeatures2d.SIFT_create()
l=[]

"""
1. import sample image
This part is import the sample image,get its SIFT
"""
img1 = cv2.imread("/Users/saber/Desktop/02.jpg",0)
kp1, des1 = sift.detectAndCompute(img1,None)


"""
2. go through all images in the folder
We need to go through all images in the folder once. 
I set DATADIR to be the path to the folder "paris"
And then convert the RGB images into GRAY as the GRAY has smaller size
"""
DATADIR="/Users/saber/Desktop/534A2/paris"
CATEGORIES=["invalides"]

for i in range(0,len(CATEGORIES)):
    category=CATEGORIES[i]
    path = os.path.join(DATADIR,category)
    #for imgindex in os.listdir(path):
    for j in range(0,len(os.listdir(path))):
        imgindex=(os.listdir(path))[j]
        img2=cv2.imread(os.path.join(path,imgindex),0)

        """
        3. image representation
        I decide to use local features, SIFT
        """
        kp2, des2 = sift.detectAndCompute(img2, None)

        """
        4. matching
        Then use Brute-Force Matching to get k best matches, I set k=2.
        """
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        """
        5. ranking
        j is the image's index in the folder
        """
        l.append([len(good),j])

        for n in good:
            idx=n[0].trainIdx
            (x,y)=kp2[idx].pt
            print(x,y)


        #print("good=",len(good))
        #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,flags=2)
        #cv2.imshow("sift_image", img3)
        #cv2.waitKey(6000)
        break
    break

"""
5. ranking
    5-1 sort the list l
    5-2 select first 10 elements in l
"""
l.sort(reverse = True)
newl=[]
#for m in range (0,10):
    #newl.append(l[m])



cv2.destroyAllWindows()


