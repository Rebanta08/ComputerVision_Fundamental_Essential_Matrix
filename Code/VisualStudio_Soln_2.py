import numpy as np
import cv2 as cv
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
img1 = cv.imread('AprilCalib_orgframe_00007.png',0)  #queryimage # left image
img2 = cv.imread('AprilCalib_orgframe_00008.png',0) #trainimage # right image
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
me1= np.mean(pts1,axis=0)
me2 = np.mean(pts2,axis=0)
pts1new = pts1-me1
pts2new = pts2-me2
a1x = np.sqrt(2/np.std(pts1new[:,0]))
a1y = np.sqrt(2/np.std(pts1new[:,1]))
a2x = np.sqrt(2/np.std(pts2new[:,0]))
a2y = np.sqrt(2/np.std(pts2new[:,1]))
t1 = np.array([[a1x,0,-a1x*me1[0]],[0,a1y,-a1y*me1[1]],[0,0,1]])
t2 = np.array([[a2x,0,-a2x*me2[0]],[0,a2y,-a2y*me2[1]],[0,0,1]])
ones = np.ones(pts1.shape[0])[:,np.newaxis]
old_c1 = np.hstack((pts1,ones)).T
old_c2 = np.hstack((pts2,ones)).T
new_c1 = np.dot(t1,old_c1).T
new_c2 = np.dot(t2,old_c2).T
new_c1 = new_c1[:,:2]
new_c2 = new_c2[:,:2]
a = new_c1
b = new_c2
A =[]
for i in range(8):
    A.append([b[i][0]*a[i][0],b[i][0]*a[i][1],b[i][0], b[i][1]*a[i][0], b[i][1]*a[i][1], b[i][1], a[i][0], a[i][1], 1])
u, s, v = np.linalg.svd(A, full_matrices=True)
F_hat = v[:,8].reshape(3,3)
u2, s2, v2 = np.linalg.svd(F_hat, full_matrices=True)
s2[2]= 0
f = np.dot(u2,np.dot(np.diag(s2),v2))
T2t = np.transpose(t2)
Tt1 = t1
F = np.matmul(T2t,f,Tt1)
F=np.true_divide(F,F[2,2])

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
lines1 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()