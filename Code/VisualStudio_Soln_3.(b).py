import numpy as np 
import cv2 as cv
import pyAprilTag
img1 = cv.imread('AprilCalib_orgframe_00007.png',0) 
img2 = cv.imread('AprilCalib_orgframe_00008.png',0) 
ids_1, corners_1, centers_1, Hs_1 = pyAprilTag.find(img1)
ids_2, corners_2, centers_2, Hs_2 = pyAprilTag.find(img2)
K=np.array([[611.4138483174528, 0, 315.6207318252974],[0, 611.9537184774845, 259.6803148373927],[0, 0, 1]], dtype='float64')
dist=np.array([[-0.4509210354054127,0.1895690122992456,0,0,0]], dtype='float64');
ids_2 = list(ids_2)
ids_1 = list(ids_1)
ids_2_index = ids_2.index(ids_1[0])
corners_1_R = corners_1[0]
Hs_1_R = Hs_1[0]
obj1_list = np.zeros((4,3))
corners_2_R = corners_2[ids_2_index]
Hs_2_R = Hs_2[ids_2_index]
obj2_list = np.zeros((4,3)) 
for i, j in enumerate(corners_1_R):
	j = np.append(j, np.ones(1))
	p0 = np.matmul(np.linalg.inv(Hs_1_R),j)
	p0 = p0/p0[2]
	p0[2] = 0
	obj1_list[i,:]= p0
ret_1,rvecs_1, tvecs_1 = cv.solvePnP(obj1_list, corners_1_R, K, dist)   
r_1 , _ = cv.Rodrigues(rvecs_1)
H1 = np.append(r_1, tvecs_1,axis = 1)
H1 = np.append(H1,np.array([[0,0,0,1]]),axis =0)
for i, j in enumerate(corners_2_R):
	j = np.append(j, np.ones(1))
	p0 = np.matmul(np.linalg.inv(Hs_2_R),j)
	p0 = p0/p0[2]
	p0[2] = 0
	obj2_list[i,:]= p0
ret_2,rvecs_2, tvecs_2 = cv.solvePnP(obj2_list, corners_2_R, K, dist)
r_2 , j= cv.Rodrigues(rvecs_2)
H2 = np.append(r_2, tvecs_2,axis = 1)
H2 = np.append(H2,np.array([[0,0,0,1]]),axis =0)
H = np.matmul(H1, np.linalg.inv(H2))
print("The final answer is:")
print(H)
print("The Rotation Vector is:")
rotation_vec, _ = cv.Rodrigues(H[0:3,0:3])
print(rotation_vec)
Translation_vec=np.reshape(H[0:3,3],(3,1))
print("The Translation Vector is:")
print(Translation_vec)