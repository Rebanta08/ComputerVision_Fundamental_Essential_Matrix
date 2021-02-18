import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import pyAprilTag
def Norm(Points):
	Centroid = np.mean(Points, axis = 0)
	Points = Points - Centroid
	Points = np.square(Points) 
	Points = np.sum(Points, axis= 1)
	Points = np.sqrt(Points)
	mean = np.mean(Points)
	Norm_Matrix = np.array([[math.sqrt(2)/mean, 0 , -(math.sqrt(2)/mean)*Centroid[0]], [0 , math.sqrt(2)/mean, -(math.sqrt(2)/mean)*Centroid[1]], [0, 0, 1]])
	return Norm_Matrix
def Essential_Matrix(k1,f,k2):
    return np.matmul(k1.T,np.matmul(f,k2))
def R_and_T(E,K):
    u, s, v = np.linalg.svd(E)
    dia = np.array([[1,0,0],[0,1,0],[0,0,0]])
    newE = np.matmul(u,np.matmul(dia,np.transpose(v)))
    u,s,v = np.linalg.svd(newE)
    w = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u3 = np.dot(u,np.transpose(np.array([0,0,1])))
    P1 = np.append(np.matmul(u,np.matmul(w,np.transpose(v))),np.array([u3]).T,axis = 1)
    P1 = P1*(np.linalg.det(P1[0:3,0:3]))
    P1 = np.matmul(K, P1)
    P2 = np.append(np.matmul(u,np.matmul(w,np.transpose(v))),-np.array([u3]).T,axis = 1)
    P2 = P2*(np.linalg.det(P2[0:3,0:3]))
    P2 = np.matmul(K, P2)
    P3 = np.append(np.matmul(u,np.matmul(np.transpose(w),np.transpose(v))),np.array([u3]).T,axis = 1)
    P3 = P3*(np.linalg.det(P3[0:3,0:3]))
    P3 = np.matmul(K, P3)
    P4 = np.append(np.matmul(u,np.matmul(np.transpose(w),np.transpose(v))),-np.array([u3]).T,axis = 1)
    P4 = P4*(np.linalg.det(P4[0:3,0:3]))
    P4 = np.matmul(K, P4)
    return P1, P2, P3, P4
def F_Matrix(pts_1,pts_2):
	[n1, C_1] = np.shape(pts_1)
	[n2, C_2] = np.shape(pts_2)
	if (C_1 != 2 or C_2!= 2):
		print("Error not correct")
	if (n1<8 or n2<8):
		print("There are not enough points")
	pts_1 = np.append(pts_1,np.ones((np.shape(pts_1)[0],1)),axis=1)   
	pts_2 = np.append(pts_2,np.ones((np.shape(pts_2)[0],1)),axis=1)   
	N1 = Norm(pts_1)
	N2 = Norm(pts_2)
	pts_1 = (np.matmul(N1,pts_1.T)).T
	pts_2 = (np.matmul(N2,pts_2.T)).T
	X_1 = pts_1[:,0]
	Y_1 = pts_1[:,1]
	X_2 = pts_2[:,0]
	Y_2 = pts_2[:,1]	
	C_1 = np.multiply(X_2,X_1)
	C_2 = np.multiply(X_2,Y_1)
	C_3 = X_2
	C_4 = np.multiply(Y_2,X_1)
	C_5 = np.multiply(Y_2,Y_1)
	C_6 = Y_2
	C_7 = X_1
	C_8 = Y_1
	C_9 = np.ones((np.shape(X_1)[0]))	
	A_Matrix = np.array([C_1, C_2, C_3, C_4, C_5, C_6, C_7, C_8, C_9]).T
	u, s , v = np.linalg.svd(A_Matrix)
	F_Matrix_Col = v[:,-1]
	F_m = np.array([[F_Matrix_Col[0],F_Matrix_Col[1],F_Matrix_Col[2]], [F_Matrix_Col[3],F_Matrix_Col[4],F_Matrix_Col[5]], [F_Matrix_Col[6],F_Matrix_Col[7],F_Matrix_Col[8]]])
	u, s, v = np.linalg.svd(F_m)
	s[2] = 0
	F_m = np.dot(u,np.dot(np.diag(s),v.T))
	F_m = np.matmul(N1.T,np.matmul(F_m,N2))
	F_m = F_m/F_m[2,2]
	return F_m
def Add_one(pts):
    pts = np.append(pts,np.ones((np.shape(pts)[0],1)),axis=1)
    return pts.T
def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])
def Triang(x1,x2,P,P1):
	d3_cord = np.empty([4,np.shape(x1)[1]])
	for i in range(np.shape(x1)[1]):
		pt1 = x1[:,i]
		pt2 = x2[:,i]
		pt1 = skew(pt1)
		pt2 = skew(pt2)
		pt1 = np.matmul(pt1,P)
		pt2 = np.matmul(pt2,P1)
		pp = np.append(pt1,pt2,axis= 0)
		u, s, v = np.linalg.svd(pp)
		xxx = v[:,-1]
		xxx = xxx/xxx[-1]
		d3_cord[:,i] = xxx
	return(d3_cord)
def Score(x1,x2):
		x1 = x1[2,:]
		x2 = x2[2,:]
		score = 0
		for i in range(np.shape(x1)[0]):
			if x1[i] > 0 and x2[i]>0:
				score = score +1
		return score
if __name__ == "__main__":
	img1 = cv.imread('AprilCalib_orgframe_00007.png',0)
	img2 = cv.imread('AprilCalib_orgframe_00008.png',0) 
	ids1, corners1, centers1, Hs1 = pyAprilTag.find(img1)
	ids2, corners2, centers2, Hs2 = pyAprilTag.find(img2)
	points2 = np.empty([40,4,2])
	C_2 = np.empty([40,2])
	ids1 = list(ids1)
	ids2 = list(ids2)
	for i in range(len(ids1)):
	    idx_curr_2 = ids2.index(ids1[i])
	    points2[i] = corners2[idx_curr_2]

	    C_2[i,:] = centers2[idx_curr_2]
	points2_new = np.empty([200,2])
	for i in range(np.shape(points2)[0]):
		points2_new[5*i,:] = points2[i][0]
		points2_new[5*i+1,:] = points2[i][1]
		points2_new[5*i+2,:] = points2[i][2]
		points2_new[5*i+3,:] = points2[i][3]
		points2_new[5*i+4,:] = C_2[i,:]
	points1_new = np.empty([200,2])
	for i in range(np.shape(corners1)[0]):	
		points1_new[5*i,:] = corners1[i][0]
		points1_new[5*i+1,:] = corners1[i][1]
		points1_new[5*i+2,:] = corners1[i][2]
		points1_new[5*i+3,:] = corners1[i][3]
		points1_new[5*i+4,:] = centers1[i,:]
	F, mask = cv.findFundamentalMat(points1_new,points2_new,cv.FM_LMEDS)
	K=np.array([[611.4138483174528, 0, 315.6207318252974],
       [0, 611.9537184774845, 259.6803148373927],
       [0, 0, 1]], dtype='float64');
	E = Essential_Matrix(K,F,K)                   
	print("\n E matrix:",E)
	P1, P2, P3, P4 = R_and_T(E,K)           
	p_list = [P1, P2, P3, P4]
	P = np.matmul(K,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
	points1_new = Add_one(points1_new)
	points2_new = Add_one(points2_new)
	d1 = Triang(points1_new,points2_new,P,P1)
	d2 = Triang(points1_new,points2_new,P,P2)
	d3 = Triang(points1_new,points2_new,P,P3)
	d4 = Triang(points1_new,points2_new,P,P4)
	x1_1_c = np.matmul(P,d1)
	x2_1_c = np.matmul(P1,d1)
	x1_2_c = np.matmul(P,d2)
	x2_2_c = np.matmul(P2,d2)
	x1_3_c = np.matmul(P,d3)
	x2_3_c = np.matmul(P3,d3)
	x1_4_c = np.matmul(P,d4)
	x2_4_c = np.matmul(P4,d4)
	score1 = Score(x1_1_c,x2_1_c)
	score2 = Score(x1_2_c,x2_2_c)
	score3 = Score(x1_3_c,x2_3_c)
	score4 = Score(x1_4_c,x2_4_c)
	l = [score1,score2,score2,score4]
	ind = l.index(max(l))
	print(" \n The final answer is:")
	H=np.matmul(np.linalg.inv(K),p_list[ind])
	print(H)
	rotation_vec, _ = cv.Rodrigues(H[0:3,0:3])
	print(" \nThe Rotation Vector is:")
	print(rotation_vec)
	Translation_vec=np.reshape(H[0:3,3],(3,1))
	print(" \n The Translation Vector is:")
	print(Translation_vec)
