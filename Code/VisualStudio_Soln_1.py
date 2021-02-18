import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import pyAprilTag

def Norm(_points_):
	_Centroid_ = np.mean(_points_, axis = 0)
	_points_ = _points_ - _Centroid_
	_points_ = np.square(_points_) 
	_points_ = np.sum(_points_, axis= 1)
	_points_ = np.sqrt(_points_)
	Mean = np.mean(_points_)
	Norm_matrix = np.array([[math.sqrt(2)/Mean, 0 , -(math.sqrt(2)/Mean)*_Centroid_[0]], [0 , math.sqrt(2)/Mean, -(math.sqrt(2)/Mean)*_Centroid_[1]], [0, 0, 1]])
	return Norm_matrix

def F_matrix(Pts_1,Pts_2):
	[n_1, col_1] = np.shape(Pts_1)
	[n_2, col_2] = np.shape(Pts_2)
	if (col_1 != 2 or col_2!= 2):
		print("Error")
	if (n_1<8 or n_2<8):
		print("Not enough points")

	Pts_1 = np.append(Pts_1,np.ones((np.shape(Pts_1)[0],1)),axis=1)  
	Pts_2 = np.append(Pts_2,np.ones((np.shape(Pts_2)[0],1)),axis=1)  
	N_1 = Norm(Pts_1)
	N_2 = Norm(Pts_2)
	Pts_1 = (np.matmul(N_1,Pts_1.T)).T
	Pts_2 = (np.matmul(N_2,Pts_2.T)).T
	X_1 = Pts_1[:,0]
	Y_1 = Pts_1[:,1]
	X_2 = Pts_2[:,0]
	Y_2 = Pts_2[:,1]	
	col_1 = np.multiply(X_2,X_1)
	col_2 = np.multiply(X_2,Y_1)
	C_3 = X_2
	C_4 = np.multiply(Y_2,X_1)
	C_5 = np.multiply(Y_2,Y_1)
	C_6 = Y_2
	C_7 = X_1
	C_8 = Y_1
	C_9 = np.ones((np.shape(X_1)[0]))	
	A_Matrix = np.array([col_1, col_2, C_3, C_4, C_5, C_6, C_7, C_8, C_9]).T
	U,S,V = np.linalg.svd(A_Matrix)
	F_Col = V[:,-1]
	F_Matrix = np.array([[F_Col[0],F_Col[1],F_Col[2]], [F_Col[3],F_Col[4],F_Col[5]], [F_Col[6],F_Col[7],F_Col[8]]])
	U, S, V = np.linalg.svd(F_Matrix)
	S[2] = 0
	F_Matrix = np.dot(U,np.dot(np.diag(S),V.T))
	F_Matrix = np.matmul(N_1.T,np.matmul(F_Matrix,N_2))
	F_Matrix = F_Matrix/F_Matrix[2,2]

	return F_Matrix

def Draw_lines(Image_1,Image_2,Lines,Points_1,Points_2):
    Row,Column = Image_1.shape
    Image_1 = cv.cvtColor(Image_1,cv.COLOR_GRAY2BGR)
    Image_2 = cv.cvtColor(Image_2,cv.COLOR_GRAY2BGR)
    for Row,pt1,pt2 in zip(Lines,Points_1,Points_2):
        Colour = tuple(np.random.randint(0,255,3).tolist())
        X_0,Y_0 = map(int, [0, -Row[2]/Row[1] ])
        X_1,Y_1 = map(int, [Column, -(Row[2]+Row[0]*Column)/Row[1] ])
        Image_1 = cv.line(Image_1, (X_0,Y_0), (X_1,Y_1), Colour,1)
        Image_1 = cv.circle(Image_1,tuple(pt1),5,Colour,-1)
        Image_2 = cv.circle(Image_2,tuple(pt2),5,Colour,-1)
    return Image_1,Image_2

if __name__ == "__main__":

	img1 = cv.imread('AprilCalib_orgframe_00007.png',0)  #queryimage # left image
	img2 = cv.imread('AprilCalib_orgframe_00008.png',0) #trainimage # right image
	ids1, corners1, centers1, Hs1 = pyAprilTag.find(img1)
	ids2, corners2, centers2, Hs2 = pyAprilTag.find(img2)
	points2 = np.empty([40,4,2])
	col_2 = np.empty([40,2])
	ids1 = list(ids1)
	ids2 = list(ids2)
	for i in range(len(ids1)):
		idx_curr_2 = ids2.index(ids1[i])
		points2[i] = corners2[idx_curr_2]

		col_2[i,:] = centers2[idx_curr_2]

	points2_new = np.empty([200,2])
	for i in range(np.shape(points2)[0]):
		points2_new[5*i,:] = points2[i][0]
		points2_new[5*i+1,:] = points2[i][1]
		points2_new[5*i+2,:] = points2[i][2]
		points2_new[5*i+3,:] = points2[i][3]
		points2_new[5*i+4,:] = col_2[i,:]

	points1_new = np.empty([200,2])
	for i in range(np.shape(corners1)[0]):	
		points1_new[5*i,:] = corners1[i][0]
		points1_new[5*i+1,:] = corners1[i][1]
		points1_new[5*i+2,:] = corners1[i][2]
		points1_new[5*i+3,:] = corners1[i][3]
		points1_new[5*i+4,:] = centers1[i,:]
	F = F_matrix(points1_new,points2_new)
	print ("Fundamental matrix is:")
	print(F)