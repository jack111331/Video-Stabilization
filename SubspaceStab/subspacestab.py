# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:08:53 2020

Subspace video stabilization :
    http://web.cecs.pdx.edu/~fliu/project/subspace_stabilization/

@author: Abhiraj
"""
#%% Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% video loading
cap = cv2.VideoCapture('./videos/18AF.avi')
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap.isOpened())

#%% KLT Setup
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.15,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
# color = np.random.randint(0,255,(200,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
trajectory_mat = []
refreshindex = [0]
idx = 0
ret,frame = cap.read()

#%% track points

while(ret != False):
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # check for threshold number of feature points
  if(p0.shape[0]<=60):
    refreshindex.append(idx)
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    print('------------------------Refresh---------------------------')
    print(p0.shape)
    old_gray = frame_gray
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # calculate optical flow
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
  # Select good points
  
  good_new = p1[st==1]
  good_old = p0[st==1]

  # draw the tracks
  points = []
  for i,(new,old) in enumerate(zip(good_new, good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    points.append(np.array([c,d,a,b]))
  trajectory_mat.append(np.array(points))
  # Now update the previous frame and previous points
  old_gray = frame_gray.copy()
  p0 = good_new.reshape(-1,1,2)
  idx += 1
  ret,frame = cap.read()
  
#%% get paths

paths = {}
pthid = 0
shiftidx = [0]
for i in range(len(refreshindex)):
  for j in range(len(trajectory_mat[refreshindex[i]])): #all paths starting after refresh
    end = 0
    if i == len(refreshindex)-1:
      end = len(trajectory_mat)
    else:
      end = refreshindex[i+1]
    sigpth = [] # one path starting after refresh
    sigpth.append([trajectory_mat[refreshindex[i]][j][0],trajectory_mat[refreshindex[i]][j][1]])  # potential error point
    for k in range(refreshindex[i]+1,end):
      for l in range(len(trajectory_mat[k])):
        if j >= len(trajectory_mat[k]):
          continue
        if trajectory_mat[k-1][j][2] == trajectory_mat[k][l][0] and trajectory_mat[k-1][j][3] == trajectory_mat[k][l][1]:
          sigpth.append([trajectory_mat[k][l][0],trajectory_mat[k][l][1]])
          #print(i,j,k,l)
    paths[pthid] = sigpth
    pthid+=1
  shiftidx.append(pthid)    
  
#%% get the trajectory matrix and visualize it

t_mat = np.zeros((2*len(paths),n_frames))
for k in range(len(shiftidx)-1):
  for i in range(2*shiftidx[k],2*shiftidx[k+1],2):
    if len(paths[np.floor(i/2)]) <= 20:
      continue 
    for j in range(len(paths[np.floor(i/2)])):
      t_mat[i,j+refreshindex[k]] = paths[np.floor(i/2)][j][0]
      t_mat[i+1,j+refreshindex[k]] = paths[np.floor(i/2)][j][1]
      
plt.figure(figsize=(10,10))
plt.imshow(t_mat, cmap = 'gray')

#%% Moving Factorization

filtered_tmat = np.zeros_like(t_mat)
r = 9
k = 50
m = 20
delta = 5
i_frame = 0
e_frame = i_frame + k
while e_frame < n_frames:
    M = t_mat[:,i_frame:e_frame] # M(2n x k)
    W = np.zeros_like(M)
    W[M != 0] = 1
    M0 = np.zeros((2*m,k))
    idx = 0
    sel_idx = []
    for i in range(0,len(t_mat),2):
        if ((t_mat[i,i_frame:e_frame]>0).sum() == k):
            M0[idx] = t_mat[i,i_frame:e_frame]
            M0[idx+1] = t_mat[i+1,i_frame:e_frame]
            sel_idx.append(idx) 
            idx += 2
            if idx == 2*m:
                break
    U, s, V = np.linalg.svd(M0, full_matrices=False)
    s = np.sqrt(np.diag(s))
    C1 = U.dot(s)
    E1 = s.dot(V)
    M_ = np.zeros((len(t_mat)-2*m,k))
    idx = 0
    for i in range(0,len(t_mat),2):
        if i in sel_idx or i-1 in sel_idx:
            continue
        else:
            M_[idx] = t_mat[i,i_frame:e_frame]
            M_[idx+1] = t_mat[i+1,i_frame:e_frame]
            idx += 2
    C_ =  M_.dot((E1.T).dot(np.linalg.inv(E1.dot(E1.T))))
    # low pass filteing the eigen trajectories
    K = np.eye(E1.shape[1])
    Kf = np.eye(E1.shape[1])
    for i in range(E1.shape[1]-1):
        for j in range(E1.shape[1]):
            if (K[i,j]==1):
                Kf[(i+1)%50,j]=1
      
    Ef1 = np.dot(E1,Kf/2)
    Ccomplete = np.zeros(((C1.shape[0]+C_.shape[0]),C1.shape[1]))
    it1,it2 = 0,0
    for i in range(0,len(Ccomplete),2):
        if i in sel_idx:
            Ccomplete[i] = C1[it1]
            Ccomplete[i+1] = C1[it1+1]
            it1 += 2
        else:
            Ccomplete[i] = C_[it2]
            Ccomplete[i+1] = C_[it2+1]
            it2 += 2
        
    Mf = np.dot(Ccomplete,Ef1)
    
    filtered_tmat[:,i_frame:e_frame] = Mf*W
    
    i_frame += delta
    e_frame += delta
    if e_frame>t_mat.shape[1]:
        e_frame = t_mat.shape[1]
plt.figure(figsize=(10,10))
plt.imshow(filtered_tmat, cmap = 'gray') # plot filtered trajectory matrix

#%% wrap the trajectory matrix to video 
plt.figure(figsize=(10,10))
plt.imshow(filtered_tmat-t_mat, cmap = 'gray')
