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
cap = cv2.VideoCapture('./videos/0.avi')
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
color = np.random.randint(0,255,(200,3))
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
  for j in range(len(trajectory_mat[refreshindex[i]])):
    end = 0
    if i == len(refreshindex)-1:
      end = len(trajectory_mat)
    else:
      end = refreshindex[i+1]
    sigpth = []
    sigpth.append([trajectory_mat[refreshindex[i]][0][0],trajectory_mat[refreshindex[i]][0][1]])  
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
      
plt.figure(figsize=(30,40))
plt.imshow(t_mat, cmap = 'gray')

#%% Moving Factorization

