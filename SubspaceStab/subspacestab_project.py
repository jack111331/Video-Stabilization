# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:19:53 2020

Subspace video stabilization :
    http://web.cecs.pdx.edu/~fliu/project/subspace_stabilization/

@author: Edge
"""
#%% Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Target Video setting
TARGET_VIDEO = './videos/4.avi'

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

# Default algorithm params
# Rank
r = 9
# window size is k
k = 50
# Take m complete trajectory
m = 20
# each roll of window
next_window_frame_delta = 5

class Trajectory:
  def _init_(self):
    self.track_list = []
    self.offset = 0

  def add_track(self, track):
    self.track_list.append(track)

  def modify_trajectory_starting_offset(self, offset):
    self.offset = offset

def construct_trajectory_matrix(cap):
  # TODO Need to be implemented
  #%% Extract first frame's feature point
  ret, previous_frame = cap.read()
  previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
  previous_feature_point = cv2.goodFeaturesToTrack(previous_frame_gray, mask = None, **feature_params)
  # Create a mask image for drawing purposes
  mask = np.zeros_like(previous_frame)
  trajectory_mat = []

  idx = 0

  #%% track points
  # intermediate trajectory_mat record each frame, each feature points' optical flow
  while(ret != False):
    ret, current_frame = cap.read()
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_feature_point = cv2.goodFeaturesToTrack(current_frame_gray, mask = None, **feature_params)

    # calculate optical flow
    # status – output status vector; each element of the vector is set to 1 if the flow for the corresponding features has moved, otherwise, it is set to 0.
    current_feature_point, status, error = cv2.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, previous_feature_point, None, **lk_params)

    # Select good points
    
    current_good_point = current_feature_point[status==1]
    previous_good_point = previous_feature_point[status==1]
    # draw the tracks
    points = []
    for i,(new,old) in enumerate(zip(current_good_point, previous_good_point)):
      a,b = new.ravel()
      c,d = old.ravel()
      points.append(np.array([c,d,a,b]))
    trajectory_mat.append(np.array(points))
    # Now update the previous frame and previous points
    previous_frame_gray = current_frame_gray.copy()
    previous_feature_point = current_good_point.reshape(-1,1,2)
    idx += 1
  # should return a trajectory matrix in following type
  # [[trajectory 1's path]
  #  [trajectory 2's path]
  # ....
  #  [trajectory n's path]]

  # shiftidx shift trajectory lots of frame
  # for example, trajectory_mat originally is
  # [
  #  [ [1, 2], [3, 4], [5, 6], []    ]
  #  [ []    , []    , []    , [7, 8] ]
  # ]
  # trajectory_mat should be compress like
  # [
  #  [ [1, 2], [3, 4], [5, 6] ]
  #  [ [7, 8] ]
  # ]
  # and trajectory_offset_index_list should be [0, 3]
  # because in second row, it start from 3rd frame
  return trajectory_list

def filter_dynamic_trajectory(trajectory_list, full_window_index_list, W, window_start_frame, window_end_frame):
  MAX_RATIO = 0.33
  total_count = [0.0] * len(full_window_index_list)
  outlier_count = [0.0] * len(full_window_index_list)
  MAX_EPI_ERROR = 2.0

  for frame in range(0, window_end_frame-next_window_frame_delta, next_window_frame_delta):
    motion_start_point_list = []
    motion_end_point_list = []
    track_index_list = []
    for fwid in range(len(full_window_index_list)):
      trajectory_index = full_window_index_list[fwid]
      trajectory_offset = trajectory_list[trajectory_index].offset
      if trajectory_offset <= frame and trajectory_offset + len(trajectory_list[trajectory_index].track_list) >= frame + next_window_frame_delta:
        motion_start_point_list.append(trajectory_list[trajectory_index].track_list[frame])
        motion_end_point_list.append(trajectory_list[trajectory_index].track_list[frame+next_window_frame_delta-1])
        track_index_list.append(fwid)

    if(len(track_index_list) < 8):
      continue

    fundamental_mat = cv2.findFoundamentalMat(motion_start_point_list, motion_end_point_list)
    if(fundamental_mat.shape[1] != 3):
      continue

    epilines_mat = cv2.computeCorrespondEpilines(motion_start_point_list, 1, fundamental_mat)
    epilines = epilines_mat.reshape(-1, 3)
    for ptid in range(track_index_list):
      epi = epilines[ptid]
      pt = np.block([motion_end_point_list[ptid], 1.0])
      total_count[track_index_list[ptid]] += 1.0
      if(epi.dot(pt) > MAX_EPI_ERROR):
        outlier_count[track_index_list[ptid]] += 1.0

  inlier_index_list = []
  for fwid in range(full_window_index_list):
    if outlier_count[fwid] / MAX_RATIO <= total_count[fwid]:
      inlier_index_list.append(full_window_index_list[fwid])
    else:
      for frame in range(window_start_frame, len(W[full_window_index_list[fwid]]))
        W[full_window_index_list[fwid]][frame] = -1

  full_window_index_list, inlier_index_list = inlier_index_list, full_window_index_list


def moving_factorization(trajectory_list, n_frames):
  # Moving factorization initial setup
  total_trajectory = len(trajectory_list)
  coefficient_mat = np.zeros((2*total_trajectory, r))
  eigen_trajectory_mat = np.zeros((r, n_frames))
  # w matrix represent missing data or not, it may changed in filter_dynamic_trajectory()
  W = np.zeros((total_trajectory, n_frames))
  full_window_index_list = []
  for tid in range(total_trajectory):
    if (trajectory_list[tid].offset == 0 and trajectory_list[tid].offset + len(trajectory_list[tid].track_list) >= k):
      full_window_index_list.append(tid)


  filter_dynamic_trajectory(trajectory_list, full_window_index_list, W, 0, k)

  A = np.zeros((2*len(full_window_index_list), k))
  for fwid in range(len(full_window_index_list)):
    trajectory_index = full_window_index_list[fwid]
    trajectory_offset = trajectory_list[trajectory_index].offset
    for frame in range(k):
      A[2*fwid : 2*fwid+2][frame] = trajectory_list[trajectory_index].track_list[frame - trajectory_offset][:]

  U, s, V = np.linalg.svd(A, full_matrices=False)
  # We just need first r rank singular value
  s = np.sqrt(np.diag(s[:r]))
  # distribute SVD's singular values' sqrt to both C and E matrix
  intermediate_C = U[:][:r].dot(s)
  eigen_trajectory_mat[:r][:k] = s.dot(V[:r][:])
  for fwid in range(len(full_window_index_list)):
    trajectory_index = full_window_index_list[fwid]
    coefficient_mat[2*trajectory_index:2*trajectory_index+2][:r] = intermediate_C[2*fwid:2*fwid+2][:r]
    for j in range(k):
      W[trajectory_index][j] = 1


  for window_start_frame in range(next_window_frame_delta, n_frames - k, next_window_frame_delta):
    window_end_frame = window_start_frame+k
    previous_full_window_index_list = [], next_full_window_index_list = []

    for tid in range(total_trajectory):
      # if this trajectory are across whole window
      if (trajectory_list[tid].offset <= window_start_frame and trajectory_list[tid].offset + len(trajectory_list[tid].track_list) >= window_end_frame):
        is_valid = True
        for l in range(window_start_frame, window_end_frame):
          if(W[tid][k] == -1):
            is_valid = False
            break

        if is_valid == False:
          continue

        if trajectory_list[tid].offset <= window_start_frame - next_window_frame_delta:
          previous_full_window_index_list.append(tid)
        else:
          next_full_window_index_list.append(tid)

    filter_dynamic_trajectory(trajectory_list, previous_full_window_index_list, W, window_start_frame, window_end_frame)
    filter_dynamic_trajectory(trajectory_list, next_full_window_index_list, W, window_start_frame, window_end_frame)

    A12 = np.zeros((2*len(previous_full_window_index_list), next_window_frame_delta))
    A2 = np.zeros((2*len(next_full_window_index_list), k))
    C1 = np.zeros((2*len(previous_full_window_index_list), r))
    A11 = np.zeros((2*len(previous_full_window_index_list), k - next_window_frame_delta))

    E1 = eigen_trajectory_mat[:][window_start_frame : window_end_frame - next_window_frame_delta]
    for fwid in len(previous_full_window_index_list):
      trajectory_index = previous_full_window_index_list[fwid]
      trajectory_offset = trajectory_list[trajectory_index].offset
      for l in range(window_end_frame - next_window_frame_delta, window_end_frame):
        A12[2*fwid : 2*fwid+2][l - window_end_frame + next_window_frame_delta] = trajectory_list[trajectory_index].track_list[l - trajectory_offset][:]

      C1[2*fwid : 2*fwid+2][:] = coefficient_mat[2*trajectory_index : 2*trajectory_index+2][:]

    for fwid in len(next_full_window_index_list):
      trajectory_index = next_full_window_index_list[fwid]
      trajectory_offset = trajectory_list[trajectory_index].offset
      for l in range(i, i+k):
        A2[2*fwid : 2*fwid+2][l - window_start_frame] = trajectory_list[trajectory_index].track_list[l - trajectory_offset][:]

    A21 = A2[0:len(A2)][0 : k - next_window_frame_delta]
    A22 = A2[0:len(A2)][k - next_window_frame_delta : k]

    EET = E1.dot(E1.T)
    EET_inv = np.linalg.inv(EET)
    C2 = A21.dot((E1.T).dot(EET_inv))

    large_C = np.block([C1, C2])
    large_A = np.block([A12, A22])

    E2 = np.linalg.inv((large_C.T).dot(large_C)).dot((large_C.T).dot(large_A))
    eigen_trajectory_mat[:][i+k-next_window_frame_delta:i+k] = E2

    for fwid in len(next_full_window_index_list):
      trajectory_index = next_full_window_index_list[fwid]
      coefficient_mat[2*trajectory_index : 2*trajectory_index+2][:] = C2[2*fwid : 2*fwid+2][:]
      for l in range(window_start_frame, window_end_frame):
        W[trajectory_index][l] = 1

  return W, coefficient_mat, eigen_trajectory_mat

def smooth_trajectory(basis, kernel_radius, kernel_sigma):
  gaussianKernel = cv2.getGaussianKernel(kernel_radius, kernel_sigma)
  return cv2.filter2D(basis, -1, gaussianKernel)

def full_frame_warping(original_t_mat, modified_t_mat, capture):
  # TODO maybe we can use cv2.remap()??
  print("Not Implemented yet")



#%% video loading
cap = cv2.VideoCapture(TARGET_VIDEO)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Is video opened?: " + str(cap.isOpened()))

# construct t_mat, the true trajectory matrix mentioned in original paper
trajectory_list = construct_trajectory_matrix(cap)  
print("All trajectory amount is", len(trajectory_list))

# construct eigen-trajectory and filter(moving factorization)
W, coefficient_mat, eigen_trajectory_mat = moving_factorization(trajectory_list, n_frames)