import numpy as np
import cv2

cap = cv2.VideoCapture('0.avi')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners=30,
                       qualityLevel=0.3,
                       minDistance=7,
                       blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors to mark onto trackers
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 are the points that have been selected to be tracked
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image to draw the trajectories
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points, using list sliced indexing
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

# # %% get paths
#
# paths = {}
# pthid = 0
# shiftidx = [0]
# for i in range(len(refreshindex)):
#     for j in range(len(trajectory_mat[refreshindex[i]])):
#         end = 0
#         if i == len(refreshindex) - 1:
#             end = len(trajectory_mat)
#         else:
#             end = refreshindex[i + 1]
#         sigpth = []
#         sigpth.append([trajectory_mat[refreshindex[i]][0][0], trajectory_mat[refreshindex[i]][0][1]])
#         for k in range(refreshindex[i] + 1, end):
#             for l in range(len(trajectory_mat[k])):
#                 if j >= len(trajectory_mat[k]):
#                     continue
#                 if trajectory_mat[k - 1][j][2] == trajectory_mat[k][l][0] and trajectory_mat[k - 1][j][3] == \
#                         trajectory_mat[k][l][1]:
#                     sigpth.append([trajectory_mat[k][l][0], trajectory_mat[k][l][1]])
#                     # print(i,j,k,l)
#         paths[pthid] = sigpth
#         pthid += 1
#     shiftidx.append(pthid)
#
# # %% get the trajectory matrix and visualize it
#
# t_mat = np.zeros((2 * len(paths), n_frames))
# for k in range(len(shiftidx) - 1):
#     for i in range(2 * shiftidx[k], 2 * shiftidx[k + 1], 2):
#         if len(paths[np.floor(i / 2)]) <= 20:
#             continue
#         for j in range(len(paths[np.floor(i / 2)])):
#             t_mat[i, j + refreshindex[k]] = paths[np.floor(i / 2)][j][0]
#             t_mat[i + 1, j + refreshindex[k]] = paths[np.floor(i / 2)][j][1]
#
# plt.figure(figsize=(30, 40))
# plt.imshow(t_mat, cmap='gray')
#
# # %% Moving Factorization
#
# W = np.zeros_like(t_mat)
# W[t_mat != 0] = 1
# r = 9
# k = 50
# m = 20
# i_frame = 0
# e_frame = i_frame + k
# while e_frame < n_frames:
#     M = t_mat[:, i_frame:e_frame]  # M(2n x k)
#     M0 = np.zeros((2 * m, k))
#     idx = 0
#     sel_idx = []
#     for i in range(0, len(t_mat), 2):
#         if ((t_mat[i, i_frame:e_frame] > 0).sum() == k):
#             M0[idx] = t_mat[i, i_frame:e_frame]
#             M0[idx + 1] = t_mat[i + 1, i_frame:e_frame]
#             sel_idx.append(idx)
#             idx += 2
#             if idx == 2 * m:
#                 break
#     U, s, V = np.linalg.svd(M0, full_matrices=False)
#     s = np.sqrt(np.diag(s))
#     C1 = U.dot(s)
#     E1 = s.dot(V)
#     M_ = np.zeros((len(t_mat) - 2 * m, k))
#     idx = 0
#     for i in range(0, len(t_mat), 2):
#         if i in sel_idx or i - 1 in sel_idx:
#             continue
#         else:
#             M_[idx] = t_mat[i, i_frame:e_frame]
#             M_[idx + 1] = t_mat[i + 1, i_frame:e_frame]
#             idx += 2
#     C_ = M_.dot((E1.T).dot(np.linalg.inv(E1.dot(E1.T))))
#
#     break
