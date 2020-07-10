import numpy as np
import cv2 as cv
from lpp import stabilize


# Crop ratio used for crop window float < 1.0
crop_ratio = 0.8


# Takes im_shape, a tuple and
# crop ratio, a float < 1.0
def get_crop_window(im_shape):
    # Get center of original frames
    img_ctr_x = round(im_shape[1] / 2)
    img_ctr_y = round(im_shape[0] / 2)
    # Get the dimensions w and h of the crop window
    # Crop ratio is a float < 1.0 since the crop window
    # needs to be smaller than the raw frames
    crop_w = round(im_shape[1] * crop_ratio)
    crop_h = round(im_shape[0] * crop_ratio)
    # Get upper left corner of centered crop window
    crop_x = round(img_ctr_x - crop_w / 2)
    crop_y = round(img_ctr_y - crop_h / 2)
    # Assemble list of corner points into a list of tuples
    corner_points = [
        (crop_x, crop_y),
        (crop_x + crop_w, crop_y),
        (crop_x, crop_y + crop_h),
        (crop_x + crop_w, crop_y + crop_h)
    ]
    # Return corner points of crop window
    return corner_points


# This needs to be replaced by the matrix multiplication
# P_t = C_t B_t in the general case
def smoothen(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:, i], radius=3)

    return smoothed_trajectory


# Read input video
cap = cv.VideoCapture('0.avi')

# Get frame count
n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print("Number of frames is {0}".format(n_frames))

# Get width and height of video stream
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv.VideoWriter_fourcc(*'MPEG')

# Get input fps, use same for output
fps = int(cap.get(cv.CAP_PROP_FPS))

# Set up output video stream
out = cv.VideoWriter('video_out.avi', fourcc, fps, (2*w, h))

# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

# Pre-define transformation-store array
# Uses 3 parameters since it is purely a coordinate transform
# A collection of n_frames homography matrices
F_transforms = np.zeros((n_frames, 3, 3), np.float32)
# Initialise all transformations with Identity matrix
F_transforms[:, :, :] = np.eye(3)

for i in range(n_frames):
    # Detect feature points in previous frame
    prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                      minDistance=30, blockSize=3)
    # Read next frame
    success, curr = cap.read()
    if not success:
        break
    # Convert to grayscale
    curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    # Sanity check
    assert prev_pts.shape == curr_pts.shape
    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    # Find transformation matrix for full 6DOF affine transform
    m, _ = cv.estimateAffine2D(prev_pts, curr_pts)  # will only work with OpenCV-3 or less
    # print(m.shape) --> (2, 3) since 6 DOF full affine transform
    # Add current transformation matrix $F_t$ to array
    # $F_t$ is a right multiplied homogeneous affine transform
    F_transforms[i, :, :2] = m.T
    # Move to next frame
    prev_gray = curr_gray
    # print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
# Get corners of decided crop window for inclusion constraints
# Input: input frame shape tuple. The corner points are $c_i = (c_i^x, c_i^y)$
corner_points = get_crop_window(prev.shape)
# Get stabilization transforms B_t by processing motion transition transforms F_t
B_transforms = stabilize(F_transforms, corner_points)
# Compute accumulated trajectory in matrix form using iterative right multiplications
C_trajectory = np.zeros
# Apply computed stabilization transforms to raw camera trajectory in
# N (Number of free parameters, 6 for full affine) dimensional parameter space
P_trajectory = smooth(C_trajectory)
# Calculate difference in smoothed_trajectory and trajectory
difference = P_trajectory - C_trajectory

# Calculate newer transformation array
transforms_smooth = transforms + difference


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv.warpAffine(frame, T, (s[1], s[0]))
    return frame


# Reset stream to first frame
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Write n_frames-1 transformed frames
for i in range(n_frames - 2):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv.warpAffine(frame, m, (w, h))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)

    # Write the frame to the file
    frame_out = cv.hconcat([frame, frame_stabilized])

    # If the image is too big, resize it.
    # if frame_out.shape[1] > 1920:

    frame_out = cv.resize(frame_out, (frame_out.shape[1], frame_out.shape[0]))

    cv.imshow("Before and After", frame_out)
    cv.waitKey(10)
    out.write(frame_out)

##
# Compare x, y components of motion of camera in original and stabilized trajectories and plot
# Trajectory calculation, integrates changes to give current value of
# x, y, theta
#print(transforms.shape)
trajectory = np.cumsum(transforms, axis=0)
#print(trajectory.shape)
# print(trajectory)