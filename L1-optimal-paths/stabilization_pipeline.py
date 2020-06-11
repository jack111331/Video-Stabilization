# Import numpy and OpenCV
import numpy as np
import cv2 as cv

# Read input video
cap = cv.VideoCapture('0.avi')

# Get frame count
n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print("Number of frames is {0}".format(n_frames))

# Get width and height of video stream
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv.VideoWriter_fourcc(*'MJPG')

# Get input fps, use same for output
fps = cap.get(cv.CAP_PROP_FPS)

# Set up output video stream
out = cv.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

# Pre-define transformation-store array
# Uses 3 parameters since it is purely a coordinate transform
transforms = np.zeros((n_frames - 1, 3), np.float32)

for i in range(n_frames - 2):
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

    # Find transformation matrix
    # This transform function is deprecated
    # use cv::estimateAffine2D, cv::estimateAffinePartial2D
    m = cv.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less

    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    # Store transformation
    transforms[i] = [dx, dy, da]

    # Move to next frame
    prev_gray = curr_gray

    # print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

print(transforms)