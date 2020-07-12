import pulp as lpp
import numpy as np
from matplotlib import pyplot as plt


# Predefined weights, choice same as the one taken in the paper
# "Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths"
# First derivative weight
w1 = 10
# Second derivative weight
w2 = 1
# Third derivative weight
w3 = 100
# Dimension of 2D motion model considered
# 3 --> Rotation and Translation only
# 4 --> Adds scaling
# 6 --> Full affine transform adds shear and aspect ratio
N = 6
# As described in the paper the affine/rotational terms are
# weighted 100x more than the translational terms
# Format for parameter vector (dx_t, dy_t, a_t, b_t, c_t, d_t)'
# For 1st derivative
c1 = [1, 1, 100, 100, 100, 100]
# For 2nd derivative
c2 = c1
# For 3rd derivative
c3 = c1


# A multiply matrices method needed to form constraints
# for the lpp optimization problem, matrix multiples F_t and B_t
# F_t --> 3x3 np array, p_t ---> 6x1 1d collection of lpp variables
# Based on
# F_t =
# [a_t, c_t, 0]
# [b_t, d_t, 0]
# [dx_t, dy_t, 1]
# B_t =
# [p3, p5, 0]
# [p4, p6, 0]
# [p1, p2, 1]
# F_t * B_t
# [a_t * p3 + c_t * p4, a_t * p5 + c_t * p6, 0]
# [b_t * p3 + d_t * p4, b_t * p5 + d_t * p6, 0]
# [p1 + dx_t * p3 + dy_t * p4, p2 + dx_t * p5 + dy_t * p6, 1]
def transform_product(F_t, p, t):
    product = [ p[t, 0] + F_t[2, 0] * p[t, 2] + F_t[2, 1] * p[t, 3],
                p[t, 1] + F_t[2, 0] * p[t, 4] + F_t[2, 1] * p[t, 5],
                F_t[0, 0] * p[t, 2] + F_t[0, 1] * p[t, 3],
                F_t[1, 0] * p[t, 2] + F_t[1, 1] * p[t, 3],
                F_t[0, 0] * p[t, 4] + F_t[0, 1] * p[t, 5],
                F_t[1, 0] * p[t, 4] + F_t[1, 1] * p[t, 5]
            ]
    return product


# Takes im_shape, a tuple and
# crop ratio, a float < 1.0
def get_crop_window(im_shape, crop_ratio=0.8):
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
    # print(corner_points)
    # Return corner points of crop window
    return corner_points


# Function that takes the number of frames $n$ and the frame pair transforms
# $F_1, F_2, \ldots, F_n$ as input and returns the stabilized
# camera trajectory parameters $\{p_t\}_{t=1}^{n}$
# These stabilized parameters are a flattened version of the transforms $B_t$
# Which can then be applied to stabilize trajectory
def stabilize(F_transforms, frame_shape):
    # Create lpp minimization problem object
    prob = lpp.LpProblem("stabilize", lpp.LpMinimize)
    # Get the number of frames in sequence to be stabilized
    n_frames = len(F_transforms)
    # Get corners of decided crop window for inclusion constraints
    # Input: input frame shape tuple. The corner points are $c_i = (c_i^x, c_i^y)$
    corner_points = get_crop_window(frame_shape)
    # Slack variables for 1st derivative, all positive
    e1 = lpp.LpVariable.dicts("e1", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    # Slack variables for 2nd derivative, all positive
    e2 = lpp.LpVariable.dicts("e2", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    # Slack variables for 3rd derivative, all positive
    e3 = lpp.LpVariable.dicts("e3", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    # Stabilization parameters for each frame, all positive
    p = lpp.LpVariable.dicts("p", ((i, j) for i in range(n_frames) for j in range(N)))
    # Construct objective to be minimized using e1, e2 and e3
    prob += w1 * lpp.lpSum([e1[i, j] * c1[j] for i in range(n_frames) for j in range(N)]) + \
            w2 * lpp.lpSum([e2[i, j] * c2[j] for i in range(n_frames) for j in range(N)]) + \
            w3 * lpp.lpSum([e3[i, j] * c3[j] for i in range(n_frames) for j in range(N)])
    # Apply smoothness constraints on the slack variables e1, e2 and e3 using params p
    for t in range(n_frames - 3):
        # Depending on in what form F_transforms come to us use raw p vectors to create smoothness constraints
        # No need to assemble p in matrix form
        res_t_prod = transform_product(F_transforms[t + 1], p, t + 1)
        res_t1_prod = transform_product(F_transforms[t + 2], p, t + 2)
        res_t2_prod = transform_product(F_transforms[t + 3], p, t + 3)
        res_t = [res_t_prod[j] - p[t, j] for j in range(N)]
        res_t1 = [res_t1_prod[j] - p[t + 1, j] for j in range(N)]
        res_t2 = [res_t2_prod[j] - p[t + 2, j] for j in range(N)]
        # Apply the smoothness constraints on the slack variables e1, e2 and e3
        for j in range(N):
            prob += -1*e1[t, j] <= res_t[j]
            prob += e1[t, j] >= res_t[j]
            prob += -1 * e2[t, j] <= res_t1[j] - res_t[j]
            prob += e2[t, j] >= res_t1[j] - res_t[j]
            prob += -1 * e3[t, j] <= res_t2[j] - 2*res_t1[j] + res_t[j]
            prob += e3[t, j] >= res_t2[j] - 2*res_t1[j] + res_t[j]
    # Constraints
    for t1 in range(n_frames):
        # Proximity Constraints
        # [a_t, b_t, c_t, d_t, (b_t + c_t), (a_t - d_t)]
        # For the parameter $a_t$
        prob += p[t1, 2] >= 0.9
        prob += p[t1, 2] <= 1.1
        # For the parameter $b_t$
        prob += p[t1, 3] >= -0.1
        prob += p[t1, 3] <= 0.1
        # For the parameter $c_t$
        prob += p[t1, 4] >= -0.1
        prob += p[t1, 4] <= 0.1
        # For the parameter $d_t$
        prob += p[t1, 5] >= 0.9
        prob += p[t1, 5] <= 1.1
        # For $b_t + c_t$
        prob += p[t1, 3] + p[t1, 4] >= -0.1
        prob += p[t1, 3] + p[t1, 4] <= 0.1
        # For $a_t - d_t$
        prob += p[t1, 2] - p[t1, 5] >= -0.05
        prob += p[t1, 2] - p[t1, 5] <= 0.05
    #     Inclusion Constraints
    #     Based on the computation $(B_t)^{T} x [c_i^x, c_i^y]$ in homogeneous coordinates
    #     Equivalent to equation 8 in the original paper where $p_t$ is an Nx1 column vector
    #     Loop over all 4 corner points of centered crop window
        for (cx, cy) in corner_points:
            prob += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy >= 0
            prob += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy <= frame_shape[1]
            prob += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy >= 0
            prob += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy <= frame_shape[0]
    # Print formulation to a text file
    prob.writeLP("formulation.lp")
    # Apply linear programming to look for optimal stabilization + re-targeting transform
    prob.solve()
    # Pre allocate array for holding computed optimal stabilization transforms
    B_transforms = np.zeros((n_frames, 3, 3), np.float32)
    # Initialise all transformations with Identity matrix
    B_transforms[:, :, :] = np.eye(3)
    # If solution found
    if prob.status == 1:
        print("Optimal solution found")
        # Return the computed stabilization transforms
        for i in range(n_frames):
            B_transforms[i, :, :2] = np.array([[p[i, 2].varValue, p[i, 4].varValue],
                                               [p[i, 3].varValue, p[i, 5].varValue],
                                               [p[i, 0].varValue, p[i, 1].varValue]])
    else:
        print("Error")
        print("Status:", lpp.LpStatus[prob.status])
    return B_transforms


# Check if stabilization works on dummy example
if __name__ == '__main__':
    n_frames = 10
    # Create stationary (identity) camera trajectory and add random noise to it
    F_transforms = np.zeros((n_frames, 3, 3), np.float32)
    # Initialise all transformations with Identity matrix
    F_transforms[:, :, :] = np.eye(3)
    # Add zero mean random noise to transition matrices
    F_transforms += np.random.normal(0, 0.01, (10, 3, 3))
    F_transforms[0, :, :] = np.eye(3)
    # print(F_transforms)
    # Accumulate by right multiplication into C_trajectory
    # $C_{t + 1} = C_t x F_t$
    C_trajectory = F_transforms.copy()
    for i in range(1, 10):
        C_trajectory[i, :, :] = C_trajectory[i - 1, :, :] @ F_transforms[i, :, :]
    # print(C_trajectory)
    # Get stabilization transforms
    B_transforms = stabilize(F_transforms, (1024, 720))
    # print(B_transforms)
    P_trajectory = C_trajectory.copy()
    # Apply transform to C_trajectory to get P_trajectory
    for i in range(10):
        P_trajectory[i, :, :] = C_trajectory[i, :, :] @ B_transforms[i, :, :]
    # Starting coordinate (0, 0) in homogeneous system
    origin = np.array([0, 0, 1])
    # Evolution of coordinate of camera trajectory under original scheme
    evolution_og = origin @ C_trajectory
    # Evolution of origin under stabilized trajectory
    evolution_stab = origin @ P_trajectory
    plt.figure()
    plt.plot(evolution_og[:, 1])
    plt.plot(evolution_stab[:, 1])
    plt.title('Original vs Stab x')
    plt.show()
    plt.close()
