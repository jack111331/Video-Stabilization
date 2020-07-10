import pulp as lpp
import numpy as np


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
# Lower and Upper bounds for *Proximity Constraints*
# Sequence of variables
# [a_t, b_t, c_t, d_t, (b_t + c_t), (a_t - d_t)]
prox_lb = [0.9, -0.1, -0.1, 0.9, -0.1, -0.05]
prox_ub = [1.1, 0.1, 0.1, 1.1, 0.1, 0.05]


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


# Function that takes the number of frames $n$ and the frame pair transforms
# $F_1, F_2, \ldots, F_n$ as input and returns the stabilized
# camera trajectory parameters $\{p_t\}_{t=1}^{n}$
# These stabilized parameters are a flattened version of the transforms $B_t$
# Which can then be applied to stabilize trajectory
def stabilize(F_transforms, corner_points):
    # Create lpp minimization problem object
    prob = lpp.LpProblem("stabilize", lpp.LpMinimize)
    # Get the number of frames in sequence to be stabilized
    n_frames = len(F_transforms)
    # Declare structures to be used later
    time_steps = np.arange(n_frames)
    dims = np.arange(N)
    # Slack variables for 1st derivative, all positive
    e1 = lpp.LpVariable.dicts("e1", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    # Slack variables for 2nd derivative, all positive
    e2 = lpp.LpVariable.dicts("e2", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    # Slack variables for 3rd derivative, all positive
    e3 = lpp.LpVariable.dicts("e3", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    # Stabilization parameters for each frame, all positive
    p = lpp.LpVariable.dicts("p", ((i, j) for i in range(n_frames) for j in range(N)))
    # Construct objective to be minimized using e1, e2 and e3
    prob += w1 * lpp.lpSum([e1[i, j] * c1[j] for i in time_steps for j in dims]) + \
            w2 * lpp.lpSum([e2[i, j] * c2[j] for i in time_steps for j in dims]) + \
            w3 * lpp.lpSum([e3[i, j] * c3[j] for i in time_steps for j in dims])
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
            prob += e1(t, j) >= res_t[j]
            prob += -1 * e2(t, j) <= res_t1[j] - res_t[j]
            prob += e2(t, j) >= res_t1[j] - res_t[j]
            prob += -1 * e3(t, j) <= res_t2[j] - 2*res_t1[j] + res_t[j]
            prob += e3(t, j) >= res_t2[j] - 2*res_t1[j] + res_t[j]
    # Proximity Constraints
    # Inclusion Constraints
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
    return B_transforms
