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
# for the lpp optimization problem, matrix multiples A and B
# m --> 2x3 B --> 3x3
def multiply_matrices(m, B):
    C =
    for i in range(3):
        for j in range(3):
            for k in range(3):


# Flattens a homogeneous transform matrix to its lpp friendly N x 1 form
def flatten_N(A):
    return [A[2][0], A[2][1], A[0][0], A[1][0], A]
# Takes im_shape, a tuple and
# crop ratio, a float < 1.0
def get_crop_window(im_shape, crop_ratio):
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


# Function that takes the number of frames $n$ and the frame pair transforms
# $F_1, F_2, \ldots, F_n$ as input and returns the stabilized
# camera trajectory parameters $\{p_t\}_{t=1}^{n}$
# These stabilized parameters are a flattened version of the transforms $B_t$
# Which can then be applied to stabilize trajectory
def stabilize(F_transforms, im_shape, crop_ratio):
    # Create lpp minimization problem object
    prob = lpp.LpProblem("stabilize", lpp.LpMinimize)
    # Get the number of frames in sequence to be stabilized
    n_frames = len(F_transforms)
    # Declare structures to be used later
    time_steps = np.arange(n_frames)
    dims = np.arange(N)
    # Slack variables for 1st derivative, all positive
    e1 = lpp.LpVariable.dicts("e1", ((i, j) for i in time_steps for j in dims), lowBound=0.0)
    # Slack variables for 2nd derivative, all positive
    e2 = lpp.LpVariable.dicts("e2", ((i, j) for i in time_steps for j in dims), lowBound=0.0)
    # Slack variables for 3rd derivative, all positive
    e3 = lpp.LpVariable.dicts("e3", ((i, j) for i in time_steps for j in dims), lowBound=0.0)
    # Stabilization parameters for each frame, all positive
    p = lpp.LpVariable.dicts("p", ((i, j) for i in time_steps for j in dims))
    # Construct objective to be minimized using e1, e2 and e3
    prob += w1 * lpp.lpSum([e1[i, j] * c1[j] for i in time_steps for j in dims]) + \
            w2 * lpp.lpSum([e2[i, j] * c2[j] for i in time_steps for j in dims]) + \
            w3 * lpp.lpSum([e3[i, j] * c3[j] for i in time_steps for j in dims])2
    # Apply smoothness constraints on the slack variables e1, e2 and e3 using params p
    for i in range(n_frames - 3):
        B_t = [p(k, 3) p(k, 5) 0; p(k, 4) p(k, 6) 0; p(k, 1) p(k, 2) 1];
        B_t1 = [p(k + 1, 3) p(k + 1, 5) 0; p(k + 1, 4) p(k + 1, 6) 0; p(k + 1, 1) p(k + 1, 2) 1];
        B_t2 = [p(k + 2, 3) p(k + 2, 5) 0; p(k + 2, 4) p(k + 2, 6) 0; p(k + 2, 1) p(k + 2, 2) 1];
        B_t3 = [p(k + 3, 3) p(k + 3, 5) 0; p(k + 3, 4) p(k + 3, 6) 0; p(k + 3, 1) p(k + 3, 2) 1];
        # Depending on in what form F_transforms come to us use raw p vectors to create smoothness constraints
        # No need to assemble p in matrix form
        res_t = F_transforms{k + 1} * B_t1 - B_t;
        res_t1 = t_transforms{k + 2} * B_t2 - B_t1;
        res_t2 = t_transforms{k + 3} * B_t3 - B_t2;

        res_t = [res_t(3, 1) res_t(3, 2) res_t(1, 1) res_t(2, 1) res_t(1, 2) res_t(2, 2)];
        res_t1 = [res_t1(3, 1) res_t1(3, 2) res_t1(1, 1) res_t1(2, 1) res_t1(1, 2) res_t1(2, 2)];
        res_t2 = [res_t2(3, 1) res_t2(3, 2) res_t2(1, 1) res_t2(2, 1) res_t2(1, 2) res_t2(2, 2)];

        # Apply the smoothness constraints on these slack variables
        for j in range(N):
            -e1(i, j) <= res_t <= e1(i, j);
            -e2(i, j) <= res_t1 - res_t <= e2(i, j);
            -e3(i, j) <= res_t2 - 2*res_t1 + res_t <= e3(i, j);
    # Apply linear programming to look for optimal stabilization + re-targeting transform

    # Create a collection of $e$ vectors of size same as $c$
    # Step 1: Assemble postfix identifiers for each of the 3*n_frames*N variables
    # Array to associate an index with every values(s) s \in S, variable
    values = np.arange(mdp.nstates)
    # Declare required number of slack variables
    # Convert above indexing to dictionary form for pulp solver, vars named as Vs_i
    val_dict = lpp.LpVariable.dicts("Vs", values)
    # Add objective function (sum) to solver, pulp auto recognises this
    # to be the objective because it is added first
    prob += lpp.lpSum([val_dict[s] for s in values]), "Sum V(s), for all s in S"
    # Get corners of decided crop window for inclusion constraints
    # The corner points are $c_i = (c_i^x, c_i^y)$
    corner_points = get_crop_window(im_shape, crop_ratio)
    # Add primary constraints to solver in a nested loop
    for s in range(mdp.nstates):
        # One constraint for every action, from class notes
        for a in range(mdp.nactions):
            value_sum += val_dict[s] - lpp.lpSum([mdp.f_trans[s][a][s_prime] * (
                    mdp.f_reward[s][a][s_prime] + mdp.gamma * val_dict[s_prime]
            ) for s_prime in values]) >= 0, "Const: Vs_{0}, action-{1}".format(s, a)
    # If the MDP is episodic, find candidate terminal states
    # May be more than one but PA2 guarantees 1
    if mdp.type == "episodic":
        term_lst = mdp.get_terminal_states()
        # Add zero value function constraint when looking
        # ahead from a terminal state
        for term_state in term_lst:
            value_sum += val_dict[term_state] == 0, "Terminal State const. for state {0}".format(term_state)
    # Print formulation to a text file
    # value_sum.writeLP("formulation.lp")
    # Invoke pulp solver
    value_sum.solve()

    # If no solution found
    if value_sum.status != 1:
        print("error")
        exit(-1)
    # init optimal values vector
    values_opt = np.zeros(mdp.nstates)
    # Before reading out converged variable values to a vector, must ensure ordering
    # assign computed optimal values to vector
    for s in range(mdp.nstates):
        # Read in pulp variable name associated with current iteration
        cur_var = value_sum.variables()[s]
        # Assign to corresponding position in values_opt
        values_opt[int(cur_var.name.split('_')[1])] = cur_var.varValue
    # Get associated policy with V^*
    pi_opt = get_max_action_value(mdp, values_opt)
    return values_opt, pi_opt
    # Return the computed transforms
    return Bt
