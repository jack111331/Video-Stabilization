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
    # Get the number of frames in sequence to be stabilized
    n_frames = len(F_transforms)
    # Create parts of weight vector $c$ to cast objective in the form $c^{T}e$
    W1 = np.repeat(w1, n_frames*N)
    W2 = np.repeat(w2, n_frames*N)
    W3 = np.repeat(w3, n_frames*N)
    # Create coefficient vector of size 3*n_frames*N
    c = np.concatenate(W1, W2, W3)
    # Apply linear programming to look for optimal stabilization + re-targeting transform
    # Initialise a PuLP LP problem solver - minimizer
    prob = lpp.LpProblem("stabilize", lpp.LpMinimize)
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
