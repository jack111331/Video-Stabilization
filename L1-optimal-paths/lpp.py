from pulp import *


def stabilize(n_frames, frame_pair_transforms):
    # Apply linear programming to look for optimal stabilization + retargetting transform
    # Initialise a PuLP Lp solver minimizer
    derivative_sum = LpProblem("stabilize", LpMinimize)
    # Array to associate an index with every values(s) s \in S, variable
    values = np.arange(mdp.nstates)
    # Declare required number of slack variables
    # Convert above indexing to dictionary form for pulp solver, vars named as Vs_i
    val_dict = LpVariable.dicts("Vs", values)
    # Add objective function (sum) to solver, pulp auto recognises this
    # to be the objective because it is added first
    value_sum += lpSum([val_dict[s] for s in values]), "Sum V(s), for all s in S"
    # Add primary constraints to solver in a nested loop
    for s in range(mdp.nstates):
        # One constraint for every action, from class notes
        for a in range(mdp.nactions):
            value_sum += val_dict[s] - lpSum([mdp.f_trans[s][a][s_prime] * (
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