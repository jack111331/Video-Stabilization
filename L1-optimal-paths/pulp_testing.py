import numpy as np
from pulp import *


def printProb(prob):
    # Print out the parameters of the variables associated
    # with the name and varValue attributes of each
    # variable associated with the problem
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    # Print the status of the LPP solution
    print("Status:", pulp.LpStatus[prob.status])


# A list of identifiers as strings
names = ['A', 'B', 'C', 'D', 'E']
# Create a dictionary out of these identifiers
PRICES = dict(zip(names, [100.0, 99.0, 100.5, 101.5, 200.0]))
# Number of variables
n = len(names)
# Create a list of LP variables objects from the list names
x = LpVariable.dicts("e", indexs=names, lowBound=0, upBound=1)
print(type(x))
print(x)
# Specify problem name and problem type
prob = pulp.LpProblem("Example-Name", pulp.LpMinimize)
# Use the variable list x and the price dictionary to assemble the objective
prob += pulp.lpSum([x[i]*PRICES[i] for i in names])
# Doing something like prob += 5 resets the problem objective
# Statements with operators are automatically interpreted as constraints
prob += pulp.lpSum([x[i] for i in names]) == 2.0
# Compute solution
prob.solve()
# Display the solution
printProb(prob)
