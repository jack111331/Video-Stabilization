import pulp as p


# Create a LP Minimization problem
Lp_prob = p.LpProblem('Problem', p.LpMinimize)

# Create problem Variables
x = p.LpVariable("x", lowBound=0)  # Create a variable x >= 0
y = p.LpVariable("y", lowBound=0)  # Create a variable y >= 0
print(type(x))
print(type(y))
print("Value of x after declaration")
print(type(p.value(x)))
print(type(x.varValue))

# Objective Function
Lp_prob += 3 * x + 5 * y
print("Value of x after adding to problem objective")
print(type(p.value(x)))
print(type(x.varValue))
print(type(Lp_prob))
# Constraints:
Lp_prob += 2 * x + 3 * y >= 12
Lp_prob += -x + y <= 3
Lp_prob += x >= 4
Lp_prob += y <= 3
# Lp_prob += x + y == 2
# Lp_prob += x + y == 3
print("Value of x after adding constraints to problem")
print(type(p.value(x)))
print(type(x.varValue))

# Display the problem
print(Lp_prob)

status = Lp_prob.solve()  # Solver
print(p.LpStatus[status])  # The solution status

# Printing the final solution
print(p.value(x), p.value(y), p.value(Lp_prob.objective))
# These are floats
print("Final values of the variables x")
print(type(p.value(x)))
print(type(x.varValue))

