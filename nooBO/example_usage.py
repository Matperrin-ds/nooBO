from nooBO import *
from plot_util import *
from botorch.test_functions import Ackley, EggHolder, Branin

# Define the test function
objective_function = Ackley(dim=2, negate = True)

# Plot objective function
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111, projection='3d')
plot_surface(objective_function, objective_function.bounds[:, 0], objective_function.bounds[:, 1], "Ackley Function", ax1)
plt.tight_layout()
plt.show()

# Create the optimization problem
problem = optimization_problem(objective_function, num_dims = 2, bounds = objective_function.bounds, n_randinit_points = 10)

# Run optimization
result = problem.run(n_iter=300)

