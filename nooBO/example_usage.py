from nooBO import *
from test_functions import *

problem = optimization_problem(eggholder, num_dims = 2)
bounds = torch.stack([-512*torch.ones(2), 512*torch.ones(2)])

result = problem.run(bounds, n_iter=300)

print(result)