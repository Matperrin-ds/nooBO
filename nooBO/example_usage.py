from nooBO import *
from test_functions import *

problem = optimization_problem(eggholder, num_dims = 2)
bounds = torch.stack([-400*torch.ones(2), 400*torch.ones(2)])

res = problem.run(bounds, n_iter=100)

print(res)